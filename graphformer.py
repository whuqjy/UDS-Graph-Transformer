"""Hydraulic-prior-guided Graph Transformer for sewer-network inference.

Requires NumPy and PyTorch. Preprocess inputs with Dataloader.py first.
All node arrays and graph tensors must use the same node ordering.
Each graph uses fixed topology; scenario IDs select hydraulic edge features.

Example:
    python Graphformer.py --data-dir data/processed --graph-dir data/graph \
        --output-dir runs/kmmi --selection-split val

Graph directory: in_degree.npy [N], out_degree.npy [N], spatial_pos.npy [N, N].
Data directory, for each split: X_<split>.npy [M, N, F], Y_<split>.npy
[M, N, T], scenario_<split>.npy [M], and edge_input_norm_list_<split>.npz
with consecutive edge_0, edge_1, ... keys and arrays [N, N, K, E].

Use a validation split for scheduling and checkpoint selection. Explicitly
passing --selection-split test reproduces the original split usage; that
split then participates in model selection and is not an independent test.
This script does not create or infer a validation split.

Default run settings match the supplied script: 4 layers, width 64, 4 heads,
FFN width 128, dropout 0.1, 5 hops, batch size 8, 200 epochs, AdamW lr 1e-3
and weight decay 1e-5. Losses and predictions are in normalized depth units.
The MSE includes all nodes and target steps, including observed nodes.

Checkpoints store state_dict plus model configuration, not pickled models:
    checkpoint = torch.load("runs/kmmi/best.pt", map_location="cpu", weights_only=True)
    model = SewerGraphEncoder(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

Learned parameter names (including Prediction) are retained for state_dict
compatibility. Importing this module does not load data or start training.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def get_activation_fn(activation_fn):
    """Resolve the activation used by the feed-forward blocks."""
    if activation_fn == "relu":
        return F.relu
    elif activation_fn == "gelu":
        return F.gelu
    elif activation_fn == "tanh":
        return torch.tanh
    else:
        raise ValueError(f"Unsupported activation function: {activation_fn}")


class SewerGraphAttention(nn.Module):
    """Multi-head attention with additive topology and hydraulic biases.
    x: [B, N, D]; bias: [N, N], [H, N, N], or [B, H, N, N].
    Padding masks mark excluded keys with True. Attention masks are additive
    logits (use -inf to exclude a pair), not Boolean masks. Output: [B, N, D]."""

    def __init__(self, embed_dim, num_heads, dropout=0.2):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.Dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(
        self,
        x,
        attn_bias=None,
        padding_mask=None,
        attn_mask=None,
    ):
        B, N, D = x.shape
        H = self.num_heads
        Dh = self.head_dim
        assert D == self.embed_dim, (
            f"Expected x feature dimension embed_dim={self.embed_dim}, got {D}"
        )
        # Split projected features into attention heads: [B, H, N, Dh].
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.view(B, N, H, Dh).transpose(1, 2)
        k = k.view(B, N, H, Dh).transpose(1, 2)
        v = v.view(B, N, H, Dh).transpose(1, 2)

        # Add graph priors to scaled dot-product logits before softmax.
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attn_bias is not None:
            if attn_bias.dim() == 2:
                attn_bias = attn_bias.unsqueeze(0).unsqueeze(0)
            elif attn_bias.dim() == 3:
                attn_bias = attn_bias.unsqueeze(0)
            elif attn_bias.dim() == 4:
                pass
            else:
                raise ValueError(
                    f"Expected attn_bias rank 2, 3, or 4; got {attn_bias.dim()}"
                )
            attn_logits = attn_logits + attn_bias
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
            elif attn_mask.dim() == 4:
                pass
            else:
                raise ValueError(
                    f"Expected attn_mask rank 2, 3, or 4; got {attn_mask.dim()}"
                )
            attn_logits = attn_logits + attn_mask
        if padding_mask is not None:
            assert padding_mask.shape == (B, N), (
                f"Expected padding_mask [B, N] = {(B, N)}, "
                f"got {padding_mask.shape}"
            )
            key_padding_mask = padding_mask[:, None, None, :]
            attn_logits = attn_logits.masked_fill(
                key_padding_mask,
                float("-inf"),
            )

        attn_weights = torch.softmax(attn_logits, dim=-1)
        attn_weights = self.Dropout(attn_weights)
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)
        return out


class SewerNodeFeature(nn.Module):
    """Project continuous node inputs and add in/out-degree embeddings.
    Inputs: x [B, N, F], in_degree/out_degree [B, N].
    Output: node embeddings [B, N, D]."""

    def __init__(
        self,
        raw_node_feat_dim,
        num_in_degree,
        num_out_degree,
        hidden_dim,
    ):
        super().__init__()
        self.raw_node_feat_dim = raw_node_feat_dim
        self.hidden_dim = hidden_dim
        self.node_encoder = nn.Linear(raw_node_feat_dim, hidden_dim)
        self.in_degree_encoder = nn.Embedding(
            num_in_degree,
            hidden_dim,
        )
        self.out_degree_encoder = nn.Embedding(
            num_out_degree,
            hidden_dim,
        )

    def forward(self, batched_data):
        x = batched_data["x"]
        in_degree = batched_data["in_degree"]
        out_degree = batched_data["out_degree"]
        B, N, F_node = x.shape
        assert F_node == self.raw_node_feat_dim, (
            f"Expected x feature dimension raw_node_feat_dim={self.raw_node_feat_dim}, "
            f"got {F_node}"
        )
        # Continuous input projection plus structural degree embeddings.
        node_feature = self.node_encoder(x)
        node_feature = (
            node_feature
            + self.in_degree_encoder(in_degree.long())
            + self.out_degree_encoder(out_degree.long())
        )
        return node_feature


class SewerGraphAttnBias(nn.Module):
    """Combine shortest-path embeddings and continuous multi-hop edge features.
    Inputs: attn_bias/spatial_pos [B, N, N], edge_input [B, N, N, K, E].
    Output: per-head attention bias [B, H, N, N].
    Spatial indices are passed directly to the embedding. Path averaging
    preserves the original Graphormer convention: replace 0 by 1, subtract
    1 from indices greater than 1, then clamp to the configured hop limit.
    Verify that spatial_pos.npy uses this encoding; raw distances are not
    converted by this script."""

    def __init__(
        self,
        num_heads,
        num_spatial,
        num_edge_dis,
        edge_feat_dim,
        multi_hop_max_dist,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_spatial = num_spatial
        self.num_edge_dis = num_edge_dis
        self.edge_feat_dim = edge_feat_dim
        self.multi_hop_max_dist = multi_hop_max_dist
        self.edge_encoder = nn.Linear(
            edge_feat_dim,
            num_heads,
            bias=False,
        )
        self.edge_dis_encoder = nn.Embedding(
            num_edge_dis * num_heads * num_heads,
            1,
        )
        self.spatial_pos_encoder = nn.Embedding(
            num_spatial,
            num_heads,
        )

    def forward(self, batched_data):
        attn_bias = batched_data["attn_bias"]
        spatial_pos = batched_data["spatial_pos"]
        x = batched_data["x"]
        edge_input = batched_data["edge_input"]
        n_graph, n_node = x.size()[:2]
        assert attn_bias.shape == (n_graph, n_node, n_node), (
            f"Expected attn_bias [B, N, N] = {(n_graph, n_node, n_node)}, "
            f"got {attn_bias.shape}"
        )
        assert spatial_pos.shape == (n_graph, n_node, n_node), (
            f"Expected spatial_pos [B, N, N] = {(n_graph, n_node, n_node)}, "
            f"got {spatial_pos.shape}"
        )
        assert edge_input.dim() == 5, (
            f"Expected edge_input [B, N, N, max_dist, edge_feat_dim]; "
            f"got rank {edge_input.dim()}"
        )
        assert edge_input.shape[0] == n_graph
        assert edge_input.shape[1] == n_node
        assert edge_input.shape[2] == n_node
        assert edge_input.shape[-1] == self.edge_feat_dim
        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(
            1,
            self.num_heads,
            1,
            1,
        )
        spatial_pos = spatial_pos.long()
        spatial_pos_bias = self.spatial_pos_encoder(spatial_pos)
        spatial_pos_bias = spatial_pos_bias.permute(0, 3, 1, 2)
        graph_attn_bias = graph_attn_bias + spatial_pos_bias
        # Decode the original distance indices before averaging over path hops.
        spatial_pos_ = spatial_pos.clone()
        spatial_pos_[spatial_pos_ == 0] = 1
        spatial_pos_ = torch.where(
            spatial_pos_ > 1,
            spatial_pos_ - 1,
            spatial_pos_,
        )
        if self.multi_hop_max_dist > 0:
            spatial_pos_ = spatial_pos_.clamp(
                min=1,
                max=self.multi_hop_max_dist,
            )
            edge_input = edge_input[
                :,
                :,
                :,
                : self.multi_hop_max_dist,
                :,
            ]
        edge_input = self.edge_encoder(edge_input)
        max_dist = edge_input.size(-2)
        if max_dist > self.num_edge_dis:
            raise ValueError(
                f"max_dist={max_dist} exceeds num_edge_dis={self.num_edge_dis}; "
                f"increase num_edge_dis or reduce multi_hop_max_dist"
            )
        # Apply a learned H x H mixing matrix separately at each hop.
        edge_input_flat = edge_input.permute(3, 0, 1, 2, 4).reshape(
            max_dist,
            -1,
            self.num_heads,
        )
        edge_dis_weight = self.edge_dis_encoder.weight.reshape(
            self.num_edge_dis,
            self.num_heads,
            self.num_heads,
        )[:max_dist, :, :]
        edge_input_flat = torch.bmm(
            edge_input_flat,
            edge_dis_weight,
        )
        edge_input = edge_input_flat.reshape(
            max_dist,
            n_graph,
            n_node,
            n_node,
            self.num_heads,
        ).permute(1, 2, 3, 0, 4)
        # Average the hop contributions using the decoded path length.
        edge_input = edge_input.sum(-2) / spatial_pos_.float().unsqueeze(-1)
        edge_input = edge_input.permute(0, 3, 1, 2)
        graph_attn_bias = graph_attn_bias + edge_input
        return graph_attn_bias


class GraphormerGraphEncoderLayer(nn.Module):
    """Attention and feed-forward residual blocks with configurable layer normalization."""

    def __init__(
        self,
        embedding_dim=128,
        ffn_embedding_dim=512,
        num_attention_heads=8,
        dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.1,
        activation_fn="gelu",
        pre_layernorm=True,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.pre_layernorm = pre_layernorm
        self.self_attn = SewerGraphAttention(
            embed_dim=embedding_dim,
            num_heads=num_attention_heads,
            dropout=attention_dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, embedding_dim)
        self.final_layer_norm = nn.LayerNorm(embedding_dim)
        self.Dropout = nn.Dropout(dropout)
        self.activation_dropout = nn.Dropout(activation_dropout)
        self.activation_fn = get_activation_fn(activation_fn)

    def forward(
        self,
        x,
        self_attn_bias=None,
        self_attn_padding_mask=None,
        self_attn_mask=None,
    ):
        residual = x
        if self.pre_layernorm:
            x = self.self_attn_layer_norm(x)
        x = self.self_attn(
            x,
            attn_bias=self_attn_bias,
            padding_mask=self_attn_padding_mask,
            attn_mask=self_attn_mask,
        )
        x = self.Dropout(x)
        x = residual + x
        if not self.pre_layernorm:
            x = self.self_attn_layer_norm(x)
        residual = x
        if self.pre_layernorm:
            x = self.final_layer_norm(x)
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.activation_dropout(x)
        x = self.fc2(x)
        x = self.Dropout(x)
        x = residual + x
        if not self.pre_layernorm:
            x = self.final_layer_norm(x)
        return x


class SewerGraphEncoder(nn.Module):
    """Predict all target time steps at every node: [B, N, F] -> [B, N, T]."""

    def __init__(
        self,
        raw_node_feat_dim,
        num_in_degree,
        num_out_degree,
        edge_feat_dim,
        num_spatial,
        num_edge_dis,
        multi_hop_max_dist,
        num_encoder_layers=4,
        embedding_dim=128,
        ffn_embedding_dim=512,
        num_attention_heads=8,
        dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.1,
        activation_fn="gelu",
        encoder_normalize_before=True,
        pre_layernorm=True,
        horizon=6,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_encoder_layers = num_encoder_layers
        self.graph_node_feature = SewerNodeFeature(
            raw_node_feat_dim=raw_node_feat_dim,
            num_in_degree=num_in_degree,
            num_out_degree=num_out_degree,
            hidden_dim=embedding_dim,
        )
        self.graph_attn_bias = SewerGraphAttnBias(
            num_heads=num_attention_heads,
            num_spatial=num_spatial,
            num_edge_dis=num_edge_dis,
            edge_feat_dim=edge_feat_dim,
            multi_hop_max_dist=multi_hop_max_dist,
        )
        self.emb_layer_norm = (
            nn.LayerNorm(embedding_dim)
            if encoder_normalize_before
            else None
        )
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                GraphormerGraphEncoderLayer(
                    embedding_dim=embedding_dim,
                    ffn_embedding_dim=ffn_embedding_dim,
                    num_attention_heads=num_attention_heads,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    pre_layernorm=pre_layernorm,
                )
                for _ in range(num_encoder_layers)
            ]
        )
        self.final_layer_norm = (
            nn.LayerNorm(embedding_dim)
            if pre_layernorm
            else None
        )
        self.Prediction = nn.Linear(embedding_dim, horizon)

    def forward(self, batched_data):
        """Encode node inputs using a shared graph bias, then predict [B, N, T]."""
        padding_mask = batched_data.get("padding_mask")
        x = self.graph_node_feature(batched_data)
        attn_bias = self.graph_attn_bias(batched_data)
        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(
                x,
                self_attn_bias=attn_bias,
                self_attn_padding_mask=padding_mask,
            )
        if self.final_layer_norm is not None:
            x = self.final_layer_norm(x)
        return self.Prediction(x)


class MultiScenarioSewerDataset(Dataset):
    """Store normalized node features, targets, and zero-based scenario IDs."""

    def __init__(
        self,
        X,
        Y,
        scenario_id,
    ):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.scenario_id = torch.tensor(scenario_id, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return {
            "x": self.X[idx],
            "y": self.Y[idx],
            "scenario_id": self.scenario_id[idx],
        }


def build_static_graph_data(
    edge_input_list,
    in_degree,
    out_degree,
    spatial_pos,
    multi_hop_max_dist,
    device,
):
    """Cache graph tensors on the device, with edges [S, N, N, K, E]."""
    # [scenarios, nodes, nodes, retained hops, edge features]
    edge_input_all = np.stack(
        [
            e[:, :, :multi_hop_max_dist, :].astype(np.float32, copy=False)
            for e in edge_input_list
        ],
        axis=0,
    )
    in_degree = torch.from_numpy(in_degree.astype(np.int64, copy=False)).to(device)
    out_degree = torch.from_numpy(out_degree.astype(np.int64, copy=False)).to(device)
    spatial_pos = torch.from_numpy(spatial_pos.astype(np.int64, copy=False)).to(device)
    edge_input_all = torch.from_numpy(edge_input_all).to(device)
    N = spatial_pos.shape[0]
    attn_bias = torch.zeros(N, N, dtype=torch.float32, device=device)
    static_graph_data = {
        "in_degree": in_degree,
        "out_degree": out_degree,
        "spatial_pos": spatial_pos,
        "attn_bias": attn_bias,
        "edge_input_all": edge_input_all,
    }
    return static_graph_data


def make_batched_data_on_device(batch, static_graph_data, device):
    """Move a sample batch and select the edge tensor for each scenario."""
    x = batch["x"].to(device, non_blocking=True)
    y = batch["y"].to(device, non_blocking=True)
    scenario_id = batch["scenario_id"].to(device, non_blocking=True)
    B = x.shape[0]
    # Each sample receives its own scenario-specific hydraulic edge tensor.
    edge_input = static_graph_data["edge_input_all"].index_select(
        0,
        scenario_id,
    )
    batched_data = {
        "x": x,
        "in_degree": static_graph_data["in_degree"].unsqueeze(0).expand(B, -1),
        "out_degree": static_graph_data["out_degree"].unsqueeze(0).expand(B, -1),
        "attn_bias": static_graph_data["attn_bias"].unsqueeze(0).expand(B, -1, -1),
        "spatial_pos": static_graph_data["spatial_pos"].unsqueeze(0).expand(B, -1, -1),
        "edge_input": edge_input,
    }
    return batched_data, y, scenario_id


def train_one_epoch(model, train_loader, optimizer, criterion, device, static_graph_data):
    """Train with MSE and gradient clipping; return sample-weighted mean loss."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch in train_loader:
        batched_data, y, scenario_id = make_batched_data_on_device(
            batch,
            static_graph_data,
            device,
        )
        pred = model(batched_data)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        batch_size = y.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size
    return total_loss / total_samples


@torch.no_grad()
def evaluate(model, data_loader, criterion, device, static_graph_data, return_pred=False):
    """Evaluate without gradients; optionally return predictions, targets, and IDs.
    Loss and returned arrays are in normalized units. Without return_pred,
    the last three return values are empty lists, preserving the original API."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_pred = []
    all_y = []
    all_scenario = []
    for batch in data_loader:
        batched_data, y, scenario_id = make_batched_data_on_device(
            batch,
            static_graph_data,
            device,
        )
        pred = model(batched_data)
        loss = criterion(pred, y)
        batch_size = y.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        if return_pred:
            all_pred.append(pred.detach().cpu())
            all_y.append(y.detach().cpu())
            all_scenario.append(scenario_id.detach().cpu())
    avg_loss = total_loss / total_samples
    if return_pred:
        all_pred = torch.cat(all_pred, dim=0)
        all_y = torch.cat(all_y, dim=0)
        all_scenario = torch.cat(all_scenario, dim=0)
        return avg_loss, all_pred, all_y, all_scenario
    return avg_loss, all_pred, all_y, all_scenario


def load_edge_input_npz(path):
    """Load scenario edge tensors in numeric scenario-ID order."""
    with np.load(path) as data:
        expected = [f"edge_{i}" for i in range(len(data.files))]
        if not expected or set(data.files) != set(expected):
            raise ValueError(f"{path}: expected consecutive edge_0, edge_1, ... keys.")
        return [data[key].astype(np.float32) for key in expected]


def load_split(data_dir, split, graph, max_hops, device, batch_size, shuffle):
    """Load preprocessed arrays and check their alignment with the fixed graph."""
    x = np.load(data_dir / f"X_{split}.npy")
    y = np.load(data_dir / f"Y_{split}.npy")
    ids = np.load(data_dir / f"scenario_{split}.npy")
    edges = load_edge_input_npz(data_dir / f"edge_input_norm_list_{split}.npz")
    n_nodes = graph["in_degree"].shape[0]
    if (x.ndim != 3 or y.ndim != 3 or x.shape[:2] != y.shape[:2]
            or x.shape[0] == 0 or x.shape[1] != n_nodes
            or x.shape[2] == 0 or y.shape[2] == 0):
        raise ValueError(f"{split}: expected nonempty, aligned X/Y [samples, nodes, features].")
    if (ids.shape != (len(x),) or not np.issubdtype(ids.dtype, np.integer)
            or ids.min() < 0 or ids.max() >= len(edges)):
        raise ValueError(f"{split}: invalid scenario IDs for the supplied edge archive.")
    if any(e.ndim != 4 or e.shape[:2] != (n_nodes, n_nodes)
           or e.shape[2] == 0 or e.shape[3] == 0 for e in edges):
        raise ValueError(f"{split}: expected edges [nodes, nodes, hops, features].")
    if any(e.shape != edges[0].shape for e in edges):
        raise ValueError(f"{split}: all scenario edge tensors must have the same shape.")
    if not all(np.isfinite(a).all() for a in [x, y, *edges]):
        raise ValueError(f"{split}: inputs contain NaN or infinite values.")
    dataset = MultiScenarioSewerDataset(x, y, ids)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        pin_memory=(device.type == "cuda"))
    static = build_static_graph_data(edges, **graph,
                                    multi_hop_max_dist=max_hops, device=device)
    dimensions = (x.shape[-1], y.shape[-1], edges[0].shape[-1])
    return loader, static, dimensions


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-split", choices=("val", "test"), required=True,
                        help="Split used for scheduling and best-checkpoint selection.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0, etc.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional NumPy/PyTorch seed; GPU determinism is not guaranteed.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--ffn-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--attention-dropout", type=float, default=0.1)
    parser.add_argument("--activation-dropout", type=float, default=0.1)
    parser.add_argument("--max-hops", type=int, default=5)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.layers, args.embedding_dim,
           args.ffn_dim, args.heads, args.max_hops) <= 0:
        parser.error("Epochs, batch size, dimensions, heads, layers, and hops must be positive.")
    if args.embedding_dim % args.heads:
        parser.error("--embedding-dim must be divisible by --heads.")
    if not all(0 <= p < 1 for p in
               (args.dropout, args.attention_dropout, args.activation_dropout)):
        parser.error("Dropout probabilities must lie in [0, 1).")
    if (not math.isfinite(args.learning_rate) or args.learning_rate <= 0
            or not math.isfinite(args.weight_decay) or args.weight_decay < 0):
        parser.error("Learning rate must be positive and weight decay nonnegative and finite.")
    if args.seed is not None:
        if not 0 <= args.seed < 2**32:
            parser.error("--seed must lie in [0, 2**32).")
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else args.device
    )
    if args.selection_split == "test":
        print("Selection uses the test split, matching the original script. "
              "This split is not an independent test set.", flush=True)

    graph = {key: np.load(args.graph_dir / f"{key}.npy")
             for key in ("in_degree", "out_degree", "spatial_pos")}
    n_nodes = graph["in_degree"].size
    if (n_nodes == 0 or graph["in_degree"].shape != (n_nodes,)
            or graph["out_degree"].shape != (n_nodes,)
            or graph["spatial_pos"].shape != (n_nodes, n_nodes)):
        raise ValueError("Expected degree arrays [N] and spatial_pos [N, N].")
    if any(not np.issubdtype(a.dtype, np.integer) or a.min() < 0 for a in graph.values()):
        raise ValueError("Graph indices must be nonnegative integers.")
    train_loader, static_train, dimensions = load_split(
        args.data_dir, "train", graph, args.max_hops, device, args.batch_size, True
    )
    selection_loader, static_selection, selection_dimensions = load_split(
        args.data_dir, args.selection_split, graph, args.max_hops,
        device, args.batch_size, False
    )
    if selection_dimensions != dimensions:
        raise ValueError("Training and selection splits have incompatible feature dimensions.")
    model_config = dict(
        raw_node_feat_dim=dimensions[0], horizon=dimensions[1], edge_feat_dim=dimensions[2],
        num_in_degree=int(graph["in_degree"].max()) + 1,
        num_out_degree=int(graph["out_degree"].max()) + 1,
        num_spatial=int(graph["spatial_pos"].max()) + 1,
        num_edge_dis=args.max_hops, multi_hop_max_dist=args.max_hops,
        num_encoder_layers=args.layers, embedding_dim=args.embedding_dim,
        ffn_embedding_dim=args.ffn_dim, num_attention_heads=args.heads,
        dropout=args.dropout, attention_dropout=args.attention_dropout,
        activation_dropout=args.activation_dropout, activation_fn="gelu",
        encoder_normalize_before=True, pre_layernorm=True,
    )
    model = SewerGraphEncoder(**model_config).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {key: str(value) if isinstance(value, Path) else value
                  for key, value in vars(args).items()}
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump({"run": run_config, "model": model_config}, file, indent=2)

    best_loss = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, static_train
        )
        selection_loss, _, _, _ = evaluate(
            model, selection_loader, criterion, device, static_selection
        )
        if not math.isfinite(train_loss) or not math.isfinite(selection_loss):
            raise RuntimeError("Non-finite loss; check data, graph encoding, and learning rate.")
        scheduler.step(selection_loss)
        history.append({"epoch": epoch, "train_mse": train_loss,
                        "selection_mse": selection_loss,
                        "learning_rate": optimizer.param_groups[0]["lr"]})
        if selection_loss < best_loss:
            best_loss = selection_loss
            torch.save({"model_state_dict": model.state_dict(),
                        "model_config": model_config, "epoch": epoch,
                        "selection_split": args.selection_split,
                        "selection_mse": best_loss}, args.output_dir / "best.pt")
        with (args.output_dir / "history.json").open("w", encoding="utf-8") as file:
            json.dump(history, file, indent=2)
        print(f"Epoch {epoch:03d} | Train MSE: {train_loss:.6f} | "
              f"{args.selection_split} MSE: {selection_loss:.6f} | Best: {best_loss:.6f}",
              flush=True)


if __name__ == "__main__":
    main()
