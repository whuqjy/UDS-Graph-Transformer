"""Evaluate a pretrained Graph Transformer on a single rainfall event.

Input data contain rainfall, node water depths, graph connectivity, and
training-set normalization statistics. The water-depth file is used both
for observed-sensor history and for evaluation targets.

Example:
    python test.py --event example_event --observed-nodes 5,3,2,242,76,232,98,126,4

Only load model checkpoints obtained from a trusted source.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

EDGE_FEATURE_COLUMNS = (
    "edge_type_id", "shape_id", "maxdepth", "length", "roughness",
    "in_offset", "out_offset", "slope", "hydraulic_radius",
    "velocity_index", "retention_time",
)
HISTORY_LENGTH = 3
TARGET_LENGTH = 9  # Three historical and six future time steps.
RAIN_FEATURES = 9  # Rainfall over the complete nine-step target window.
HOP_LIMIT = 5

def get_activation_fn(activation_fn):
    if activation_fn == 'relu':
        return F.relu
    elif activation_fn == 'gelu':
        return F.gelu
    elif activation_fn == 'tanh':
        return torch.tanh
    else:
        raise ValueError(f'Unsupported activation function: {activation_fn}')

class SewerGraphAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0.2):
        super().__init__()
        assert embed_dim % num_heads == 0, 'Invalid model input or configuration.'
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

    def forward(self, x, attn_bias=None, padding_mask=None, attn_mask=None):
        B, N, D = x.shape
        H = self.num_heads
        Dh = self.head_dim
        assert D == self.embed_dim, f'Invalid model input or configuration.{self.embed_dim}Invalid model input or configuration.{D}'
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.view(B, N, H, Dh).transpose(1, 2)
        k = k.view(B, N, H, Dh).transpose(1, 2)
        v = v.view(B, N, H, Dh).transpose(1, 2)
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attn_bias is not None:
            if attn_bias.dim() == 2:
                attn_bias = attn_bias.unsqueeze(0).unsqueeze(0)
            elif attn_bias.dim() == 3:
                attn_bias = attn_bias.unsqueeze(0)
            elif attn_bias.dim() == 4:
                pass
            else:
                raise ValueError(f'Invalid model input or configuration.{attn_bias.dim()}')
            attn_logits = attn_logits + attn_bias
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
            elif attn_mask.dim() == 4:
                pass
            else:
                raise ValueError(f'Invalid model input or configuration.{attn_mask.dim()}')
            attn_logits = attn_logits + attn_mask
        if padding_mask is not None:
            assert padding_mask.shape == (B, N), f'Invalid model input or configuration.{(B, N)}Invalid model input or configuration.{padding_mask.shape}'
            key_padding_mask = padding_mask[:, None, None, :]
            attn_logits = attn_logits.masked_fill(key_padding_mask, float('-inf'))
        attn_weights = torch.softmax(attn_logits, dim=-1)
        attn_weights = self.Dropout(attn_weights)
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)
        return out

class SewerNodeFeature(nn.Module):

    def __init__(self, raw_node_feat_dim, num_in_degree, num_out_degree, hidden_dim):
        super().__init__()
        self.raw_node_feat_dim = raw_node_feat_dim
        self.hidden_dim = hidden_dim
        self.node_encoder = nn.Linear(raw_node_feat_dim, hidden_dim)
        self.in_degree_encoder = nn.Embedding(num_in_degree, hidden_dim)
        self.out_degree_encoder = nn.Embedding(num_out_degree, hidden_dim)

    def forward(self, batched_data):
        x = batched_data['x']
        in_degree = batched_data['in_degree']
        out_degree = batched_data['out_degree']
        B, N, F_node = x.shape
        assert F_node == self.raw_node_feat_dim, f'Invalid model input or configuration.{self.raw_node_feat_dim}Invalid model input or configuration.{F_node}'
        node_feature = self.node_encoder(x)
        node_feature = node_feature + self.in_degree_encoder(in_degree.long()) + self.out_degree_encoder(out_degree.long())
        return node_feature

class SewerGraphAttnBias(nn.Module):

    def __init__(self, num_heads, num_spatial, num_edge_dis, edge_feat_dim, multi_hop_max_dist):
        super().__init__()
        self.num_heads = num_heads
        self.num_spatial = num_spatial
        self.num_edge_dis = num_edge_dis
        self.edge_feat_dim = edge_feat_dim
        self.multi_hop_max_dist = multi_hop_max_dist
        self.edge_encoder = nn.Linear(edge_feat_dim, num_heads, bias=False)
        self.edge_dis_encoder = nn.Embedding(num_edge_dis * num_heads * num_heads, 1)
        self.spatial_pos_encoder = nn.Embedding(num_spatial, num_heads)

    def forward(self, batched_data):
        attn_bias = batched_data['attn_bias']
        spatial_pos = batched_data['spatial_pos']
        x = batched_data['x']
        edge_input = batched_data['edge_input']
        n_graph, n_node = x.size()[:2]
        assert attn_bias.shape == (n_graph, n_node, n_node), f'Invalid model input or configuration.{(n_graph, n_node, n_node)}Invalid model input or configuration.{attn_bias.shape}'
        assert spatial_pos.shape == (n_graph, n_node, n_node), f'Invalid model input or configuration.{(n_graph, n_node, n_node)}Invalid model input or configuration.{spatial_pos.shape}'
        assert edge_input.dim() == 5, f'Invalid model input or configuration.{edge_input.dim()}'
        assert edge_input.shape[0] == n_graph
        assert edge_input.shape[1] == n_node
        assert edge_input.shape[2] == n_node
        assert edge_input.shape[-1] == self.edge_feat_dim
        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        spatial_pos = spatial_pos.long()
        spatial_pos_bias = self.spatial_pos_encoder(spatial_pos)
        spatial_pos_bias = spatial_pos_bias.permute(0, 3, 1, 2)
        graph_attn_bias = graph_attn_bias + spatial_pos_bias
        spatial_pos_ = spatial_pos.clone()
        spatial_pos_[spatial_pos_ == 0] = 1
        spatial_pos_ = torch.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)
        if self.multi_hop_max_dist > 0:
            spatial_pos_ = spatial_pos_.clamp(min=1, max=self.multi_hop_max_dist)
            edge_input = edge_input[:, :, :, :self.multi_hop_max_dist, :]
        edge_input = self.edge_encoder(edge_input)
        max_dist = edge_input.size(-2)
        if max_dist > self.num_edge_dis:
            raise ValueError(f'max_dist={max_dist}Invalid model input or configuration.{self.num_edge_dis}Invalid model input or configuration.')
        edge_input_flat = edge_input.permute(3, 0, 1, 2, 4).reshape(max_dist, -1, self.num_heads)
        edge_dis_weight = self.edge_dis_encoder.weight.reshape(self.num_edge_dis, self.num_heads, self.num_heads)[:max_dist, :, :]
        edge_input_flat = torch.bmm(edge_input_flat, edge_dis_weight)
        edge_input = edge_input_flat.reshape(max_dist, n_graph, n_node, n_node, self.num_heads).permute(1, 2, 3, 0, 4)
        edge_input = edge_input.sum(-2) / spatial_pos_.float().unsqueeze(-1)
        edge_input = edge_input.permute(0, 3, 1, 2)
        graph_attn_bias = graph_attn_bias + edge_input
        return graph_attn_bias

class GraphormerGraphEncoderLayer(nn.Module):

    def __init__(self, embedding_dim=128, ffn_embedding_dim=512, num_attention_heads=8, dropout=0.1, attention_dropout=0.1, activation_dropout=0.1, activation_fn='gelu', pre_layernorm=True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.pre_layernorm = pre_layernorm
        self.self_attn = SewerGraphAttention(embed_dim=embedding_dim, num_heads=num_attention_heads, dropout=attention_dropout)
        self.self_attn_layer_norm = nn.LayerNorm(embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, embedding_dim)
        self.final_layer_norm = nn.LayerNorm(embedding_dim)
        self.Dropout = nn.Dropout(dropout)
        self.activation_dropout = nn.Dropout(activation_dropout)
        self.activation_fn = get_activation_fn(activation_fn)

    def forward(self, x, self_attn_bias=None, self_attn_padding_mask=None, self_attn_mask=None):
        residual = x
        if self.pre_layernorm:
            x = self.self_attn_layer_norm(x)
        x = self.self_attn(x, attn_bias=self_attn_bias, padding_mask=self_attn_padding_mask, attn_mask=self_attn_mask)
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

    def __init__(self, raw_node_feat_dim, num_in_degree, num_out_degree, edge_feat_dim, num_spatial, num_edge_dis, multi_hop_max_dist, num_encoder_layers=4, embedding_dim=128, ffn_embedding_dim=512, num_attention_heads=8, dropout=0.1, attention_dropout=0.1, activation_dropout=0.1, activation_fn='gelu', encoder_normalize_before=True, pre_layernorm=True, horizon=6):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_encoder_layers = num_encoder_layers
        self.graph_node_feature = SewerNodeFeature(raw_node_feat_dim=raw_node_feat_dim, num_in_degree=num_in_degree, num_out_degree=num_out_degree, hidden_dim=embedding_dim)
        self.graph_attn_bias = SewerGraphAttnBias(num_heads=num_attention_heads, num_spatial=num_spatial, num_edge_dis=num_edge_dis, edge_feat_dim=edge_feat_dim, multi_hop_max_dist=multi_hop_max_dist)
        self.emb_layer_norm = nn.LayerNorm(embedding_dim) if encoder_normalize_before else None
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([GraphormerGraphEncoderLayer(embedding_dim=embedding_dim, ffn_embedding_dim=ffn_embedding_dim, num_attention_heads=num_attention_heads, dropout=dropout, attention_dropout=attention_dropout, activation_dropout=activation_dropout, activation_fn=activation_fn, pre_layernorm=pre_layernorm) for _ in range(num_encoder_layers)])
        self.final_layer_norm = nn.LayerNorm(embedding_dim) if pre_layernorm else None
        self.Prediction = nn.Linear(embedding_dim, horizon)

    def forward(self, batched_data, perturb=None, last_state_only=True, token_embeddings=None, attn_mask=None):
        padding_mask = batched_data.get('padding_mask', None)
        if token_embeddings is not None:
            x = token_embeddings
        else:
            x = self.graph_node_feature(batched_data)
        if perturb is not None:
            x = x + perturb
        attn_bias = self.graph_attn_bias(batched_data)
        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)
        x = self.dropout(x)
        inner_states = []
        if not last_state_only:
            inner_states.append(x)
        for layer in self.layers:
            x = layer(x, self_attn_bias=attn_bias, self_attn_padding_mask=padding_mask, self_attn_mask=attn_mask)
            if not last_state_only:
                inner_states.append(x)
        node_rep = x
        if self.final_layer_norm is not None:
            node_rep = self.final_layer_norm(node_rep)
        if last_state_only:
            inner_states = [node_rep]
        water_level = self.Prediction(node_rep)
        return water_level


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", default="example_event", help="Event filename prefix.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/example"))
    parser.add_argument("--rain-dir", type=Path, default=None,
                        help="Defaults to --data-dir.")
    parser.add_argument("--static-file", type=Path, default=None)
    parser.add_argument("--adjacency-file", type=Path, default=Path("data/adjacency_matrix.xlsx"))
    parser.add_argument("--spatial-file", type=Path, default=Path("data/spatial_pos_matrix_undirected.xlsx"))
    parser.add_argument("--scaler-file", type=Path, default=Path("data/train_scalers.npz"))
    parser.add_argument("--model-file", type=Path, default=Path("models/pretrained.pth"))
    parser.add_argument("--observed-nodes", type=str, required=True,
                        help="Comma-separated, zero-based monitored node indices.")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--trusted-full-model", action="store_true",
                        help="Permit loading an entire pickled PyTorch module (trusted files only).")
    return parser.parse_args()


def block_mean(values, minutes=10):
    # Preserve the original averaging behavior, including a final partial block.
    return np.asarray([values[i:i + minutes].mean(axis=0)
                       for i in range(0, len(values), minutes)], dtype=np.float32)


def load_static(path):
    table = pd.read_excel(path, sheet_name="node_static_features")
    names = table["node"].astype(str).str.strip().tolist()
    feature_names = [c for c in table.columns if c not in ("node", "node_type")]
    return names, table[feature_names].to_numpy(dtype=np.float32)


def load_matrix(path):
    table = pd.read_excel(path)
    names = table.iloc[:, 0].astype(str).str.strip().tolist()
    columns = table.columns[1:].astype(str).str.strip().tolist()
    if names != columns:
        raise ValueError(f"Matrix row/column node order mismatch: {path}")
    return names, table.iloc[:, 1:].to_numpy()


def read_event(depth_file, rain_file, names):
    depth_frame = pd.read_csv(depth_file, sep=r"\s+")
    depth_frame.columns = depth_frame.columns.astype(str).str.strip()
    depth = block_mean(depth_frame[names].to_numpy(dtype=np.float32))
    rain_frame = pd.read_csv(rain_file, sep=r"\s+")
    # Rainfall input convention: omit the first and final original records.
    rain_1min = rain_frame.iloc[1:-1, -1:].to_numpy(dtype=np.float32)
    rain = block_mean(rain_1min)
    length = min(len(depth), len(rain))
    if length < TARGET_LENGTH:
        raise ValueError("Event has fewer than nine aligned 10-minute samples.")
    return depth[:length], rain[:length]


def build_samples(depth, rain, static):
    time_steps, num_nodes = depth.shape
    inputs, targets = [], []
    for t in range(HISTORY_LENGTH, time_steps - TARGET_LENGTH + HISTORY_LENGTH + 1):
        initial = t - HISTORY_LENGTH
        depth_history = depth[initial:t].T
        rain_window = np.repeat(rain[initial:initial + RAIN_FEATURES].T, num_nodes, axis=0)
        inputs.append(np.concatenate((static, rain_window, depth_history), axis=1))
        targets.append(depth[initial:initial + TARGET_LENGTH].T)
    return np.asarray(inputs, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def build_edge_inputs(names, adjacency, static_file):
    edge_table = pd.read_excel(static_file, sheet_name="edge_static_features")
    graph = nx.Graph()
    graph.add_nodes_from(names)
    edges = {}
    for _, row in edge_table.iterrows():
        u, v = str(row["from_node"]).strip(), str(row["to_node"]).strip()
        feature = row[list(EDGE_FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
        edges[(u, v)] = edges[(v, u)] = feature
    for i, j in zip(*np.nonzero(adjacency)):
        graph.add_edge(names[i], names[j])
    n = len(names)
    result = np.zeros((n, n, HOP_LIMIT, len(EDGE_FEATURE_COLUMNS)), dtype=np.float32)
    for i, source in enumerate(names):
        paths = nx.single_source_shortest_path(graph, source)
        for j, destination in enumerate(names):
            if source == destination or destination not in paths:
                continue
            path = paths[destination]
            for k, (u, v) in enumerate(zip(path[:-1], path[1:])):
                if k >= HOP_LIMIT:
                    break
                if (u, v) in edges:
                    result[i, j, k] = edges[(u, v)]
    return result


def load_scalers(path):
    keys = ("node_static_min", "node_static_max", "rain_min", "rain_max",
            "water_min", "water_max", "edge_min", "edge_max")
    with np.load(path, allow_pickle=False) as saved:
        missing = set(keys) - set(saved.files)
        if missing:
            raise ValueError(f"Missing scaler values: {sorted(missing)}")
        return {key: saved[key] for key in keys}


def minmax(values, low, high):
    return (values - low) / (high - low + 1e-8)


def prepare_inputs(X, edge, scalers, num_static, observed):
    X = X.copy()
    X[:, :, :num_static] = minmax(X[:, :, :num_static],
                                  scalers["node_static_min"], scalers["node_static_max"])
    start, end = num_static, num_static + RAIN_FEATURES
    X[:, :, start:end] = minmax(X[:, :, start:end], scalers["rain_min"], scalers["rain_max"])
    X[:, :, end:end + HISTORY_LENGTH] = minmax(
        X[:, :, end:end + HISTORY_LENGTH], scalers["water_min"], scalers["water_max"])
    mask = np.zeros(X.shape[1], dtype=bool)
    mask[observed] = True
    X[:, ~mask, -HISTORY_LENGTH:] = 0.0
    flags = np.broadcast_to(mask.astype(np.float32)[None, :, None], (*X.shape[:2], 1))
    X = np.concatenate((X, flags), axis=-1).astype(np.float32)
    edge = minmax(edge, np.asarray(scalers["edge_min"]).reshape(1, 1, 1, -1),
                  np.asarray(scalers["edge_max"]).reshape(1, 1, 1, -1))
    return X, edge.astype(np.float32)


def load_checkpoint(path, device, permit_pickle):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        if not permit_pickle:
            raise RuntimeError(
                "Safe checkpoint loading failed. Provide a state_dict checkpoint, "
                "or pass --trusted-full-model only for a trusted complete-module file."
            ) from error
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, nn.Module):
        return checkpoint.to(device), None
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must contain a state_dict or a PyTorch module.")
    # Ignore the optional DataParallel prefix without altering parameter names.
    state = {k.removeprefix("module."): v for k, v in checkpoint.items()}
    return None, state


def build_model(state, input_dim, edge_features, adjacency, spatial):
    dim = input_dim
    embedding = state["graph_node_feature.node_encoder.weight"].shape[0]
    ffn_dim = state["layers.0.fc1.weight"].shape[0]
    layer_ids = {int(key.split(".")[1]) for key in state if key.startswith("layers.")}
    num_layers = max(layer_ids) + 1
    # Number of attention heads is identifiable from the edge encoder's output.
    heads = state["graph_attn_bias.edge_encoder.weight"].shape[0]
    num_spatial = state["graph_attn_bias.spatial_pos_encoder.weight"].shape[0]
    num_in_degree = state["graph_node_feature.in_degree_encoder.weight"].shape[0]
    num_out_degree = state["graph_node_feature.out_degree_encoder.weight"].shape[0]
    num_edge_dis = state["graph_attn_bias.edge_dis_encoder.weight"].numel() // (heads * heads)
    horizon = state["Prediction.weight"].shape[0]
    model = SewerGraphEncoder(
        raw_node_feat_dim=dim, num_in_degree=num_in_degree,
        num_out_degree=num_out_degree, edge_feat_dim=edge_features,
        num_spatial=num_spatial, num_edge_dis=num_edge_dis,
        multi_hop_max_dist=HOP_LIMIT, num_encoder_layers=num_layers,
        embedding_dim=embedding, ffn_embedding_dim=ffn_dim,
        num_attention_heads=heads, dropout=0.1, attention_dropout=0.1,
        activation_dropout=0.1, activation_fn="gelu",
        encoder_normalize_before=True, pre_layernorm=True, horizon=horizon,
    )
    model.load_state_dict(state, strict=True)
    return model


def evaluate(model, X, edge, adjacency, spatial, batch_size, device):
    model.eval()
    count, n, _ = X.shape
    in_degree = torch.as_tensor(adjacency.sum(axis=0), dtype=torch.long, device=device)
    out_degree = torch.as_tensor(adjacency.sum(axis=1), dtype=torch.long, device=device)
    edge = torch.as_tensor(edge, device=device)
    spatial = torch.as_tensor(spatial, dtype=torch.long, device=device)
    attention = torch.zeros(n, n, device=device)
    preds = []
    with torch.inference_mode():
        for start in range(0, count, batch_size):
            batch = torch.as_tensor(X[start:start + batch_size], device=device)
            b = len(batch)
            inputs = {
                "x": batch,
                "in_degree": in_degree.unsqueeze(0).expand(b, -1),
                "out_degree": out_degree.unsqueeze(0).expand(b, -1),
                "attn_bias": attention.unsqueeze(0).expand(b, -1, -1),
                "spatial_pos": spatial.unsqueeze(0).expand(b, -1, -1),
                "edge_input": edge.unsqueeze(0).expand(b, -1, -1, -1, -1),
            }
            preds.append(model(inputs).cpu().numpy())
    return np.concatenate(preds, axis=0)


def main():
    args = arguments()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    data_dir = args.data_dir
    static_file = args.static_file or data_dir / "static_features.xlsx"
    rain_dir = args.rain_dir or data_dir
    observed = np.asarray([int(item.strip()) for item in args.observed_nodes.split(",")], dtype=np.int64)
    if not len(observed) or len(set(observed.tolist())) != len(observed):
        raise ValueError("Observed nodes must be nonempty and unique.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names, static = load_static(static_file)
    n = len(names)
    if observed.min() < 0 or observed.max() >= n:
        raise ValueError(f"Observed node indices must be between 0 and {n-1}.")
    adj_names, adjacency = load_matrix(args.adjacency_file)
    spatial_names, spatial = load_matrix(args.spatial_file)
    if names != adj_names or names != spatial_names:
        raise ValueError("Node order differs among the static, adjacency, and spatial files.")
    adjacency = adjacency.astype(np.float32)
    spatial = spatial.astype(np.int64)
    depth, rain = read_event(data_dir / f"{args.event}_depth.txt",
                             rain_dir / f"{args.event}.txt", names)
    scalers = load_scalers(args.scaler_file)
    loaded_model, state = load_checkpoint(args.model_file, device, args.trusted_full_model)
    X, Y = build_samples(depth, rain, static)
    edge = build_edge_inputs(names, adjacency, static_file)
    X, edge = prepare_inputs(X, edge, scalers, static.shape[1], observed)
    input_dim = X.shape[-1]
    if state is not None:
        required_dim = state["graph_node_feature.node_encoder.weight"].shape[1]
    else:
        required_dim = loaded_model.graph_node_feature.node_encoder.in_features
    if required_dim != input_dim:
        raise ValueError(
            f"Checkpoint expects {required_dim} node features, but the specified input "
            f"contains {input_dim} ({static.shape[1]} static + {RAIN_FEATURES} rainfall "
            f"+ {HISTORY_LENGTH} historical water depth + 1 observation flag). "
            "Verify that the checkpoint and feature definitions correspond to the same training run."
        )
    if loaded_model is None:
        loaded_model = build_model(state, input_dim, edge.shape[-1], adjacency, spatial).to(device)
    pred_norm = evaluate(loaded_model, X, edge, adjacency, spatial, args.batch_size, device)
    true_depth = Y
    predicted_depth = pred_norm * (scalers["water_max"] - scalers["water_min"] + 1e-8) + scalers["water_min"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.event}_predictions.npz"
    np.savez_compressed(output, predictions=predicted_depth, targets=true_depth,
                        observed_nodes=observed, node_names=np.array(names))
    rmse = float(np.sqrt(np.mean((predicted_depth - true_depth) ** 2)))
    mae = float(np.mean(np.abs(predicted_depth - true_depth)))
    print(f"Event: {args.event} | windows: {len(X)} | nodes: {n} | rainfall features: {RAIN_FEATURES}")
    print(f"MAE: {mae:.6f} m | RMSE: {rmse:.6f} m")
    print(f"Saved predictions: {output}")


if __name__ == "__main__":
    main()
