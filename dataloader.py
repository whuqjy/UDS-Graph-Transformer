"""Prepare multi-scenario sewer-network data for Graphformer.

Raw files in each scenario directory:
    X_<scenario>_<split>.npy: [samples, nodes, features].
    Y_<scenario>_<split>.npy: [samples, nodes, target_steps].
    Edge_<scenario>_<split>.npy: [nodes, nodes, hops, edge_features].

X columns must be ordered as static features, rainfall, and historical depths.
The default rainfall block has nine steps (history and forecast combined).
Node indices are zero-based and must follow the ordering used by the graph.

Fit scalers on training data only; reuse that file for validation and testing.
Example (replace paths and node indices with your experiment configuration):
    python Dataloader.py --root data/raw_train --output-dir data/processed \
        --split train --scaler-path data/train_scalers.npz --fit-scalers \
        --observed-nodes 82 90 201
    python Dataloader.py --root data/raw_test --output-dir data/processed \
        --split test --scaler-path data/train_scalers.npz \
        --observed-nodes 82 90 201

Outputs retain the filenames and array layout expected by Graphformer.
Only NumPy is required. Importing this module does not read or write data.
"""

import argparse
from pathlib import Path

import numpy as np

EPS = 1e-8
# Order defines scenario IDs and must remain consistent across data splits.
SCENARIO_NAMES = (
    "S00_no_perturb",
    "S01_conduits_30%_3timesmanning",
    "S02_conduits_40%_2timesmanning",
    "S03_maxdepth_30%_0.8",
    "S04_maxdepth_40%_0.9",
)
SCALER_KEYS = (
    "node_static_min", "node_static_max", "rain_min", "rain_max",
    "water_min", "water_max", "edge_min", "edge_max",
)


def load_all_scenarios(root, split):
    """Concatenate samples; retain one static edge tensor per scenario."""
    x_parts, y_parts, scenario_ids, edges = [], [], [], []
    for sid, name in enumerate(SCENARIO_NAMES):
        folder = Path(root) / name
        suffix = f"S{sid:02d}_{split}"
        x = np.load(folder / f"X_{suffix}.npy").astype(np.float32)
        y = np.load(folder / f"Y_{suffix}.npy").astype(np.float32)
        edge = np.load(folder / f"Edge_{suffix}.npy").astype(np.float32)
        if x.ndim != 3 or y.ndim != 3 or x.shape[:2] != y.shape[:2]:
            raise ValueError(f"{name}: expected matching [samples, nodes, ...] X/Y.")
        if edge.ndim != 4 or edge.shape[:2] != (x.shape[1], x.shape[1]):
            raise ValueError(f"{name}: expected edge shape [nodes, nodes, hops, features].")
        if x_parts and (x.shape[1:] != x_parts[0].shape[1:]
                        or y.shape[1:] != y_parts[0].shape[1:]
                        or edge.shape != edges[0].shape):
            raise ValueError(f"{name}: inconsistent feature or graph dimensions.")
        x_parts.append(x)
        y_parts.append(y)
        scenario_ids.append(np.full(x.shape[0], sid, dtype=np.int64))
        edges.append(edge)
        print(f"{name}: X={x.shape}, Y={y.shape}, edge={edge.shape}")
    return (np.concatenate(x_parts), np.concatenate(y_parts),
            np.concatenate(scenario_ids), edges)


def validate_feature_layout(x, static_dim, rain_steps, depth_steps):
    """Ensure that depth masking and normalization address the same columns."""
    if min(static_dim, rain_steps, depth_steps) <= 0:
        raise ValueError("Feature dimensions must be positive.")
    if x.shape[-1] != static_dim + rain_steps + depth_steps:
        raise ValueError("X must contain exactly static + rainfall + depth columns.")


def fit_scalers(x, y, edges, static_dim, rain_steps):
    """Fit the original min/max statistics using training arrays only.

    Static and edge features use separate bounds per feature. All rainfall
    steps share one range; historical and target depths share one water range.
    Edge padding participates in fitting, as in the original implementation.
    """
    static = x[..., :static_dim]
    rain = x[..., static_dim:static_dim + rain_steps]
    depth = x[..., static_dim + rain_steps:]
    edge_values = np.concatenate([e.reshape(-1, e.shape[-1]) for e in edges])
    return {
        "node_static_min": static.min(axis=(0, 1), keepdims=True),
        "node_static_max": static.max(axis=(0, 1), keepdims=True),
        "rain_min": rain.min(keepdims=True),
        "rain_max": rain.max(keepdims=True),
        "water_min": np.minimum(depth.min(), y.min()),
        "water_max": np.maximum(depth.max(), y.max()),
        "edge_min": edge_values.min(axis=0, keepdims=True),
        "edge_max": edge_values.max(axis=0, keepdims=True),
    }


def load_train_scalers(path):
    """Load the same scaler keys used by the original preprocessing script."""
    with np.load(path) as data:
        missing = set(SCALER_KEYS) - set(data.files)
        if missing:
            raise ValueError(f"Missing scaler keys: {sorted(missing)}")
        return {key: data[key] for key in SCALER_KEYS}


def minmax_transform(array, lower, upper):
    """Scale without clipping; test values may lie outside the training range."""
    return (array - lower) / (upper - lower + EPS)


def normalize_with_given_scalers(x, y, edges, scalers, static_dim, rain_steps):
    """Normalize using fixed statistics before masking unobserved depths."""
    rain_end = static_dim + rain_steps
    x_norm = x.copy()
    for columns, prefix in (
        (slice(0, static_dim), "node_static"),
        (slice(static_dim, rain_end), "rain"),
        (slice(rain_end, None), "water"),
    ):
        x_norm[..., columns] = minmax_transform(
            x[..., columns], scalers[f"{prefix}_min"], scalers[f"{prefix}_max"]
        )
    y_norm = minmax_transform(y, scalers["water_min"], scalers["water_max"])
    edge_min = scalers["edge_min"].reshape(1, 1, 1, -1)
    edge_max = scalers["edge_max"].reshape(1, 1, 1, -1)
    edge_norm = [minmax_transform(e, edge_min, edge_max).astype(np.float32)
                 for e in edges]
    return x_norm.astype(np.float32), y_norm.astype(np.float32), edge_norm


def apply_observed_node_mask(x, observed_nodes, depth_steps):
    """Zero normalized depths at unobserved nodes; preserve all other features.

    The mask is shared across samples. Zero is a missing-input placeholder
    in normalized space, not necessarily a physical water depth of zero.
    """
    nodes = np.asarray(observed_nodes)
    if nodes.ndim != 1 or nodes.size == 0 or not np.issubdtype(nodes.dtype, np.integer):
        raise ValueError("Observed nodes must be a nonempty list of integer indices.")
    if nodes.min() < 0 or nodes.max() >= x.shape[1]:
        raise ValueError(f"Observed node indices must lie in [0, {x.shape[1] - 1}].")
    if np.unique(nodes).size != nodes.size:
        raise ValueError("Observed node indices must be unique.")
    if not 0 < depth_steps <= x.shape[-1]:
        raise ValueError("Invalid number of historical depth steps.")
    mask = np.zeros(x.shape[1], dtype=bool)
    mask[nodes] = True
    x_masked = x.copy()
    x_masked[:, ~mask, -depth_steps:] = 0.0
    return x_masked.astype(np.float32), mask


def add_observed_flag(x, node_mask):
    """Append one binary column to distinguish observed and unobserved nodes."""
    flag = np.broadcast_to(node_mask[None, :, None], (*x.shape[:2], 1))
    return np.concatenate((x, flag), axis=-1).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="Raw scenario directory.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--scaler-path", type=Path, required=True)
    parser.add_argument("--fit-scalers", action="store_true",
                        help="Fit and save scalers; allowed only for the training split.")
    parser.add_argument("--observed-nodes", type=int, nargs="+", required=True,
                        help="Zero-based sensor indices; use the same list for each split.")
    parser.add_argument("--static-dim", type=int, default=17)
    parser.add_argument("--rain-steps", type=int, default=9)
    parser.add_argument("--depth-steps", type=int, default=3)
    args = parser.parse_args()
    if args.fit_scalers and args.split != "train":
        parser.error("--fit-scalers is only allowed with --split train.")

    x, y, scenario_ids, edges = load_all_scenarios(args.root, args.split)
    validate_feature_layout(x, args.static_dim, args.rain_steps, args.depth_steps)
    if args.fit_scalers:
        scalers = fit_scalers(x, y, edges, args.static_dim, args.rain_steps)
    else:
        scalers = load_train_scalers(args.scaler_path)
    x, y, edges = normalize_with_given_scalers(
        x, y, edges, scalers, args.static_dim, args.rain_steps
    )
    x, node_mask = apply_observed_node_mask(x, args.observed_nodes, args.depth_steps)
    x = add_observed_flag(x, node_mask)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.fit_scalers:
        args.scaler_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.scaler_path, **scalers)
    np.save(args.output_dir / f"X_{args.split}.npy", x)
    np.save(args.output_dir / f"Y_{args.split}.npy", y)
    np.save(args.output_dir / f"scenario_{args.split}.npy", scenario_ids)
    np.savez_compressed(
        args.output_dir / f"edge_input_norm_list_{args.split}.npz",
        **{f"edge_{i}": edge for i, edge in enumerate(edges)},
    )
    print(f"Saved {args.split}: X={x.shape}, Y={y.shape} to {args.output_dir}")


if __name__ == "__main__":
    main()
