"""Build graph-based learning arrays from drainage simulation events.

Expected inputs
---------------
* Node and edge features: two sheets in a single Excel workbook.
* Adjacency and spatial-distance matrices: labeled Excel matrices.
* Water depths: whitespace-delimited ``*_depth.txt`` files, one per event.
* Rainfall: matching whitespace-delimited ``*.txt`` files.

The preprocessing intentionally retains the original window definition:
``X = [node features, rainfall over the target window, past water depths]``
and ``Y = water depths over the target window``. The target window includes
historical time steps; it is not a future-only forecast target.

Example
-------
python dataset.py --scenario S00 --split test \\
    --static-file data/S00/static_features_S00.xlsx \\
    --depth-dir data/S00/depth \\
    --rain-dir data/rainfall \\
    --adjacency-file data/adjacency_matrix.xlsx \\
    --spatial-file data/spatial_pos_matrix_undirected.xlsx \\
    --output-dir outputs/S00
"""

from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


EDGE_FEATURE_COLUMNS = (
    "edge_type_id", "shape_id", "maxdepth", "length", "roughness",
    "in_offset", "out_offset", "slope", "hydraulic_radius",
    "velocity_index", "retention_time",
)


def block_mean_array(values: np.ndarray, block_size: int = 10) -> np.ndarray:
    """Average successive, nonoverlapping blocks, including a partial last block."""
    if block_size < 1 or values.shape[0] == 0:
        raise ValueError("Block size must be positive and input must be nonempty.")
    # Accumulate using the same float32 input behavior as the original script.
    return np.stack(
        [values[start:start + block_size].mean(axis=0)
         for start in range(0, len(values), block_size)],
        axis=0,
    ).astype(np.float32)


def read_node_features(path: Path, sheet: str) -> tuple[list[str], np.ndarray]:
    frame = pd.read_excel(path, sheet_name=sheet)
    if "node" not in frame:
        raise ValueError("Node feature sheet requires a 'node' column.")
    names = frame["node"].astype(str).tolist()
    if len(names) != len(set(names)):
        raise ValueError("Duplicate node names in the node feature sheet.")
    feature_cols = [col for col in frame if col not in ("node", "node_type")]
    return names, frame[feature_cols].to_numpy(dtype=np.float32)


def read_depth(path: Path, node_names: list[str], interval: int) -> np.ndarray:
    frame = pd.read_csv(path, sep=r"\s+", engine="python")
    missing = set(node_names).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing depth columns in {path.name}: {sorted(missing)}")
    return block_mean_array(frame.loc[:, node_names].to_numpy(dtype=np.float32), interval)


def read_rain(path: Path, interval: int) -> np.ndarray:
    frame = pd.read_csv(path, sep=r"\s+", engine="python")
    # Preserve the original convention: final column; drop first and last rows.
    values = frame.iloc[:, [-1]].to_numpy(dtype=np.float32)[1:-1]
    return block_mean_array(values, interval)


def build_event_samples(
    depth: np.ndarray,
    rain: np.ndarray,
    node_features: np.ndarray,
    history_len: int = 3,
    target_len: int = 9,
) -> tuple[np.ndarray, np.ndarray]:
    """Create overlapping windows, retaining the original time indexing.

    X: [samples, nodes, static_features + target_len + history_len]
    Y: [samples, nodes, target_len]
    """
    if history_len < 1 or target_len < history_len:
        raise ValueError("Require 1 <= history_len <= target_len.")
    length = min(len(depth), len(rain))
    if length < target_len:
        raise ValueError(f"Event has {length} steps but needs at least {target_len}.")
    if node_features.shape[0] != depth.shape[1]:
        raise ValueError("Node-feature and depth node counts differ.")
    depth, rain = depth[:length], rain[:length]
    num_nodes = depth.shape[1]
    x_windows, y_windows = [], []
    for t in range(history_len, length - target_len + history_len + 1):
        past_depth = depth[t - history_len:t].T
        window_rain = np.repeat(
            rain[t - history_len:t - history_len + target_len].T,
            num_nodes,
            axis=0,
        )
        x_windows.append(np.concatenate([node_features, window_rain, past_depth], axis=1))
        y_windows.append(depth[t - history_len:t - history_len + target_len].T)
    return (np.stack(x_windows).astype(np.float32),
            np.stack(y_windows).astype(np.float32))


def read_labeled_matrix(path: Path) -> tuple[list[str], np.ndarray]:
    frame = pd.read_excel(path)
    row_names = frame.iloc[:, 0].astype(str).str.strip().tolist()
    col_names = frame.columns[1:].astype(str).str.strip().tolist()
    if row_names != col_names:
        raise ValueError(f"Row and column node order differ in {path.name}.")
    if len(row_names) != len(set(row_names)):
        raise ValueError(f"Duplicate node labels in {path.name}.")
    return row_names, frame.iloc[:, 1:].to_numpy()


def build_edge_input(
    node_names: list[str],
    adjacency: np.ndarray,
    edge_features: pd.DataFrame,
    max_hops: int,
    directed: bool = False,
) -> np.ndarray:
    """Store features along one unweighted shortest path between each node pair.

    For reproducibility, edge features are assigned to both orientations, even
    when the adjacency is directed, as in the source implementation.
    """
    graph = nx.DiGraph() if directed else nx.Graph()
    graph.add_nodes_from(node_names)
    for i, j in np.argwhere(adjacency != 0):
        graph.add_edge(node_names[int(i)], node_names[int(j)])

    features = {}
    for _, row in edge_features.iterrows():
        u, v = str(row["from_node"]).strip(), str(row["to_node"]).strip()
        value = row[list(EDGE_FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
        features[u, v] = value
        features[v, u] = value

    result = np.zeros(
        (len(node_names), len(node_names), max_hops, len(EDGE_FEATURE_COLUMNS)),
        dtype=np.float32,
    )
    # networkx shortest_path uses the same shortest-path definition as the source.
    for i, source in enumerate(node_names):
        paths = nx.single_source_shortest_path(graph, source)
        for j, target in enumerate(node_names):
            if source == target or target not in paths:
                continue
            path = paths[target]
            for hop, (u, v) in enumerate(zip(path[:-1], path[1:])):
                if hop >= max_hops:
                    break
                if (u, v) in features:
                    result[i, j, hop] = features[u, v]
    return result


def build_dataset(args: argparse.Namespace) -> None:
    node_names, node_features = read_node_features(args.static_file, args.node_sheet)
    adj_names, adjacency = read_labeled_matrix(args.adjacency_file)
    spatial_names, spatial_pos = read_labeled_matrix(args.spatial_file)
    if node_names != adj_names or adj_names != spatial_names:
        raise ValueError("Node order must match across static, adjacency, and spatial inputs.")
    spatial_pos = spatial_pos.astype(np.int64)
    max_hops = int(spatial_pos.max()) if args.max_hops is None else args.max_hops
    if max_hops < 1:
        raise ValueError("Maximum hop count must be at least 1.")

    edge_frame = pd.read_excel(args.static_file, sheet_name=args.edge_sheet)
    required = {"from_node", "to_node", *EDGE_FEATURE_COLUMNS}
    if missing := required.difference(edge_frame.columns):
        raise ValueError(f"Missing edge-feature columns: {sorted(missing)}")
    if args.standardize_edges:
        cols = list(EDGE_FEATURE_COLUMNS)
        # Match pandas' default sample-standard-deviation convention.
        edge_frame[cols] = (edge_frame[cols] - edge_frame[cols].mean()) / (
            edge_frame[cols].std() + 1e-6
        )
    edge_input = build_edge_input(
        adj_names, adjacency, edge_frame, max_hops, directed=args.directed,
    )

    depth_files = sorted(args.depth_dir.glob("*_depth.txt"))
    if not depth_files:
        raise FileNotFoundError(f"No *_depth.txt files in {args.depth_dir}")
    inputs, targets = [], []
    for depth_path in depth_files:
        rain_path = args.rain_dir / depth_path.name.replace("_depth", "")
        if not rain_path.is_file():
            raise FileNotFoundError(f"No matching rainfall file: {rain_path.name}")
        depth = read_depth(depth_path, node_names, args.interval)
        rain = read_rain(rain_path, args.interval)
        x_event, y_event = build_event_samples(
            depth, rain, node_features, args.history_len, args.target_len,
        )
        inputs.append(x_event)
        targets.append(y_event)
        print(f"Processed {depth_path.name}: {len(x_event)} samples")

    X, Y = np.concatenate(inputs), np.concatenate(targets)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.scenario}_{args.split}"
    np.save(args.output_dir / f"X_{suffix}.npy", X)
    np.save(args.output_dir / f"Y_{suffix}.npy", Y)
    np.save(args.output_dir / f"Edge_{suffix}.npy", edge_input)
    print(f"Saved X: {X.shape}; Y: {Y.shape}; Edge: {edge_input.shape}")
    print(f"Output directory: {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="S00", help="Scenario identifier")
    parser.add_argument("--split", default="test", help="Dataset split identifier")
    parser.add_argument("--static-file", type=Path, default=Path("data/S00/static_features_S00.xlsx"))
    parser.add_argument("--depth-dir", type=Path, default=Path("data/S00/depth"))
    parser.add_argument("--rain-dir", type=Path, default=Path("data/rainfall"))
    parser.add_argument("--adjacency-file", type=Path, default=Path("data/adjacency_matrix.xlsx"))
    parser.add_argument("--spatial-file", type=Path, default=Path("data/spatial_pos_matrix_undirected.xlsx"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/S00"))
    parser.add_argument("--node-sheet", default="node_static_features")
    parser.add_argument("--edge-sheet", default="edge_static_features")
    parser.add_argument("--history-len", type=int, default=3)
    parser.add_argument("--target-len", type=int, default=9)
    parser.add_argument("--interval", type=int, default=10, help="Rows per averaging block")
    parser.add_argument("--max-hops", type=int, default=None, help="Defaults to max spatial position")
    parser.add_argument("--directed", action="store_true")
    parser.add_argument("--standardize-edges", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    build_dataset(parse_args())
