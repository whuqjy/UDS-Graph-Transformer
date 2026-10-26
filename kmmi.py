"""K-means-constrained Gaussian mutual information sensor placement.

The input is a water-level matrix of shape (time_steps, nodes).
Each node is clustered using its independently normalized time series.
One sensor is then selected from each cluster by maximizing the Gaussian
mutual information of the accumulated sensor set.

Example:
    python kmmi.py --input data/water_levels.npy --sensors 12
    python kmmi.py --input data/water_levels.csv --sensors 9 --output results/kmmi.csv
"""

import argparse
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans


def load_water_levels(path: Path, csv_header: bool = False) -> np.ndarray:
    """Load a two-dimensional water-level matrix from a .npy or .csv file."""
    if path.suffix.lower() == ".npy":
        water_levels = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".csv":
        water_levels = np.loadtxt(
            path, delimiter=",", skiprows=1 if csv_header else 0
        )
    else:
        raise ValueError("Input format must be .npy or .csv.")

    water_levels = np.asarray(water_levels, dtype=np.float64)
    if water_levels.ndim != 2:
        raise ValueError("Water levels must have shape (time_steps, nodes).")
    if water_levels.shape[0] < 2 or water_levels.shape[1] < 1:
        raise ValueError("At least two time steps and one node are required.")
    if not np.isfinite(water_levels).all():
        raise ValueError("Water levels contain NaN or infinite values.")
    return water_levels


def gaussian_mutual_information(
    covariance: np.ndarray, selected_indices: list[int], noise_variance: float
) -> float:
    """Compute 0.5 * log(det(I + covariance[A, A] / noise_variance))."""
    if not selected_indices:
        return 0.0

    selected_covariance = covariance[np.ix_(selected_indices, selected_indices)]
    matrix = np.eye(len(selected_indices)) + selected_covariance / noise_variance
    sign, log_determinant = np.linalg.slogdet(matrix)
    if sign <= 0:
        raise ValueError("Mutual information matrix is not positive definite.")
    return float(0.5 * log_determinant)


def select_sensors(
    water_levels: np.ndarray,
    n_sensors: int = 12,
    noise_std: float = 0.01,
    random_state: int = 42,
    n_init: int = 20,
    normalization_eps: float = 1e-8,
) -> list[dict]:
    """Select one sensor per cluster using KMeans-constrained Gaussian MI.

    Cluster selection proceeds from the largest to the smallest cluster.
    At each step, the node yielding the highest total mutual information
    conditional on previously selected nodes is selected.

    Returns:
        Selection records in greedy order, using zero-based node indices.
    """
    water_levels = np.asarray(water_levels, dtype=np.float64)
    if water_levels.ndim != 2 or water_levels.shape[0] < 2:
        raise ValueError("Expected a 2D matrix with at least two time steps.")
    if not np.isfinite(water_levels).all():
        raise ValueError("Water levels contain NaN or infinite values.")
    n_nodes = water_levels.shape[1]
    if not 1 <= n_sensors <= n_nodes:
        raise ValueError("n_sensors must be between 1 and the number of nodes.")
    if noise_std <= 0 or not np.isfinite(noise_std):
        raise ValueError("noise_std must be a finite positive number.")
    if normalization_eps <= 0:
        raise ValueError("normalization_eps must be positive.")

    # Preserve the original per-node min-max normalization.
    features = water_levels.T
    node_min = features.min(axis=1, keepdims=True)
    node_max = features.max(axis=1, keepdims=True)
    normalized = (features - node_min) / (node_max - node_min + normalization_eps)

    labels = KMeans(
        n_clusters=n_sensors, random_state=random_state, n_init=n_init
    ).fit_predict(normalized)

    # Keep NumPy's original descending cluster-size ordering.
    cluster_sizes = np.bincount(labels, minlength=n_sensors)
    cluster_order = np.argsort(cluster_sizes)[::-1]
    covariance = np.cov(water_levels, rowvar=False, ddof=1)
    covariance = np.atleast_2d(covariance)

    selected = []
    previous_mi = 0.0
    records = []
    noise_variance = noise_std**2

    for step, cluster in enumerate(cluster_order, start=1):
        candidates = np.flatnonzero(labels == cluster)
        best_node = None
        best_mi = -np.inf

        for node in candidates:
            trial_mi = gaussian_mutual_information(
                covariance, selected + [int(node)], noise_variance
            )
            if trial_mi > best_mi:
                best_node, best_mi = int(node), trial_mi

        if best_node is None:
            raise RuntimeError(f"No candidate found in cluster {cluster}.")
        records.append(
            {
                "step": step,
                "cluster": int(cluster),
                "cluster_size": int(cluster_sizes[cluster]),
                "node_index": best_node,
                "node_name": f"Node_{best_node}",
                "mutual_information": best_mi,
                "marginal_gain": best_mi - previous_mi,
            }
        )
        selected.append(best_node)
        previous_mi = best_mi

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("data/water_levels.npy"),
        help="Input matrix (.npy or .csv), with rows as time steps and columns as nodes.",
    )
    parser.add_argument("--sensors", type=int, default=12, help="Number of sensors.")
    parser.add_argument(
        "--noise-std", type=float, default=0.01,
        help="Observation noise standard deviation in meters.",
    )
    parser.add_argument("--seed", type=int, default=42, help="KMeans random seed.")
    parser.add_argument(
        "--csv-header", action="store_true",
        help="Skip the first row when loading a CSV input file.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/kmmi_selection.csv"),
        help="CSV path for the selected sensors.",
    )
    args = parser.parse_args()

    water_levels = load_water_levels(args.input, csv_header=args.csv_header)
    records = select_sensors(
        water_levels, n_sensors=args.sensors,
        noise_std=args.noise_std, random_state=args.seed,
    )

    print("KMeans-constrained Gaussian MI sensor placement")
    for record in records:
        print(
            f"Step {record['step']:02d} | Cluster {record['cluster']:02d} "
            f"(size={record['cluster_size']}) | Node {record['node_index']} "
            f"| MI={record['mutual_information']:.6f} "
            f"| Gain={record['marginal_gain']:.6f}"
        )
    print("Selected node indices:", [r["node_index"] for r in records])

    import csv

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved results to: {args.output}")


if __name__ == "__main__":
    main()
