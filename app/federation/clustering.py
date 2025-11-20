from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def cluster_aware_average_selected(
    items: Sequence[Tuple[Sequence[np.ndarray], int]],
    indices: Sequence[int],
    *,
    num_clusters: int = 1,
    max_dim: int = 4096,
    max_iter: int = 25,
    tol: float = 1e-4,  # reserved for future stopping criteria
    random_state: Optional[int] = None,
) -> Tuple[List[List[np.ndarray]], List[int]]:
    """
    Cluster-aware aggregation over selected parameters (e.g., LoRA A matrices).

    Parameters
    ----------
    items:
        Sequence of (arrays, weight), where `arrays` is the LoRA state
        and `weight` is typically the number of local examples.
    indices:
        Indices into `arrays` specifying which parameters to aggregate.
    num_clusters:
        Number of clusters for k-means. If <=1, behaves like global FedAvg.
    max_dim:
        Max embedding dimension for clustering (downsamples large vectors).
    max_iter:
        Max k-means iterations.
    tol:
        Currently unused (reserved for future stopping criteria).
    random_state:
        Seed for reproducibility.

    Returns
    -------
    aggregated_per_item:
        List of length len(items); each element is a list of ndarrays
        giving the aggregated parameters for that item's cluster.
    labels:
        List of cluster labels per item (same order as `items`).
    """
    n = len(items)
    if n == 0 or not indices:
        return [], []

    # ---------- No-cluster or trivial cluster cases ----------
    if num_clusters <= 1:
        # Single global FedAvg across all items
        total_weight = float(sum(weight for _, weight in items))
        if total_weight <= 0.0:
            global_avg = [np.array(items[0][0][idx], copy=True) for idx in indices]
        else:
            global_avg = [
                np.zeros_like(items[0][0][idx], dtype=np.float32) for idx in indices
            ]
            for arrays, weight in items:
                frac = float(weight) / total_weight
                for pos, idx in enumerate(indices):
                    global_avg[pos] += arrays[idx].astype(np.float32, copy=False) * frac

        aggregated_per_item = [global_avg for _ in range(n)]
        labels = [0] * n
        return aggregated_per_item, labels

    if num_clusters >= n:
        # Each item gets its own "cluster" (no cross-client mixing)
        aggregated_per_item: List[List[np.ndarray]] = []
        labels: List[int] = []
        for i, (arrays, _) in enumerate(items):
            agg = [np.array(arrays[idx], copy=True) for idx in indices]
            aggregated_per_item.append(agg)
            labels.append(i)
        return aggregated_per_item, labels

    # ---------- Build embeddings from selected LoRA params ----------
    def _build_embedding(arrays: Sequence[np.ndarray]) -> np.ndarray:
        selected: List[np.ndarray] = []
        for idx in indices:
            w = arrays[idx].astype(np.float32, copy=False).ravel()
            selected.append(w)
        if not selected:
            return np.zeros((max_dim,), dtype=np.float32)
        vec = np.concatenate(selected, axis=0)
        if vec.size > max_dim:
            step = math.ceil(vec.size / max_dim)
            vec = vec[::step][:max_dim]
        return vec

    embeddings = [_build_embedding(arrays) for arrays, _ in items]
    X = np.stack(embeddings, axis=0)  # (n, d)

    rng = np.random.RandomState(0 if random_state is None else random_state)

    # ---------- k-means++ initialization ----------
    centroids = np.empty((num_clusters, X.shape[1]), dtype=np.float32)
    first_idx = rng.randint(0, n)
    centroids[0] = X[first_idx]
    for k in range(1, num_clusters):
        dists_sq = np.min(
            ((X[:, None, :] - centroids[None, :k, :]) ** 2).sum(axis=2),
            axis=1,
        )
        probs = dists_sq / (dists_sq.sum() + 1e-12)
        next_idx = rng.choice(n, p=probs)
        centroids[k] = X[next_idx]

    labels = np.zeros(n, dtype=np.int32)

    # ---------- Lloyd's k-means iterations ----------
    for _ in range(max_iter):
        dists = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(dists, axis=1)

        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        for k in range(num_clusters):
            mask = labels == k
            if not np.any(mask):
                centroids[k] = X[rng.randint(0, n)]
            else:
                centroids[k] = X[mask].mean(axis=0)

    # ---------- Cluster-wise weighted averaging on selected indices ----------
    clusters: Dict[int, List[Tuple[Sequence[np.ndarray], int]]] = defaultdict(list)
    for i, (arrays, weight) in enumerate(items):
        cid = int(labels[i])
        clusters[cid].append((arrays, weight))

    cluster_avgs: Dict[int, List[np.ndarray]] = {}
    for cid, cluster_items in clusters.items():
        total_weight = float(sum(w for _, w in cluster_items))
        if total_weight <= 0.0:
            cluster_avgs[cid] = [
                np.array(cluster_items[0][0][idx], copy=True) for idx in indices
            ]
            continue

        acc = [
            np.zeros_like(cluster_items[0][0][idx], dtype=np.float32)
            for idx in indices
        ]
        for arrays, weight in cluster_items:
            frac = float(weight) / total_weight
            for pos, idx in enumerate(indices):
                acc[pos] += arrays[idx].astype(np.float32, copy=False) * frac
        cluster_avgs[cid] = acc

    # ---------- Expand back to per-item aggregated params ----------
    aggregated_per_item: List[List[np.ndarray]] = []
    for i in range(n):
        cid = int(labels[i])
        avg_for_cluster = cluster_avgs[cid]
        aggregated_per_item.append([np.array(a, copy=True) for a in avg_for_cluster])

    return aggregated_per_item, labels.tolist()
