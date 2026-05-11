"""
Neighborhood graph construction for transcript and spatial modalities.

Both functions return (neighbor_indices, distances) as numpy arrays so they
can be precomputed once and stored, avoiding redundant computation across
training epochs.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def build_knn_graph(
    embedding: np.ndarray,
    k: int = 15,
    metric: str = "euclidean",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a k-nearest-neighbour graph from an embedding matrix.

    Parameters
    ----------
    embedding : (N, D) array — PCA coords or any Euclidean feature space.
    k         : number of neighbours per cell (self excluded).
    metric    : distance metric passed to sklearn NearestNeighbors.

    Returns
    -------
    neighbors  : (N, k) int64 indices of k nearest neighbours.
    distances  : (N, k) float32 Euclidean distances.
    """
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric, algorithm="auto", n_jobs=-1)
    nn.fit(embedding)
    distances, indices = nn.kneighbors(embedding)
    # Exclude self (always the first entry at distance 0)
    return indices[:, 1:].astype(np.int64), distances[:, 1:].astype(np.float32)


def build_spatial_graph(
    spatial_coords: np.ndarray,
    k: int = 6,
    radius: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a spatial adjacency graph from 2-D slide coordinates.

    Uses k-NN by default.  When ``radius`` is given, every spot within that
    Euclidean distance is a neighbour; rows are padded to the same width with
    ``-1`` (invalid) entries.

    Returns
    -------
    neighbors : (N, k_max) int64 neighbour indices; -1 = no neighbour.
    distances : (N, k_max) float32 distances; inf where neighbour is absent.
    """
    from sklearn.neighbors import NearestNeighbors

    if radius is None:
        return build_knn_graph(spatial_coords, k=k, metric="euclidean")

    nn = NearestNeighbors(radius=radius, metric="euclidean", n_jobs=-1)
    nn.fit(spatial_coords)
    dist_lists, idx_lists = nn.radius_neighbors(spatial_coords, sort_results=True)

    N = len(spatial_coords)
    max_nb = max((len(d) - 1 for d in dist_lists), default=0)
    neighbors = np.full((N, max_nb), -1, dtype=np.int64)
    distances = np.full((N, max_nb), np.inf, dtype=np.float32)

    for i, (drow, irow) in enumerate(zip(dist_lists, idx_lists)):
        # Remove self (distance 0)
        mask = irow != i
        nb = irow[mask][:max_nb]
        db = drow[mask][:max_nb]
        neighbors[i, : len(nb)] = nb
        distances[i, : len(db)] = db

    return neighbors, distances


class NeighborSampler:
    """
    Lightweight lookup wrapper for precomputed neighbor graphs.

    Stores both transcript and spatial neighbour indices and supports
    per-cell lookup by integer index array.
    """

    def __init__(
        self,
        transcript_neighbors: np.ndarray,  # (N, k_t)
        spatial_neighbors: Optional[np.ndarray] = None,  # (N, k_s)
    ) -> None:
        self.transcript_neighbors = transcript_neighbors
        self.spatial_neighbors = spatial_neighbors

    def get_transcript_neighbors(self, indices: np.ndarray) -> np.ndarray:
        """Return (B, k_t) neighbour indices for a batch of cell indices."""
        return self.transcript_neighbors[indices]

    def get_spatial_neighbors(self, indices: np.ndarray) -> Optional[np.ndarray]:
        """Return (B, k_s) spatial neighbour indices, or None."""
        if self.spatial_neighbors is None:
            return None
        return self.spatial_neighbors[indices]
