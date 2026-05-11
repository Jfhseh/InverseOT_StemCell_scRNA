"""
Evaluation metrics for Phase 1 hyperbolic crypt representations.

Metrics
-------
spearman_radius_depth
    Spearman rank correlation between hyperbolic radius ||z||_2 and the
    crypt-axis label.  Primary metric for Phase 1.

neighborhood_preservation
    Fraction of k-NN neighbours in the input (PCA) space that are also k-NN
    neighbours in the latent space.

silhouette (optional)
    Silhouette score using binned crypt labels as cluster assignments.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


def evaluate_embeddings(
    z: np.ndarray,
    crypt_labels: Optional[np.ndarray] = None,
    expression: Optional[np.ndarray] = None,
    k_nbrs: int = 15,
) -> Dict[str, float]:
    """
    Compute evaluation metrics for a set of hyperbolic embeddings.

    Parameters
    ----------
    z            : (N, D) Poincaré ball embeddings.
    crypt_labels : (N,) float crypt-axis labels.  If None, axis metrics are
                   skipped.
    expression   : (N, D_ref) reference feature vectors (e.g. PCA coords)
                   used for neighborhood preservation.  If None, NP is skipped.
    k_nbrs       : neighbourhood size for the NP metric.

    Returns
    -------
    Dict mapping metric name → scalar float.
    """
    metrics: Dict[str, float] = {}

    # Hyperbolic radius
    radius = np.linalg.norm(z, axis=1)  # (N,)

    # --- Spearman correlation between radius and crypt depth ---
    if crypt_labels is not None:
        valid = ~np.isnan(crypt_labels)
        if valid.sum() > 1:
            from scipy.stats import spearmanr
            rho, pval = spearmanr(radius[valid], crypt_labels[valid])
            metrics["spearman_radius_depth"] = float(rho)
            metrics["spearman_pval"] = float(pval)
            logger.debug("Spearman ρ=%.4f (p=%.3g)", rho, pval)

    # --- Neighborhood preservation ---
    if expression is not None:
        np_score = _neighborhood_preservation(expression, z, k=k_nbrs)
        metrics["neighborhood_preservation"] = np_score

    # --- Silhouette (if labels are ordinal bins) ---
    if crypt_labels is not None:
        valid = ~np.isnan(crypt_labels)
        if valid.sum() > 10:
            labels_int = _bin_labels(crypt_labels[valid], n_bins=4)
            if len(np.unique(labels_int)) > 1:
                try:
                    from sklearn.metrics import silhouette_score
                    sil = silhouette_score(z[valid], labels_int, metric="euclidean")
                    metrics["silhouette"] = float(sil)
                except Exception:
                    pass

    # Basic sanity checks
    metrics["mean_radius"] = float(radius.mean())
    metrics["std_radius"] = float(radius.std())
    metrics["max_radius"] = float(radius.max())

    return metrics


def _neighborhood_preservation(
    ref: np.ndarray,
    lat: np.ndarray,
    k: int = 15,
) -> float:
    """
    Fraction of k-NN neighbours shared between the reference and latent spaces.

    Computed by building k-NN graphs in both spaces and computing the mean
    Jaccard overlap between each cell's reference and latent neighbour sets.
    """
    from sklearn.neighbors import NearestNeighbors

    def _knn(X: np.ndarray) -> np.ndarray:
        nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=-1)
        nn.fit(X)
        _, idx = nn.kneighbors(X)
        return idx[:, 1:]  # exclude self

    ref_nb = _knn(ref)  # (N, k)
    lat_nb = _knn(lat)  # (N, k)

    overlaps = []
    for i in range(len(ref)):
        shared = len(set(ref_nb[i]) & set(lat_nb[i]))
        overlaps.append(shared / k)
    return float(np.mean(overlaps))


def _bin_labels(labels: np.ndarray, n_bins: int = 4) -> np.ndarray:
    """Bin continuous labels into n_bins ordinal classes."""
    bins = np.linspace(labels.min(), labels.max() + 1e-9, n_bins + 1)
    return np.digitize(labels, bins[1:])
