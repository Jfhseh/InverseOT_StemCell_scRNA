"""
Preprocessing utilities.

Wraps scanpy/sklearn operations so that the training pipeline can accept
data that has already been preprocessed externally (the common case for
scRNA-seq workflows) as well as raw counts.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def select_hvg(
    expression: np.ndarray,
    n_top_genes: int = 2000,
    flavor: str = "seurat_v3",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select highly variable genes.

    Tries scanpy first; falls back to variance-ranking if scanpy is absent.

    Returns
    -------
    expression_hvg : (N, n_top_genes) subset of the input matrix.
    gene_mask      : (G,) boolean array indicating selected genes.
    """
    try:
        import anndata as ad
        import scanpy as sc

        adata = ad.AnnData(X=expression)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor=flavor)
        mask: np.ndarray = adata.var["highly_variable"].values
        logger.info("HVG selection via scanpy: %d / %d genes kept", mask.sum(), len(mask))
        return expression[:, mask], mask

    except ImportError:
        logger.warning(
            "scanpy not available; falling back to variance-based HVG selection"
        )
        variances = np.var(expression, axis=0)
        top_idx = np.argsort(variances)[-n_top_genes:]
        mask = np.zeros(expression.shape[1], dtype=bool)
        mask[top_idx] = True
        return expression[:, mask], mask


def pca_reduce(
    expression: np.ndarray,
    n_components: int = 50,
    whiten: bool = False,
    random_state: int = 42,
) -> Tuple[np.ndarray, Any]:
    """
    Center, scale, and PCA-reduce an expression matrix.

    Returns
    -------
    embedding : (N, n_components) float32 array.
    pca_model : fitted sklearn PCA object (can be used to transform new data).
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(expression)

    n_components = min(n_components, min(X_scaled.shape) - 1)
    pca = PCA(n_components=n_components, whiten=whiten, random_state=random_state)
    embedding = pca.fit_transform(X_scaled).astype(np.float32)

    explained = pca.explained_variance_ratio_.sum()
    logger.info(
        "PCA: %d components explain %.1f%% of variance", n_components, 100 * explained
    )
    return embedding, pca


def log1p_normalize(
    expression: np.ndarray,
    scale_factor: float = 1e4,
) -> np.ndarray:
    """Library-size normalize then log1p transform (common scRNA-seq step)."""
    counts = expression.sum(axis=1, keepdims=True).clip(min=1.0)
    normed = expression / counts * scale_factor
    return np.log1p(normed).astype(np.float32)


def normalize_crypt_labels(
    labels: np.ndarray,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> np.ndarray:
    """
    Normalize crypt-axis labels to [0, 1].

    Parameters
    ----------
    labels :
        Raw ordinal integers or continuous depth scores.
    min_val / max_val :
        Optional fixed range; uses data range if not supplied.
    """
    labels = labels.astype(np.float32)
    lo = labels.min() if min_val is None else float(min_val)
    hi = labels.max() if max_val is None else float(max_val)
    if hi > lo:
        return (labels - lo) / (hi - lo)
    return np.zeros_like(labels)
