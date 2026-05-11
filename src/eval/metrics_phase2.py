"""
Phase 2B evaluation metrics.

Extends the Phase 1 metrics with per-zone breakdowns and donor mixing score.

Functions
---------
evaluate_phase2
    Full evaluation: Spearman, NP@15, Silhouette, DMS — global and per-zone.
    Returns a flat dict for easy JSON serialization and comparison tables.

compute_dms
    Within-zone donor mixing score (fraction of k-NN from different donor).

compute_np15_per_zone
    Per-zone neighborhood preservation.

radial_summary
    Radial spread metrics: r_mean, r_std, radial_gap, per-zone medians.

Interpretation rules (from phase2B.md)
---------------------------------------
- high Spearman + radial_gap ≈ 0 → rank preserved but radial geometry compressed
- high DMS within zones + high NP@15_within_zone → cross-donor alignment succeeds
  without sacrificing local structure
- high Silhouette alone is not diagnostic; interpret with DMS and r_std
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

_ZONE_BINS = np.array([0.0, 0.125, 0.375, 0.75, 1.01])
_ZONE_NAMES = {0: "sub-crypt", 1: "crypt-base", 2: "crypt-mid", 3: "crypt-apex"}


def _depths_to_zones(depths: np.ndarray) -> np.ndarray:
    return np.digitize(depths, _ZONE_BINS[1:]).clip(0, 3)


# ---------------------------------------------------------------------------
# Donor mixing score
# ---------------------------------------------------------------------------

def compute_dms(
    z: np.ndarray,
    zones: np.ndarray,
    donor_ids: np.ndarray,
    k: int = 15,
) -> Dict[str, float]:
    """Compute within-zone donor mixing score.

    For each cell, compute fraction of k nearest neighbors (in embedding)
    from a different donor. Average per zone and globally.

    Parameters
    ----------
    z         : (N, D) embeddings
    zones     : (N,) integer zone indices {0,1,2,3}
    donor_ids : (N,) donor identifier array (any dtype, compared by equality)
    k         : neighborhood size

    Returns
    -------
    Dict with keys: dms_z{0..3}, mean_dms, global_dms
    """
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(z)
    _, idx = nn.kneighbors(z)
    idx = idx[:, 1:]  # exclude self

    cross_donor = np.array([
        np.mean(donor_ids[idx[i]] != donor_ids[i])
        for i in range(len(z))
    ])

    result: Dict[str, float] = {}
    per_zone = []
    for zk in range(4):
        mask = (zones == zk)
        if mask.sum() > 0:
            val = float(np.mean(cross_donor[mask]))
            result[f"dms_z{zk}"] = val
            per_zone.append(val)
        else:
            result[f"dms_z{zk}"] = float("nan")

    result["mean_dms"] = float(np.nanmean(per_zone)) if per_zone else float("nan")
    result["global_dms"] = float(np.mean(cross_donor))
    return result


# ---------------------------------------------------------------------------
# Per-zone neighborhood preservation
# ---------------------------------------------------------------------------

def compute_np15_per_zone(
    z: np.ndarray,
    expression: np.ndarray,
    zones: np.ndarray,
    k: int = 15,
) -> Dict[str, float]:
    """Neighborhood preservation @ k, globally and per zone.

    Parameters
    ----------
    z          : (N, D) embeddings
    expression : (N, D_ref) reference features (PCA space)
    zones      : (N,) integer zone indices
    k          : neighborhood size

    Returns
    -------
    Dict with keys: np15_global, np15_z{0..3}, mean_np15_within_zone
    """
    from sklearn.neighbors import NearestNeighbors

    def knn(X, k):
        nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(X)
        _, idx = nn.kneighbors(X)
        return idx[:, 1:]

    ref_nb = knn(expression, k)
    lat_nb = knn(z, k)

    per_cell = np.array([
        len(set(ref_nb[i]) & set(lat_nb[i])) / k
        for i in range(len(z))
    ])

    result: Dict[str, float] = {"np15_global": float(per_cell.mean())}
    per_zone = []
    for zk in range(4):
        mask = (zones == zk)
        if mask.sum() > 0:
            val = float(per_cell[mask].mean())
            result[f"np15_z{zk}"] = val
            per_zone.append(val)
        else:
            result[f"np15_z{zk}"] = float("nan")

    result["mean_np15_within_zone"] = float(np.nanmean(per_zone)) if per_zone else float("nan")
    return result


# ---------------------------------------------------------------------------
# Radial summary
# ---------------------------------------------------------------------------

def radial_summary(
    z: np.ndarray,
    depths: np.ndarray,
    zones: np.ndarray,
) -> Dict[str, float]:
    """Compute radial spread metrics.

    Returns
    -------
    Dict with: r_mean, r_std, radial_gap (median z3 - median z0),
    gap_01, gap_12, gap_23, median_r_z{0..3}
    """
    radii = np.linalg.norm(z, axis=1)
    result: Dict[str, float] = {
        "r_mean": float(radii.mean()),
        "r_std":  float(radii.std()),
        "r_max":  float(radii.max()),
    }

    med_r = {}
    for zk in range(4):
        mask = (zones == zk)
        if mask.sum() > 0:
            med_r[zk] = float(np.median(radii[mask]))
            result[f"median_r_z{zk}"] = med_r[zk]
        else:
            result[f"median_r_z{zk}"] = float("nan")

    # Overall gap: zone 3 - zone 0
    if 0 in med_r and 3 in med_r:
        result["radial_gap"] = med_r[3] - med_r[0]
    else:
        result["radial_gap"] = float("nan")

    # Adjacent gaps
    for k, label in [(0, "gap_01"), (1, "gap_12"), (2, "gap_23")]:
        if k in med_r and (k + 1) in med_r:
            result[label] = med_r[k + 1] - med_r[k]
        else:
            result[label] = float("nan")

    return result


# ---------------------------------------------------------------------------
# Full Phase 2 evaluation
# ---------------------------------------------------------------------------

def evaluate_phase2(
    z: np.ndarray,
    crypt_labels: np.ndarray,
    expression: np.ndarray,
    donor_ids: Optional[np.ndarray] = None,
    k_nbrs: int = 15,
) -> Dict[str, float]:
    """Full Phase 2B evaluation: global and per-zone metrics.

    Parameters
    ----------
    z            : (N, D) Poincaré ball embeddings
    crypt_labels : (N,) depth labels in [0,1]
    expression   : (N, D_ref) PCA or expression features
    donor_ids    : (N,) donor identifier (age group etc.). If None, DMS skipped.
    k_nbrs       : neighborhood size for NP@15 and DMS

    Returns
    -------
    Flat dict of all metrics for easy JSON serialization / table printing.
    Keys: spearman_rho, np15_global, np15_z{0..3}, mean_np15_within_zone,
          silhouette, r_mean, r_std, radial_gap, gap_{01,12,23},
          median_r_z{0..3}, dms_z{0..3}, mean_dms, global_dms (if donor_ids given)
    """
    from scipy.stats import spearmanr
    from sklearn.metrics import silhouette_score

    valid = ~np.isnan(crypt_labels)
    z_v   = z[valid]
    y_v   = crypt_labels[valid]
    expr_v = expression[valid]
    zones_v = _depths_to_zones(y_v)

    metrics: Dict[str, float] = {}

    # Spearman ρ (radius vs depth)
    radii = np.linalg.norm(z_v, axis=1)
    rho, pval = spearmanr(radii, y_v)
    metrics["spearman_rho"]  = float(rho)
    metrics["spearman_pval"] = float(pval)

    # Radial spread
    metrics.update(radial_summary(z_v, y_v, zones_v))

    # NP@15 global + per-zone
    metrics.update(compute_np15_per_zone(z_v, expr_v, zones_v, k=k_nbrs))

    # Zone silhouette
    n_uniq = len(np.unique(zones_v))
    if n_uniq >= 2:
        try:
            sil = silhouette_score(z_v, zones_v, metric="euclidean")
            metrics["silhouette"] = float(sil)
        except Exception:
            metrics["silhouette"] = float("nan")
    else:
        metrics["silhouette"] = float("nan")

    # DMS (requires donor_ids)
    if donor_ids is not None:
        donor_v = donor_ids[valid]
        metrics.update(compute_dms(z_v, zones_v, donor_v, k=k_nbrs))
    else:
        for k in ["dms_z0", "dms_z1", "dms_z2", "dms_z3", "mean_dms", "global_dms"]:
            metrics[k] = float("nan")

    return metrics


def log_metrics(metrics: Dict[str, float], label: str = "") -> None:
    """Print a one-line summary of the key Phase 2B metrics."""
    prefix = f"[{label}] " if label else ""
    logger.info(
        "%sρ=%.4f  NP@15=%.3f  Sil=%.3f  r_std=%.3f  radial_gap=%.3f  "
        "DMS=%.3f  mean_DMS=%.3f",
        prefix,
        metrics.get("spearman_rho", float("nan")),
        metrics.get("np15_global", float("nan")),
        metrics.get("silhouette", float("nan")),
        metrics.get("r_std", float("nan")),
        metrics.get("radial_gap", float("nan")),
        metrics.get("global_dms", float("nan")),
        metrics.get("mean_dms", float("nan")),
    )
