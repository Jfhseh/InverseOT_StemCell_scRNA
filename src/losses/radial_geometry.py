"""
Differentiable intrinsic radial geometry regularizers for Phase 2B.

These losses address the radial compression failure mode observed in Phase 2 V1,
where all cells collapsed to r_std ≈ 0 while Spearman ρ remained high due to
rank preservation within a very narrow radial band.

Losses
------
zone_radial_center_loss
    Pull each zone's mean Euclidean norm toward a target radius.
    Prevents all zones from collapsing to the same radius band.

adjacent_radial_margin_loss
    Enforce a minimum radial gap between adjacent zones.
    softplus(margin - (r_bar_{k+1} - r_bar_k)) for each adjacent pair.

within_zone_radius_variance_floor  [optional, disabled by default]
    Penalize zones with radius std < sigma_min.
    Keeps within-zone diversity; do not set too high or it fights IOT.

boundary_penalty
    Penalize Euclidean norms approaching max_norm.
    Prevents the boundary collapse seen in unsupervised training.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Hyperbolic intrinsic radius (for reference; losses use Euclidean norm)
# ---------------------------------------------------------------------------

def hyperbolic_radius(z: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Intrinsic hyperbolic radius from origin: r = 2 * atanh(clamp(||z||, 1-eps)).

    For Poincaré ball with curvature c=1:
        d_H(0, z) = 2 * atanh(||z||)

    Note: evaluation metrics use Euclidean norm ||z||, not r.
    This function is for analysis; training losses use norm directly.
    """
    norms = z.norm(dim=-1).clamp(max=1.0 - eps)
    return 2.0 * torch.atanh(norms)


# ---------------------------------------------------------------------------
# Zone mean radius utility
# ---------------------------------------------------------------------------

def _zone_mean_radii(z: torch.Tensor, zones: torch.Tensor) -> dict:
    """Compute mean Euclidean norm per zone; returns dict {zone_id: mean_r tensor}."""
    radii = z.norm(dim=-1)
    r_bar = {}
    for k in range(4):
        mask = (zones == k)
        if mask.sum() >= 1:
            r_bar[k] = radii[mask].mean()
    return r_bar


# ---------------------------------------------------------------------------
# Loss A: zone radial center loss
# ---------------------------------------------------------------------------

def zone_radial_center_loss(
    z: torch.Tensor,
    zones: torch.Tensor,
    target_centers: list[float] | None = None,
) -> torch.Tensor:
    """Pull each zone's mean radius toward its target.

    L = mean_k (r_bar_k - target_k)^2

    Parameters
    ----------
    z              : (B, D) Poincaré ball embeddings
    zones          : (B,) integer zone indices {0,1,2,3}
    target_centers : list of 4 target radii for zones 0→3.
                     Default: [0.15, 0.35, 0.55, 0.75]

    Returns
    -------
    Scalar loss tensor.
    """
    if target_centers is None:
        target_centers = [0.15, 0.35, 0.55, 0.75]

    r_bar = _zone_mean_radii(z, zones)
    if not r_bar:
        return z.sum() * 0.0

    loss = z.new_zeros(1).squeeze()
    n = 0
    for k, target in enumerate(target_centers):
        if k in r_bar:
            loss = loss + (r_bar[k] - target) ** 2
            n += 1

    return loss / max(n, 1)


# ---------------------------------------------------------------------------
# Loss B: adjacent radial margin loss
# ---------------------------------------------------------------------------

def adjacent_radial_margin_loss(
    z: torch.Tensor,
    zones: torch.Tensor,
    margin: float = 0.1,
) -> torch.Tensor:
    """Enforce minimum radial gap between adjacent zones.

    For each adjacent pair (k, k+1) present in the batch:
        L += softplus(margin - (r_bar_{k+1} - r_bar_k))

    softplus is near zero when the gap exceeds the margin; ramps up when
    zones collapse radially.

    Parameters
    ----------
    z      : (B, D) Poincaré ball embeddings
    zones  : (B,) integer zone indices {0,1,2,3}
    margin : minimum desired radial gap between adjacent zones

    Returns
    -------
    Scalar loss tensor.
    """
    r_bar = _zone_mean_radii(z, zones)

    loss = z.new_zeros(1).squeeze()
    n = 0
    for k in range(3):
        if k in r_bar and (k + 1) in r_bar:
            gap = r_bar[k + 1] - r_bar[k]
            loss = loss + F.softplus(margin - gap)
            n += 1

    return loss / max(n, 1)


# ---------------------------------------------------------------------------
# Loss D: within-zone radius variance floor  [optional]
# ---------------------------------------------------------------------------

def within_zone_radius_variance_floor(
    z: torch.Tensor,
    zones: torch.Tensor,
    sigma_min: float = 0.05,
) -> torch.Tensor:
    """Penalize zones where radius std < sigma_min.

    L = mean_k softplus(sigma_min - std_k)

    Keep this weak (lambda ≤ 0.1) — it conflicts with the IOT objective
    which tries to pull same-zone cells to the same radius.
    Set lambda = 0 when lambda_iot is large.

    Parameters
    ----------
    z         : (B, D) Poincaré ball embeddings
    zones     : (B,) integer zone indices {0,1,2,3}
    sigma_min : target minimum radius std per zone

    Returns
    -------
    Scalar loss tensor.
    """
    radii = z.norm(dim=-1)
    loss = z.new_zeros(1).squeeze()
    n = 0
    for k in range(4):
        mask = (zones == k)
        if mask.sum() >= 2:
            std_k = radii[mask].std()
            loss = loss + F.softplus(sigma_min - std_k)
            n += 1

    return loss / max(n, 1)


# ---------------------------------------------------------------------------
# Loss E: boundary penalty
# ---------------------------------------------------------------------------

def boundary_penalty(
    z: torch.Tensor,
    threshold: float = 0.90,
) -> torch.Tensor:
    """Penalize Euclidean norms that exceed threshold.

    L = mean softplus(||z_i|| - threshold)

    The encoder's _clip_tangent already hard-clips at max_norm=0.95, so
    this loss acts as a soft regularizer in the region (threshold, max_norm).
    Set threshold slightly below max_norm (default 0.90 < 0.95).

    Parameters
    ----------
    z         : (B, D) Poincaré ball embeddings
    threshold : soft upper bound on ||z||

    Returns
    -------
    Scalar loss tensor.
    """
    norms = z.norm(dim=-1)
    return F.softplus(norms - threshold).mean()
