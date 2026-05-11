"""
Multi-scale target coupling construction for Phase 2B inverse OT.

Target coupling:
    P_star = normalize(
        w_zone     * P_zone
      + w_adjacent * P_adjacent
      + w_expr     * P_expr
    )

Definitions
-----------
P_zone[i,j]     = 1.0 if zone(i) == zone(j), else 0.0
P_adjacent[i,j] = exp(-|zone(i) - zone(j)| / tau_zone)
P_expr[i,j]     = exp(-sq_dist_norm(i,j) / tau_expr)
                   masked to |zone(i) - zone(j)| <= max_zone_gap

Normalization: combined P_raw is made doubly stochastic via log-domain
Sinkhorn with uniform marginals a = b = 1/B.

Phase 2B design rationale (from phase2B.md)
-------------------------------------------
* P_zone  (w=1.0): aligns ALL same-zone cells across donors by construction.
* P_adjacent (w=0.5): soft zone-distance decay; preserves crypt ordering.
* P_expr (w=0.25): within-zone/adjacent-zone transcript similarity; keeps
  local transcript structure and prevents NP@15 collapse seen in V1.
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# Sinkhorn (self-contained copy to avoid circular imports with inverse_ot)
# ---------------------------------------------------------------------------

def _log_sinkhorn(log_C: torch.Tensor, n_iter: int = 50) -> torch.Tensor:
    """Log-domain Sinkhorn with uniform marginals a = b = 1/B."""
    B = log_C.shape[0]
    log_r = -torch.full((B,), float(B), device=log_C.device, dtype=log_C.dtype).log()
    log_u = torch.zeros(B, device=log_C.device, dtype=log_C.dtype)
    log_v = torch.zeros(B, device=log_C.device, dtype=log_C.dtype)
    for _ in range(n_iter):
        log_u = log_r - torch.logsumexp(log_C + log_v.unsqueeze(0), dim=1)
        log_v = log_r - torch.logsumexp(log_C + log_u.unsqueeze(1), dim=0)
    return log_u.unsqueeze(1) + log_C + log_v.unsqueeze(0)


# ---------------------------------------------------------------------------
# Zone utility (public — imported by trainer and other modules)
# ---------------------------------------------------------------------------

_ZONE_BOUNDARIES = [0.125, 0.375, 0.75]


def depth_to_zone(depths: torch.Tensor) -> torch.Tensor:
    """Map continuous crypt depths in [0,1] to integer zone indices {0,1,2,3}.

    Boundaries follow SCP2595 MROI labeling:
        zone 0  sub-crypt  depth < 0.125
        zone 1  crypt base 0.125 ≤ depth < 0.375
        zone 2  crypt mid  0.375 ≤ depth < 0.75
        zone 3  crypt apex depth ≥ 0.75
    """
    zones = torch.zeros(len(depths), dtype=torch.long, device=depths.device)
    for i, boundary in enumerate(_ZONE_BOUNDARIES):
        zones[depths >= boundary] = i + 1
    return zones


# ---------------------------------------------------------------------------
# P_zone: binary same-zone compatibility
# ---------------------------------------------------------------------------

def build_zone_compatibility(
    zones_source: torch.Tensor,
    zones_target: torch.Tensor,
) -> torch.Tensor:
    """Binary same-zone indicator: P[i,j] = 1.0 iff zone(i) == zone(j)."""
    return (zones_source.unsqueeze(1) == zones_target.unsqueeze(0)).float()


def build_cross_donor_zone_compatibility(
    zones_source: torch.Tensor,
    zones_target: torch.Tensor,
    donors_source: torch.Tensor,
    donors_target: torch.Tensor,
) -> torch.Tensor:
    """Cross-donor same-zone indicator: P[i,j] = 1.0 iff zone(i)==zone(j) AND donor(i)!=donor(j).

    This is the donor-aware replacement for build_zone_compatibility.  By zeroing
    out same-donor same-zone entries, IOT only pulls CROSS-DONOR same-zone cells
    together, leaving within-donor variation (and thus r_std) intact.
    """
    same_zone = (zones_source.unsqueeze(1) == zones_target.unsqueeze(0))
    diff_donor = (donors_source.unsqueeze(1) != donors_target.unsqueeze(0))
    return (same_zone & diff_donor).float()


# ---------------------------------------------------------------------------
# P_adjacent: soft zone-distance compatibility
# ---------------------------------------------------------------------------

def build_adjacent_zone_compatibility(
    zones_source: torch.Tensor,
    zones_target: torch.Tensor,
    tau_zone: float = 1.0,
) -> torch.Tensor:
    """Soft compatibility: P[i,j] = exp(-|zone(i) - zone(j)| / tau_zone).

    tau_zone = 1.0 → same-zone=1.0, adjacent=0.37, skip-one=0.14, opposite=0.05.
    """
    diff = (zones_source.float().unsqueeze(1) - zones_target.float().unsqueeze(0)).abs()
    return torch.exp(-diff / tau_zone)


# ---------------------------------------------------------------------------
# P_expr: transcript similarity within same or adjacent zones
# ---------------------------------------------------------------------------

def build_transcript_compatibility(
    expr_source: torch.Tensor,
    expr_target: torch.Tensor,
    zones_source: torch.Tensor,
    zones_target: torch.Tensor,
    tau_expr: float = 1.0,
    max_zone_gap: int = 1,
) -> torch.Tensor:
    """Transcript similarity, masked to same/adjacent zones.

    P[i,j] = exp(-sq_dist_norm(i,j) / tau_expr) if |zone(i)-zone(j)| <= max_zone_gap
              else 0.

    sq_dist is normalized by the batch median to make tau_expr scale-invariant
    (PCA row norms vary ~5× across datasets).
    """
    sq_dist = torch.cdist(expr_source.float(), expr_target.float()).pow(2)
    sq_dist_norm = sq_dist / sq_dist.median().clamp(min=1e-4)
    P = torch.exp(-sq_dist_norm / tau_expr)
    zone_diff = (zones_source.unsqueeze(1) - zones_target.unsqueeze(0)).abs()
    return P * (zone_diff <= max_zone_gap).float()


# ---------------------------------------------------------------------------
# Combined multi-scale target coupling
# ---------------------------------------------------------------------------

def build_multiscale_target_coupling(
    zones: torch.Tensor,
    expr: torch.Tensor,
    donor_ids: torch.Tensor | None = None,
    tau_zone: float = 1.0,
    tau_expr: float = 1.0,
    w_zone: float = 1.0,
    w_adjacent: float = 0.5,
    w_expr: float = 0.25,
    max_zone_gap: int = 1,
    n_sink_iter: int = 50,
) -> torch.Tensor:
    """Build multi-scale doubly-stochastic target coupling P_star.

    When donor_ids is provided (donor-aware mode):
        P_star = Sinkhorn_normalize(
            w_zone * P_cross_donor_zone + w_adjacent * P_adjacent + w_expr * P_expr
        )
    where P_cross_donor_zone[i,j] = 1 iff zone(i)==zone(j) AND donor(i)!=donor(j).
    This prevents IOT from collapsing within-donor same-zone variation (which would
    set r_std → 0), while still driving cross-donor same-zone alignment (DMS).

    Without donor_ids (donor-agnostic mode, original behavior):
        P_star = Sinkhorn_normalize(
            w_zone * P_zone + w_adjacent * P_adjacent + w_expr * P_expr
        )

    All computation is done under no_grad — P_star is a fixed target.

    Parameters
    ----------
    zones        : (B,) integer zone indices {0,1,2,3}
    expr         : (B, D) expression features (used for P_expr)
    donor_ids    : (B,) integer donor IDs. If provided, use cross-donor P_zone.
    tau_zone     : zone adjacency bandwidth (P_adjacent)
    tau_expr     : expression distance bandwidth (P_expr, relative to median)
    w_zone       : weight for zone identity component
    w_adjacent   : weight for soft zone-distance component
    w_expr       : weight for transcript similarity component
    max_zone_gap : max zone distance for P_expr mask
    n_sink_iter  : Sinkhorn iterations for normalization

    Returns
    -------
    (B, B) doubly-stochastic coupling tensor (no grad).
    """
    with torch.no_grad():
        if donor_ids is not None:
            P_zone = build_cross_donor_zone_compatibility(zones, zones, donor_ids, donor_ids)
        else:
            P_zone = build_zone_compatibility(zones, zones)
        P_adjacent = build_adjacent_zone_compatibility(zones, zones, tau_zone=tau_zone)
        P_expr     = build_transcript_compatibility(
            expr, expr, zones, zones, tau_expr=tau_expr, max_zone_gap=max_zone_gap
        )

        P_raw = w_zone * P_zone + w_adjacent * P_adjacent + w_expr * P_expr
        P_raw = P_raw.clamp(min=1e-9)

        log_P = _log_sinkhorn(P_raw.log(), n_iter=n_sink_iter)
        return log_P.exp()
