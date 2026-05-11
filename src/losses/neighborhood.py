"""
Transcript neighborhood preservation loss.

Strategy: within-batch triplet loss using batched semi-hard negative mining.

For each anchor we identify its k closest neighbours in the input feature space
(PCA / HVG embedding) as positives, and the next n_neg closest as semi-hard
negatives.  This is fully vectorised — no per-cell Python loops — and runs
efficiently on MPS.

Loss formula::

    L = mean_{anchor i}  mean_{p in pos(i)}  mean_{n in neg(i)}
            max(0, d_hyp(i, p) - d_hyp(i, n) + margin)

Negatives are the n_neg nearest non-positives (semi-hard mining), which gives
a stronger gradient signal than random sampling and avoids per-cell Python
overhead that serialises MPS dispatch.
"""

from __future__ import annotations

import torch


def pairwise_hyp_dist(z: torch.Tensor, manifold) -> torch.Tensor:
    """(B, B) pairwise hyperbolic distance matrix via broadcasting."""
    return manifold.dist(z.unsqueeze(1), z.unsqueeze(0))


def transcript_neighbor_loss(
    z: torch.Tensor,
    x_ref: torch.Tensor,
    manifold,
    k: int = 5,
    margin: float = 0.5,
    n_neg: int = 8,
) -> torch.Tensor:
    """
    Transcript neighborhood preservation loss (vectorised within-batch triplet).

    Parameters
    ----------
    z       : (B, D_hyp) hyperbolic embeddings.
    x_ref   : (B, D_ref) reference feature vectors (PCA / input space).
    manifold: geoopt PoincareBall instance.
    k       : positives per anchor (k nearest in input space).
    margin  : triplet margin.
    n_neg   : semi-hard negatives per anchor (k+1 … k+n_neg nearest).

    Returns
    -------
    Scalar loss tensor.
    """
    B = z.shape[0]
    n_neg = min(n_neg, B - k - 1)
    if B < k + 2 or n_neg <= 0:
        return z.new_zeros(1).squeeze()

    ref_dist = torch.cdist(x_ref.float(), x_ref.float())   # (B, B)
    hyp_dist = pairwise_hyp_dist(z, manifold)              # (B, B)

    # Exclude self by inflating diagonal, then take k positives + n_neg negatives
    # in a single topk call.  Positions 0…k-1 → positives; k…k+n_neg-1 → negatives.
    ref_ns = ref_dist + torch.eye(B, device=z.device, dtype=ref_dist.dtype) * 1e9
    _, nn_idx = ref_ns.topk(k + n_neg, dim=1, largest=False)  # (B, k+n_neg)

    pos_idx = nn_idx[:, :k]          # (B, k)
    neg_idx = nn_idx[:, k:k + n_neg] # (B, n_neg)

    d_pos = hyp_dist.gather(1, pos_idx)   # (B, k)
    d_neg = hyp_dist.gather(1, neg_idx)   # (B, n_neg)

    # Triplet: (B, k, n_neg)
    triplet = (d_pos.unsqueeze(2) - d_neg.unsqueeze(1) + margin).clamp(min=0.0)
    return triplet.mean()


def within_zone_triplet_loss(
    z: torch.Tensor,
    x_ref: torch.Tensor,
    zones: torch.Tensor,
    manifold,
    k: int = 5,
    margin: float = 0.5,
    n_neg: int = 8,
) -> torch.Tensor:
    """
    Within-zone transcript neighborhood triplet loss (vectorised).

    For each anchor in zone q:
      - Positives: k nearest in expression space within the SAME zone.
      - Negatives: next n_neg nearest within the same zone (semi-hard).

    Iterates over at most 4 zones (not over B cells), so MPS dispatch overhead
    is O(n_zones) ≤ 4 instead of O(B).

    Parameters
    ----------
    z       : (B, D_hyp) hyperbolic embeddings.
    x_ref   : (B, D_ref) reference feature vectors (PCA space).
    zones   : (B,) integer zone indices {0,1,2,3}.
    manifold: geoopt PoincareBall instance.
    k       : positives per anchor within the zone.
    margin  : triplet margin.
    n_neg   : semi-hard negatives per anchor within the zone.

    Returns
    -------
    Scalar loss tensor (mean over zones with sufficient cells).
    """
    hyp_dist = pairwise_hyp_dist(z, manifold)              # (B, B)
    ref_dist = torch.cdist(x_ref.float(), x_ref.float())   # (B, B)

    zone_losses = []

    for zone_id in zones.unique():
        zone_idx = (zones == zone_id).nonzero(as_tuple=False).squeeze(1)
        n_zone = len(zone_idx)
        n_neg_z = min(n_neg, n_zone - k - 1)
        if n_zone < k + 2 or n_neg_z <= 0:
            continue

        zone_ref = ref_dist[zone_idx][:, zone_idx]   # (n_zone, n_zone)
        zone_hyp = hyp_dist[zone_idx][:, zone_idx]   # (n_zone, n_zone)

        # Inflate diagonal to exclude self
        ref_ns = zone_ref + torch.eye(n_zone, device=z.device, dtype=zone_ref.dtype) * 1e9
        _, nn_local = ref_ns.topk(k + n_neg_z, dim=1, largest=False)  # (n_zone, k+n_neg_z)

        pos_local = nn_local[:, :k]              # (n_zone, k)
        neg_local = nn_local[:, k:k + n_neg_z]  # (n_zone, n_neg_z)

        d_pos = zone_hyp.gather(1, pos_local)    # (n_zone, k)
        d_neg = zone_hyp.gather(1, neg_local)    # (n_zone, n_neg_z)

        triplet = (d_pos.unsqueeze(2) - d_neg.unsqueeze(1) + margin).clamp(min=0.0)
        zone_losses.append(triplet.mean())

    if not zone_losses:
        return z.sum() * 0.0
    return torch.stack(zone_losses).mean()
