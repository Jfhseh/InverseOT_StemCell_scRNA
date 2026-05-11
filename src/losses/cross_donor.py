"""
Cross-donor zone triplet loss for hyperbolic crypt representations.

For each anchor cell (zone k, donor d):
  - Positive j: same zone k, DIFFERENT donor/age (cross-donor)
  - Negative m: different zone

Encourages age-invariant zone representations, directly targeting the
cross-donor mixing score (DMS) in the held-out LODO evaluation.

The loss is fully hyperbolic: all distances are d_H on the Poincaré ball.
It does not conflict with the MSE crypt loss (both agree on radial ordering)
and applies orthogonal pressure to the transcript triplet loss.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def cross_donor_zone_triplet_loss(
    z: torch.Tensor,
    zones: torch.Tensor,
    ages: torch.Tensor,
    manifold,
    margin: float = 0.5,
) -> torch.Tensor:
    """
    Vectorized soft triplet loss with cross-donor positives.

    For each anchor i in the batch:
      d_pos  = mean d_H(z_i, z_j) over all j with zone(j)==zone(i), age(j)!=age(i)
      d_neg  = mean d_H(z_i, z_m) over all m with zone(m)!=zone(i)
      loss_i = relu(d_pos - d_neg + margin)

    Parameters
    ----------
    z        : (B, D) hyperbolic embeddings in the Poincaré ball.
    zones    : (B,) integer zone indices {0,1,2,3}.
    ages     : (B,) integer donor/age indices.
    manifold : geoopt.PoincareBall instance.
    margin   : triplet margin.

    Returns
    -------
    Scalar loss (mean over anchors that have at least one valid positive and negative).
    """
    B = len(z)
    if B < 2:
        return z.sum() * 0.0

    # Pairwise hyperbolic distances (B, B)
    dist_mat = manifold.dist(z.unsqueeze(1), z.unsqueeze(0))

    same_zone = zones.unsqueeze(1) == zones.unsqueeze(0)   # (B, B)
    same_age  = ages.unsqueeze(1)  == ages.unsqueeze(0)    # (B, B)

    # Positive: same zone, different age
    pos_mask = same_zone & ~same_age
    # Negative: different zone (any age)
    neg_mask = ~same_zone

    pos_count = pos_mask.float().sum(dim=1)   # (B,)
    neg_count = neg_mask.float().sum(dim=1)   # (B,)
    valid = (pos_count > 0) & (neg_count > 0)

    if valid.sum() == 0:
        return z.sum() * 0.0

    # Mean positive and negative distances per anchor
    pos_dist = (dist_mat * pos_mask.float()).sum(dim=1) / pos_count.clamp(min=1)
    neg_dist = (dist_mat * neg_mask.float()).sum(dim=1) / neg_count.clamp(min=1)

    triplet = F.relu(pos_dist - neg_dist + margin)
    return triplet[valid].mean()
