"""
Crypt-axis weak supervision loss.

Supervision signal: the hyperbolic radius ||z||_2 should correlate with
crypt-axis position (depth).  Convention: depth=0 (stem-cell base) near the
origin, depth=1 (apex / luminal) at the periphery.  This aligns with the
tree-structure intuition of hyperbolic space: the root is the progenitor
niche, branches radiate outward toward differentiated apex cells.

Two modes are supported depending on the label type:

continuous
    Labels are real-valued depth scores in [0, 1].
    Loss = MSE(radius, depth_score).

ordinal
    Labels are integer bin indices (e.g. 0=subcrypt, 1=base, 2=mid, 3=apex).
    Loss = soft margin ranking:
        for all pairs (i, j) where label_i < label_j,
            max(0, radius_i - radius_j + margin).
    Only pairs within the same batch are considered.

Both modes degrade gracefully when labels are missing (returns 0).
"""

from __future__ import annotations

import torch


def crypt_axis_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    label_type: str = "continuous",
    margin: float = 0.1,
) -> torch.Tensor:
    """
    Crypt-axis weak supervision loss.

    Parameters
    ----------
    z          : (B, D) hyperbolic embeddings.
    labels     : (B,) crypt-axis labels (float; NaN entries are ignored).
    label_type : ``"continuous"`` or ``"ordinal"``.
    margin     : ranking margin (ordinal mode only).

    Returns
    -------
    Scalar loss tensor.
    """
    # Hyperbolic radius: ||z||_2 for each cell in the batch
    radius = z.norm(dim=-1)  # (B,)

    # Filter out cells with NaN labels
    valid = ~torch.isnan(labels)
    if valid.sum() < 2:
        return z.new_zeros(1).squeeze()

    r = radius[valid]
    y = labels[valid]

    if label_type == "continuous":
        return _continuous_loss(r, y)
    elif label_type == "ordinal":
        return _ordinal_ranking_loss(r, y, margin)
    else:
        raise ValueError(f"label_type must be 'continuous' or 'ordinal', got '{label_type}'")


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _continuous_loss(radius: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
    """MSE between hyperbolic radius and normalised depth score."""
    # Normalise depth to [0, 1] within the batch (in case labels aren't
    # pre-normalised), using the batch's own range.
    d_min, d_max = depth.min(), depth.max()
    if d_max > d_min:
        depth_norm = (depth - d_min) / (d_max - d_min)
    else:
        # All labels are identical → no gradient signal, return 0
        return radius.new_zeros(1).squeeze()
    return torch.nn.functional.mse_loss(radius, depth_norm)


def _ordinal_ranking_loss(
    radius: torch.Tensor,
    labels: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """
    Pairwise ranking loss: cells with lower ordinal label should have
    smaller radius.

    Only pairs where label_i < label_j contribute to the loss.

    Complexity: O(B^2) in the worst case; fine for typical batch sizes ≤ 512.
    """
    B = len(labels)
    if B < 2:
        return radius.new_zeros(1).squeeze()

    # (B, B) matrix of label differences; positive where label_j > label_i
    label_diff = labels.unsqueeze(1) - labels.unsqueeze(0)   # label_j - label_i
    # We want: wherever label_j > label_i, radius_j > radius_i
    # Violation when radius_i - radius_j + margin > 0
    pair_mask = label_diff > 0   # (B, B); True where j > i ordinal-wise

    if pair_mask.sum() == 0:
        return radius.new_zeros(1).squeeze()

    # radius_i - radius_j for each pair
    r_diff = radius.unsqueeze(1) - radius.unsqueeze(0)   # r_i - r_j
    violations = (r_diff + margin).clamp(min=0.0)        # (B, B)

    # Average over valid pairs only
    return violations[pair_mask].mean()
