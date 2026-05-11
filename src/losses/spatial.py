"""
Spatial smoothness loss.

Cells that are spatially adjacent (close 2-D Euclidean distance on the slide)
should also be close in hyperbolic latent space.

This is a pure *attraction* loss — it pulls spatial neighbours together.
Repulsion is handled globally by the transcript triplet loss.  The combination
means: "transcriptomically similar cells cluster together; spatially adjacent
cells are additionally pulled close regardless of transcriptomics."

Loss formula::

    L_spatial = mean_{anchor i}  mean_{j in spatial-nbrs(i)} d_hyp(z_i, z_j)

We use within-batch spatial k-NN (same as the transcript loss) to avoid
needing a precomputed global spatial graph during training.
"""

from __future__ import annotations

import torch

from .neighborhood import pairwise_hyp_dist


def spatial_smoothness_loss(
    z: torch.Tensor,
    spatial_coords: torch.Tensor,
    manifold,
    k: int = 6,
) -> torch.Tensor:
    """
    Spatial adjacency consistency loss (within-batch).

    Parameters
    ----------
    z             : (B, D_hyp) hyperbolic embeddings.
    spatial_coords: (B, 2) 2-D slide coordinates for the same batch.
    manifold      : geoopt PoincareBall instance.
    k             : number of spatial neighbours per cell.

    Returns
    -------
    Scalar loss tensor.  Returns zero if B < k+2 (too small a batch).
    """
    B = z.shape[0]
    if B < k + 2:
        return z.new_zeros(1).squeeze()

    # Within-batch spatial pairwise distances
    sp_dist = torch.cdist(spatial_coords.float(), spatial_coords.float())  # (B, B)

    # Top-k spatial neighbours (exclude self)
    _, nn_idx = sp_dist.topk(k + 1, dim=1, largest=False)  # (B, k+1)
    nb_idx = nn_idx[:, 1:]  # (B, k)

    # Pairwise hyperbolic distances
    hyp_dist = pairwise_hyp_dist(z, manifold)  # (B, B)

    # Average hyperbolic distance to spatial neighbours
    # Gather: for each row i, distances to its k spatial neighbours
    nb_hyp_dist = hyp_dist.gather(1, nb_idx)  # (B, k)

    return nb_hyp_dist.mean()
