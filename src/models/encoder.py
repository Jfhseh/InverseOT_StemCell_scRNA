"""
MLP encoder that maps per-cell feature vectors into the Poincaré ball.

Key design choices
------------------
* Simple MLP backbone: no graph message-passing in Phase 1.  A GNN variant
  can be swapped in later by subclassing or replacing ``encode_euclidean``.
* Final tangent-space output is mapped to the manifold via ``expmap0``
  (exponential map at the origin).
* Tangent norms are clipped *before* ``expmap0`` to keep all points strictly
  inside the ball (||z|| < max_norm < 1).  This prevents two numerical issues:
    1. Points near the boundary cause hyperbolic distances to blow up.
    2. Gradients through atanh diverge as ||z|| → 1.
* Curvature c is fixed by default; set ``learn_curvature=True`` to make it a
  learnable scalar (experimental — can be unstable early in training).
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

try:
    import geoopt
except ImportError as e:
    raise ImportError(
        "geoopt is required for hyperbolic operations. "
        "Install with: pip install geoopt"
    ) from e


class MLPHyperbolicEncoder(nn.Module):
    """
    MLP → Poincaré ball encoder.

    Architecture::

        Input (D_in)
          └─ Linear → BN → ReLU → Dropout
          └─ ...  (one block per entry in hidden_dims)
          └─ Linear  →  tangent vector v ∈ T_0 B^n_c
          └─ clip_tangent(v)
          └─ expmap0(v)  →  z ∈ B^n_c

    Parameters
    ----------
    input_dim    : dimensionality of the input feature vector.
    hidden_dims  : widths of hidden layers (e.g. [256, 128]).
    latent_dim   : dimension of the Poincaré ball embedding.
    dropout      : dropout probability in each hidden block.
    curvature    : Poincaré ball curvature c > 0.  Higher c = more hierarchy.
    learn_curvature : if True, c is an unconstrained learnable parameter.
    max_norm     : mapped points are guaranteed to have ||z|| < max_norm.
                   0.95 keeps a safe margin from the boundary.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
        dropout: float = 0.1,
        curvature: float = 1.0,
        learn_curvature: bool = False,
        max_norm: float = 0.95,
    ) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.max_norm = max_norm

        # --- manifold ---
        if learn_curvature:
            # Wrap curvature as a ManifoldParameter on the positive reals
            # (geoopt uses Euclidean manifold for positive scalars via softplus)
            c_init = torch.tensor([curvature])
            self._log_c = nn.Parameter(torch.log(c_init))
            self._learn_curvature = True
        else:
            self._learn_curvature = False
            self.register_buffer("_fixed_c", torch.tensor([curvature]))

        # We build the manifold lazily (c may change during training)
        self._base_curvature = curvature

        # --- MLP backbone ---
        layers: list[nn.Module] = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ]
            in_dim = h_dim
        # Final linear → tangent vector at origin (no activation)
        layers.append(nn.Linear(in_dim, latent_dim))
        self.mlp = nn.Sequential(*layers)

        self._init_weights()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @property
    def curvature(self) -> float:
        """Return current curvature as a Python float."""
        if self._learn_curvature:
            return float(torch.exp(self._log_c))
        return float(self._fixed_c)

    def _get_manifold(self) -> geoopt.PoincareBall:
        """Construct manifold with current curvature (cheap object, no params)."""
        return geoopt.PoincareBall(c=self.curvature)

    def _clip_tangent(self, v: torch.Tensor) -> torch.Tensor:
        """
        Scale tangent vectors so that ``expmap0(v)`` stays within ``max_norm``.

        For the Poincaré ball with curvature c, geoopt's expmap0 satisfies:

            expmap_0(v) = tanh(sqrt(c) * ||v||) * v / (sqrt(c) * ||v||)
            ||expmap_0(v)|| = tanh(sqrt(c) * ||v||) / sqrt(c)

        We want ||expmap_0(v)|| < max_norm, so:

            tanh(sqrt(c) * ||v||) < max_norm * sqrt(c)
            ||v|| < atanh(max_norm * sqrt(c)) / sqrt(c)   (:= v_max)

        Any tangent vector exceeding v_max is rescaled to exactly v_max.
        The clamp(max=1.0) ensures we never *scale up* short vectors that are
        already inside the safe region.
        """
        c = self.curvature
        sqrt_c = c ** 0.5
        # atanh argument must be < 1; for typical max_norm ∈ (0, 1) and c > 0
        # this is satisfied as long as max_norm * sqrt(c) < 1.
        target = torch.tensor(self.max_norm * sqrt_c, dtype=v.dtype, device=v.device).clamp(
            max=1.0 - 1e-6
        )
        v_max = torch.atanh(target) / sqrt_c
        v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        # scale ≤ 1: only shrink, never enlarge
        scale = (v_max / v_norm).clamp(max=1.0)
        return v * scale

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def encode_euclidean(self, x: torch.Tensor) -> torch.Tensor:
        """Return the raw tangent vector (before projection to the manifold).

        Useful for debugging or for initialising a Euclidean baseline.
        """
        return self.mlp(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of feature vectors into the Poincaré ball.

        Parameters
        ----------
        x : (B, input_dim) float tensor.

        Returns
        -------
        z : (B, latent_dim) tensor, every row satisfies ||z||_2 < max_norm < 1.
        """
        v = self.mlp(x)                    # tangent vector at origin
        v = self._clip_tangent(v)          # enforce safe norm
        z = self._get_manifold().expmap0(v)  # project onto the ball
        return z

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def hyperbolic_radius(self, z: torch.Tensor) -> torch.Tensor:
        """Euclidean norm of Poincaré ball coordinates — the 'depth' proxy.

        For a Poincaré ball with curvature c, the geodesic distance from the
        origin to z is (2/sqrt(c)) * atanh(sqrt(c) * ||z||_2).
        We return ||z||_2 directly as a simpler, monotone proxy.
        """
        return z.norm(dim=-1)

    @torch.no_grad()
    def encode_dataset(
        self,
        expression: torch.Tensor,
        batch_size: int = 512,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Encode a full dataset in chunks, returning (N, latent_dim) on CPU."""
        self.eval()
        self.to(device)
        chunks = []
        for start in range(0, len(expression), batch_size):
            xb = expression[start : start + batch_size].to(device)
            chunks.append(self(xb).cpu())
        return torch.cat(chunks, dim=0)
