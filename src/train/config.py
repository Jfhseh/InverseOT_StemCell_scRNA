"""
Phase 1 training configuration.

Uses a plain Python dataclass so the config can be:
  * imported and mutated programmatically,
  * serialised to / deserialised from YAML (via to_yaml / from_yaml),
  * printed as a clean dict for experiment logging.

All path fields are empty strings by default so that the training script
can inject them without modifying the defaults.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class Phase1Config:
    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    data_path: str = ""
    """Path to an .h5ad AnnData file, or a .npz archive produced by
    CryptDataset.save_npz().  Leave empty to use a synthetic dataset
    (useful for smoke tests)."""

    use_rep: str = "X_pca"
    """Which representation to use as the encoder input.
    'X_pca' → adata.obsm['X_pca']; 'X' → raw expression matrix."""

    spatial_key: Optional[str] = "spatial"
    """obsm key for 2-D spatial coordinates.  Set to None to disable
    the spatial loss (transcript-only mode)."""

    crypt_label_key: Optional[str] = None
    """obs column for crypt-axis labels.  None → no axis supervision."""

    label_type: str = "continuous"
    """'continuous' (MSE on radius) or 'ordinal' (pairwise ranking)."""

    metadata_keys: List[str] = field(default_factory=list)
    """obs columns to store as metadata (donor, section, etc.)."""

    # ------------------------------------------------------------------
    # Preprocessing (only applied when data_path points to raw counts)
    # ------------------------------------------------------------------
    n_hvg: int = 2000
    """Number of highly variable genes to select (0 = skip)."""

    n_pca_components: int = 50
    """Number of PCA components (0 = skip; used only if use_rep='X')."""

    normalize_labels: bool = True
    """Rescale crypt labels to [0, 1] before training."""

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    """MLP hidden layer widths."""

    latent_dim: int = 32
    """Dimension of the Poincaré ball embedding."""

    curvature: float = 1.0
    """Poincaré ball curvature c.  Higher = stronger hierarchy signal."""

    learn_curvature: bool = False
    """Whether to treat curvature as a learnable parameter."""

    dropout: float = 0.1
    max_norm: float = 0.95
    """Maximum allowed ||z||_2 after expmap0 (keeps points off the boundary)."""

    # ------------------------------------------------------------------
    # Loss weights
    # ------------------------------------------------------------------
    lambda_transcript: float = 1.0
    lambda_spatial: float = 0.5
    lambda_crypt: float = 1.0

    # Loss hyper-parameters
    k_pos: int = 5
    """Number of transcript positives per anchor (within-batch k-NN)."""

    k_spatial: int = 6
    """Number of spatial neighbours per anchor (within-batch)."""

    triplet_margin: float = 0.5
    rank_margin: float = 0.1
    n_neg: int = 8
    """Number of negatives sampled per anchor in the triplet loss."""

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    batch_size: int = 256
    n_epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    """Max gradient norm; 0 = disabled."""

    # ------------------------------------------------------------------
    # Evaluation & logging
    # ------------------------------------------------------------------
    eval_every: int = 10
    """Evaluate every N epochs."""

    save_every: int = 50
    """Save a model checkpoint every N epochs (0 = only at end)."""

    # ------------------------------------------------------------------
    # Reproducibility & output
    # ------------------------------------------------------------------
    seed: int = 42
    device: str = "auto"
    """'auto' selects CUDA > MPS > CPU in that order."""

    output_dir: str = "./outputs"
    run_name: str = "phase1"

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        import yaml
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Phase1Config":
        import yaml
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**d)

    @classmethod
    def from_dict(cls, d: dict) -> "Phase1Config":
        return cls(**{k: v for k, v in d.items() if k in {f.name for f in dataclasses.fields(cls)}})

    def __repr__(self) -> str:  # pretty-print for logging
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class Phase2Config(Phase1Config):
    """Phase 2 config: extends Phase1Config with inverse-OT hyperparameters.

    By default, the hard crypt-axis MSE loss (lambda_crypt) is disabled and
    replaced by the bilevel IOT loss (lambda_iot).  The transcript triplet loss
    is kept to preserve local neighbourhood structure.
    """

    # Override Phase1 defaults for Phase 2
    lambda_crypt: float = 0.0       # disable hard MSE supervision
    lambda_spatial: float = 0.0     # keep spatial loss off (collapses radial variance)
    n_epochs: int = 200
    run_name: str = "phase2"

    # --- Inverse-OT loss ---
    lambda_iot: float = 1.0
    """Weight for the bilevel IOT loss (KL + regularizers)."""

    epsilon_ot: float = 0.1
    """Entropic regularization for Sinkhorn inner loop (relative to median cost)."""

    n_sink_iter: int = 20
    """Number of Sinkhorn iterations in the inner loop."""

    coupling_mode: str = "gaussian"
    """Target coupling mode: 'gaussian' (smooth depth similarity) or
    'zone_block' (discrete MROI prototype / cross-donor)."""

    sigma_depth: float = 0.15
    """Gaussian coupling bandwidth (coupling_mode='gaussian').
    0.15 ≈ half the gap between adjacent MROI zones (0, 0.25, 0.5, 1.0)."""

    sigma_zone: float = 0.8
    """Zone-block coupling bandwidth in zone units (coupling_mode='zone_block').
    0.8 → same-zone=1.0, adjacent=0.21, skip-one=0.002."""

    # --- Regularization weights ---
    lambda_unif: float = 0.1
    """Weight for Wang-Isola uniformity penalty (spreads embeddings out)."""

    lambda_var: float = 0.25
    """Weight for per-dimension variance penalty (prevents dimension collapse)."""

    lambda_cov: float = 0.1
    """Weight for off-diagonal covariance penalty (decorrelates dimensions)."""

    gamma_var: float = 1.0
    """Target minimum per-dimension std for the variance penalty."""

    lambda_rad_var: float = 0.0
    """Weight for explicit radial variance penalty."""

    gamma_rad_var: float = 0.1
    """Target standard deviation for the hyperbolic radius ||z||."""

    prototype_grouping: str = "zone"
    """Grouping strategy for prototypes: 'zone' or 'zone_age'."""

    # --- Cross-donor zone triplet loss ---
    lambda_cross_donor: float = 0.0
    """Weight for cross-donor zone triplet loss.  0 = disabled."""

    margin_cross_donor: float = 0.5
    """Triplet margin for cross-donor zone triplet loss."""


@dataclass
class Phase2V2Config(Phase2Config):
    """Phase 2B config: multi-scale bilevel IOT with radial geometry regularizers.

    Addresses the two V1 failure modes:
      1. Radial compression (r_std ≈ 0): fixed by zone_radial_center_loss +
         adjacent_radial_margin_loss which enforce radial spread.
      2. NP@15 collapse: partially fixed by within_zone_triplet_loss which
         preserves local transcript structure within each MROI zone.

    V1 regularizers (unif/var/cov) are disabled — the radial geometry losses
    serve the same purpose more directly.
    """

    # Override V1 defaults
    n_epochs: int = 150
    run_name: str = "phase2v2"
    coupling_mode: str = "multiscale"

    # Disable V1 dimension-collapse regularizers (replaced by radial geometry)
    lambda_unif: float = 0.0
    lambda_var: float = 0.0
    lambda_cov: float = 0.0
    lambda_rad_var: float = 0.0

    # --- Multi-scale IOT target coupling ---
    tau_zone: float = 1.0
    """Zone adjacency bandwidth for P_adjacent component."""

    tau_expr: float = 1.0
    """Expression distance bandwidth for P_expr (relative to batch median sq-dist)."""

    w_zone: float = 1.0
    """Weight for same-zone binary component P_zone."""

    w_adjacent: float = 0.5
    """Weight for soft zone-distance component P_adjacent."""

    w_expr: float = 0.25
    """Weight for transcript similarity component P_expr."""

    max_zone_gap: int = 1
    """Max zone distance for P_expr mask (1 = same + adjacent zones only)."""

    n_star_sink_iter: int = 50
    """Sinkhorn iterations to normalize P_star target coupling."""

    # --- Radial geometry losses ---
    lambda_radial: float = 1.0
    """Combined weight for radial center + margin losses."""

    eta_margin: float = 0.5
    """Sub-weight for adjacent_radial_margin_loss within lambda_radial."""

    zone_target_centers: List[float] = field(
        default_factory=lambda: [0.15, 0.35, 0.55, 0.75]
    )
    """Target Euclidean norms for zones 0→3. Sets the absolute radial positions."""

    radial_margin: float = 0.1
    """Minimum desired radial gap between adjacent zones (margin loss)."""

    lambda_radius_var_floor: float = 0.0
    """Within-zone radius variance floor weight. Keep at 0 while lambda_iot > 0."""

    sigma_min: float = 0.05
    """Target minimum per-zone radius std for variance floor loss."""

    # --- Boundary penalty ---
    lambda_boundary: float = 0.01
    """Weight for boundary penalty (soft upper bound on ||z||)."""

    boundary_threshold: float = 0.90
    """Soft upper bound on Euclidean norm (encoder hard-clips at max_norm=0.95)."""

    # --- Within-zone local transcript preservation ---
    lambda_local: float = 0.25
    """Weight for within-zone transcript neighborhood triplet loss."""

    k_local: int = 5
    """Positives per anchor in within-zone triplet loss."""

    n_neg_local: int = 8
    """Negatives per anchor in within-zone triplet loss."""
