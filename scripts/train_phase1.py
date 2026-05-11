#!/usr/bin/env python3
"""
Phase 1 training script.

Usage examples
--------------
# Synthetic data (smoke-test / sanity check):
    python scripts/train_phase1.py

# Real AnnData with pre-computed PCA and crypt labels:
    python scripts/train_phase1.py \\
        --data_path data/colon_atlas.h5ad \\
        --crypt_label_key crypt_depth \\
        --label_type continuous \\
        --n_epochs 200

# Override any config field directly:
    python scripts/train_phase1.py \\
        --config configs/phase1_default.yaml \\
        --latent_dim 16 \\
        --batch_size 512

All CLI arguments override the YAML config (if one is provided).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

# Make sure the project root is on the path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import CryptDataset
from src.data.preprocessing import normalize_crypt_labels, pca_reduce, select_hvg
from src.eval.plots import save_all_plots
from src.models.encoder import MLPHyperbolicEncoder
from src.train.config import Phase1Config
from src.train.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train Phase 1 hyperbolic crypt encoder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    # Allow any Phase1Config field to be overridden on the command line
    p.add_argument("--data_path", type=str)
    p.add_argument("--use_rep", type=str)
    p.add_argument("--spatial_key", type=str)
    p.add_argument("--crypt_label_key", type=str)
    p.add_argument("--label_type", choices=["continuous", "ordinal"])
    p.add_argument("--n_hvg", type=int)
    p.add_argument("--n_pca_components", type=int)
    p.add_argument("--latent_dim", type=int)
    p.add_argument("--hidden_dims", type=int, nargs="+")
    p.add_argument("--curvature", type=float)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--n_epochs", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--lambda_transcript", type=float)
    p.add_argument("--lambda_spatial", type=float)
    p.add_argument("--lambda_crypt", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--device", type=str)
    p.add_argument("--output_dir", type=str)
    p.add_argument("--run_name", type=str)
    return p


def _merge_args(config: Phase1Config, args: argparse.Namespace) -> Phase1Config:
    """Override config fields with any non-None CLI arguments."""
    for field in config.__dataclass_fields__:
        cli_val = getattr(args, field, None)
        if cli_val is not None:
            setattr(config, field, cli_val)
    return config


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def _load_dataset(cfg: Phase1Config) -> CryptDataset:
    """Load CryptDataset from the path specified in the config."""
    if not cfg.data_path:
        logger.info("No data_path provided — using a synthetic dataset for testing.")
        return _make_synthetic_dataset(cfg)

    path = Path(cfg.data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    if path.suffix == ".npz":
        return CryptDataset.from_npz(path)

    if path.suffix in (".h5ad", ".h5"):
        import anndata as ad
        logger.info("Loading AnnData from %s", path)
        adata = ad.read_h5ad(path)
        logger.info("  cells=%d  obs=%s", adata.n_obs, list(adata.obs.columns[:8]))

        # Optional preprocessing when working with raw expression
        if cfg.use_rep == "X":
            X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.array(adata.X)
            if cfg.n_hvg > 0:
                X, _ = select_hvg(X, n_top_genes=cfg.n_hvg)
            if cfg.n_pca_components > 0:
                X, _ = pca_reduce(X, n_components=cfg.n_pca_components)
            import anndata as ad
            adata.obsm["X_pca"] = X
            cfg.use_rep = "X_pca"

        dataset = CryptDataset.from_anndata(
            adata,
            use_rep=cfg.use_rep,
            spatial_key=cfg.spatial_key,
            crypt_label_key=cfg.crypt_label_key,
            metadata_keys=cfg.metadata_keys or [],
        )

        if dataset.has_labels and cfg.normalize_labels:
            labels_np = normalize_crypt_labels(dataset.crypt_labels.numpy())
            dataset.crypt_labels = torch.as_tensor(labels_np, dtype=torch.float32)

        return dataset

    raise ValueError(f"Unsupported file format: {path.suffix}")


def _make_synthetic_dataset(cfg: Phase1Config, n_cells: int = 512) -> CryptDataset:
    """Create a small synthetic dataset when no real data path is given."""
    rng = np.random.default_rng(cfg.seed)
    n_features = cfg.hidden_dims[0] if cfg.hidden_dims else 50
    expression = rng.standard_normal((n_cells, n_features)).astype(np.float32)
    spatial = rng.random((n_cells, 2)).astype(np.float32)
    labels = np.linspace(0.0, 1.0, n_cells, dtype=np.float32)
    logger.info(
        "Synthetic dataset: %d cells × %d features  (replace with real data for science)",
        n_cells, n_features,
    )
    return CryptDataset(expression, spatial_coords=spatial, crypt_labels=labels)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Build config: start from YAML (or defaults), then apply CLI overrides
    if args.config:
        cfg = Phase1Config.from_yaml(args.config)
        logger.info("Config loaded from %s", args.config)
    else:
        cfg = Phase1Config()

    cfg = _merge_args(cfg, args)
    logger.info("Effective config:\n%s", cfg)

    # Reproducibility
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Load data
    dataset = _load_dataset(cfg)
    logger.info(
        "Dataset: %d cells  input_dim=%d  spatial=%s  labels=%s",
        len(dataset), dataset.input_dim, dataset.has_spatial, dataset.has_labels,
    )

    # Build model
    model = MLPHyperbolicEncoder(
        input_dim=dataset.input_dim,
        hidden_dims=cfg.hidden_dims,
        latent_dim=cfg.latent_dim,
        dropout=cfg.dropout,
        curvature=cfg.curvature,
        learn_curvature=cfg.learn_curvature,
        max_norm=cfg.max_norm,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model: %d trainable parameters", n_params)

    # Save config alongside outputs for reproducibility
    out_dir = Path(cfg.output_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_yaml(out_dir / "config.yaml")

    # Train
    trainer = Trainer(cfg, model, dataset)
    trainer.train()

    # Post-training plots
    try:
        z_np = np.load(out_dir / "embeddings.npy")
        labels_np = dataset.crypt_labels.numpy() if dataset.has_labels else None
        save_all_plots(z_np, labels_np, out_dir, prefix=cfg.run_name)
        logger.info("Plots saved to %s", out_dir)
    except Exception as exc:
        logger.warning("Could not generate plots: %s", exc)


if __name__ == "__main__":
    main()
