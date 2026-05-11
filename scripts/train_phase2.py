"""
Phase 2 training script: bilevel inverse-OT on SCP2595 MROI spatial data.

Runs three experiments and reports a comparison table:
  1. Phase 1 supervised (MSE, baseline to beat)
  2. Phase 2 IOT only   (λ_crypt=0, λ_iot=1.0)
  3. Phase 2 IOT + MSE  (λ_crypt=0.3, λ_iot=1.0)  — optional hybrid

Usage:
    cd /Users/gavinye/classes/cs2212_project
    source .venv/bin/activate
    python scripts/train_phase2.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_PATH = Path("data/processed/colon_spatial_mroi.h5ad")
OUT_DIR   = Path("outputs/phase2_experiments")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset():
    import anndata as ad
    from src.data.dataset import CryptDataset
    from src.data.preprocessing import normalize_crypt_labels

    logger.info("Loading %s ...", DATA_PATH)
    adata = ad.read_h5ad(DATA_PATH)
    logger.info("  %d spots × %d genes  PCA dim=%s  spatial=%s",
                adata.n_obs, adata.n_vars,
                adata.obsm["X_pca"].shape, adata.obsm["spatial"].shape)

    labels_raw  = adata.obs["crypt_depth"].values.astype(np.float32)
    labels_norm = normalize_crypt_labels(labels_raw)

    dataset = CryptDataset(
        expression    = adata.obsm["X_pca"].astype(np.float32),
        crypt_labels  = labels_norm,
        cell_ids      = np.array(adata.obs_names),
        spatial_coords= adata.obsm["spatial"].astype(np.float32),
        metadata      = {
            "mroi":   adata.obs["mroi_norm"].values,
            "age":    adata.obs["Age"].values,
            "region": adata.obs["Region"].values,
        },
    )
    return dataset, adata


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _make_model(dataset):
    from src.models.encoder import MLPHyperbolicEncoder
    return MLPHyperbolicEncoder(
        input_dim   = dataset.input_dim,
        hidden_dims = [256, 128],
        latent_dim  = 32,
        dropout     = 0.1,
        curvature   = 1.0,
        max_norm    = 0.95,
    )


def train_phase1_baseline(dataset, n_epochs: int = 150) -> tuple:
    """Phase 1 supervised model (MSE crypt loss) for comparison."""
    import torch
    from src.models.encoder import MLPHyperbolicEncoder
    from src.train.config import Phase1Config
    from src.train.trainer import Trainer

    torch.manual_seed(42)
    np.random.seed(42)

    model  = _make_model(dataset)
    config = Phase1Config(
        batch_size       = 256,
        n_epochs         = n_epochs,
        lr               = 1e-3,
        weight_decay     = 1e-4,
        grad_clip        = 1.0,
        lambda_transcript= 1.0,
        lambda_spatial   = 0.0,
        lambda_crypt     = 1.0,
        label_type       = "continuous",
        k_pos            = 10,
        triplet_margin   = 0.5,
        n_neg            = 10,
        eval_every       = n_epochs,
        save_every       = 0,
        device           = "mps",
        output_dir       = str(OUT_DIR),
        run_name         = "p1_baseline",
        seed             = 42,
    )
    trainer = Trainer(config, model, dataset)
    t0 = time.time()
    trainer.train()
    logger.info("Phase 1 baseline finished in %.1f s", time.time() - t0)
    return model, trainer


def train_phase2_iot(
    dataset,
    run_name: str = "p2_iot",
    lambda_iot: float = 1.0,
    lambda_crypt: float = 0.0,
    coupling_mode: str = "gaussian",
    epsilon_ot: float = 0.1,
    sigma_depth: float = 0.15,
    sigma_zone: float = 0.8,
    lambda_unif: float = 0.1,
    lambda_var: float = 0.25,
    lambda_cov: float = 0.1,
    n_epochs: int = 200,
    warm_start_ckpt: str | None = None,
    warm_start_lr_scale: float = 0.3,
) -> tuple:
    """
    warm_start_ckpt: path to a Phase 1 checkpoint (.pt).  When supplied the
        model is initialised from those weights before Phase 2 training begins.
        This is required for pure-IOT runs (λ_crypt=0): the IOT loss shapes
        relative geometry but has no radial signal, so from random init the
        transcript triplet loss drives everything to the boundary.  Warm-starting
        from Phase 1 preserves the radial gradient while IOT refines the coupling.
    warm_start_lr_scale: multiplier applied to lr when warm-starting (default 0.3)
        — fine-tuning rather than re-training.
    """
    import torch
    from src.train.config import Phase2Config
    from src.train.trainer import Phase2Trainer

    torch.manual_seed(42)
    np.random.seed(42)

    model  = _make_model(dataset)
    lr = 1e-3
    if warm_start_ckpt is not None:
        ckpt = torch.load(warm_start_ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        lr = lr * warm_start_lr_scale
        logger.info("Warm-started from %s  (lr scaled to %.4f)", warm_start_ckpt, lr)
    config = Phase2Config(
        batch_size        = 256,
        n_epochs          = n_epochs,
        lr                = lr,
        weight_decay      = 1e-4,
        grad_clip         = 1.0,
        lambda_transcript = 1.0,
        lambda_spatial    = 0.0,
        lambda_crypt      = lambda_crypt,
        lambda_iot        = lambda_iot,
        coupling_mode     = coupling_mode,
        epsilon_ot        = epsilon_ot,
        n_sink_iter       = 20,
        sigma_depth       = sigma_depth,
        sigma_zone        = sigma_zone,
        lambda_unif       = lambda_unif,
        lambda_var        = lambda_var,
        lambda_cov        = lambda_cov,
        gamma_var         = 1.0,
        label_type        = "continuous",
        k_pos             = 10,
        triplet_margin    = 0.5,
        n_neg             = 10,
        eval_every        = n_epochs,
        save_every        = 0,
        device            = "mps",
        output_dir        = str(OUT_DIR),
        run_name          = run_name,
        seed              = 42,
    )
    trainer = Phase2Trainer(config, model, dataset)
    t0 = time.time()
    trainer.train()
    logger.info("%s finished in %.1f s", run_name, time.time() - t0)
    return model, trainer


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, dataset) -> dict:
    import torch
    from scipy.stats import spearmanr
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors

    model.eval()
    with torch.no_grad():
        z = model.encode_dataset(dataset.expression, batch_size=512, device="cpu")
    z_np  = z.numpy()
    depth = dataset.crypt_labels.numpy()
    radius= np.linalg.norm(z_np, axis=1)

    rho, pval = spearmanr(radius, depth)

    def knn(X, k=15):
        nn = NearestNeighbors(n_neighbors=k+1, n_jobs=-1).fit(X)
        _, idx = nn.kneighbors(X)
        return idx[:, 1:]
    ref_nb = knn(dataset.expression.numpy())
    lat_nb = knn(z_np)
    np15   = np.mean([len(set(ref_nb[i]) & set(lat_nb[i])) / 15
                      for i in range(len(z_np))])

    bins = np.array([0.0, 0.125, 0.375, 0.75, 1.01])
    binned = np.digitize(depth, bins) - 1
    sil = silhouette_score(z_np, binned, metric="euclidean")

    # Donor mixing score (DMS): within k-NN, fraction from a different donor.
    # High DMS = donors are well-mixed in the embedding (cross-donor alignment).
    # Uses Age as donor proxy (each age group = distinct biological sample).
    dms = float("nan")
    dms_per_zone: dict = {}
    donor_ids = dataset.metadata.get("age", None)
    if donor_ids is not None:
        donor_ids = np.asarray(donor_ids)
        lat_nb_arr = knn(z_np)
        per_cell = np.array([
            np.mean(donor_ids[lat_nb_arr[i]] != donor_ids[i])
            for i in range(len(z_np))
        ])
        dms = float(per_cell.mean())
        for zone in range(4):
            mask = binned == zone
            if mask.sum() > 0:
                dms_per_zone[f"dms_zone{zone}"] = float(per_cell[mask].mean())

    logger.info("  Spearman ρ=%.4f (p=%.2g)  NP@15=%.4f  Sil=%.4f  "
                "r_mean=%.3f  r_std=%.3f  DMS=%.4f",
                rho, pval, np15, sil, radius.mean(), radius.std(), dms)
    return {
        "spearman_rho": rho, "spearman_pval": pval,
        "np15": np15, "silhouette": sil,
        "radius_mean": radius.mean(), "radius_std": radius.std(),
        "dms": dms,
        **dms_per_zone,
        "z": z_np, "radius": radius, "depth": depth,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def make_plots(results: dict, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    for name, res in results.items():
        if "z" not in res:
            continue
        z = res["z"]; depth = res["depth"]; radius = res["radius"]
        rho = res["spearman_rho"]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Poincaré disk
        ax = axes[0]
        theta = np.linspace(0, 2*np.pi, 300)
        ax.plot(np.cos(theta), np.sin(theta), 'k-', lw=0.6, alpha=0.3)
        sc = ax.scatter(z[:, 0], z[:, 1], c=depth, cmap="viridis",
                        s=3, alpha=0.4, linewidths=0)
        plt.colorbar(sc, ax=ax, label="crypt depth", shrink=0.8)
        ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal")
        ax.set_title(f"{name}\nρ={rho:.3f}")

        # Radius vs depth
        axes[1].scatter(depth, radius, s=2, alpha=0.3, c=depth, cmap="viridis")
        axes[1].set_xlabel("crypt depth"); axes[1].set_ylabel("||z||")
        axes[1].set_title(f"Radius vs depth  ρ={rho:.3f}")

        plt.tight_layout()
        slug = name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
        fig.savefig(out_dir / f"{slug}_poincare.png", dpi=150)
        plt.close(fig)

    logger.info("Plots saved to %s", out_dir)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results: dict) -> None:
    print("\n" + "=" * 78)
    print("PHASE 2 RESULTS — SCP2595 Spatial Transcriptomics (n=7,180 crypt spots)")
    print("=" * 78)
    print(f"{'Model':<42} {'Spearman ρ':>11} {'NP@15':>7} {'Silhouette':>11}")
    print("-" * 78)
    for name, res in results.items():
        if "spearman_rho" not in res:
            continue
        marker = " ◄ phase 2" if "iot" in name.lower() else ""
        print(f"{name:<42} {res['spearman_rho']:>11.4f} "
              f"{res['np15']:>7.4f} {res['silhouette']:>11.4f}{marker}")
    print("=" * 78)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run scripts/prepare_scp2595.py first."
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset, adata = load_dataset()
    all_results: dict = {}

    # --- Phase 1 baseline ---
    logger.info("\n=== Phase 1 baseline (MSE crypt supervision) ===")
    model_p1, _ = train_phase1_baseline(dataset, n_epochs=150)
    all_results["Phase 1 (MSE sup, ours)"] = evaluate_model(model_p1, dataset)

    # --- Phase 2 IOT only ---
    logger.info("\n=== Phase 2: IOT only (λ_crypt=0, λ_iot=1.0) ===")
    model_iot, _ = train_phase2_iot(
        dataset,
        run_name    = "p2_iot_only",
        lambda_iot  = 1.0,
        lambda_crypt= 0.0,
        epsilon_ot  = 0.1,
        sigma_depth = 0.15,
        lambda_unif = 0.1,
        lambda_var  = 0.25,
        lambda_cov  = 0.1,
        n_epochs    = 200,
    )
    all_results["Phase 2 (IOT only)"] = evaluate_model(model_iot, dataset)

    # --- Phase 2 IOT + light MSE hybrid ---
    logger.info("\n=== Phase 2: IOT + MSE hybrid (λ_crypt=0.3, λ_iot=1.0) ===")
    model_hybrid, _ = train_phase2_iot(
        dataset,
        run_name      = "p2_iot_hybrid",
        lambda_iot    = 1.0,
        lambda_crypt  = 0.3,
        coupling_mode = "gaussian",
        epsilon_ot    = 0.1,
        sigma_depth   = 0.15,
        lambda_unif   = 0.1,
        lambda_var    = 0.25,
        lambda_cov    = 0.1,
        n_epochs      = 200,
    )
    all_results["Phase 2 (IOT + MSE hybrid)"] = evaluate_model(model_hybrid, dataset)

    # --- Phase 2 prototype zone-block IOT (cross-donor) ---
    logger.info("\n=== Phase 2: prototype zone-block IOT (cross-donor, λ_iot=1.0) ===")
    model_proto, _ = train_phase2_iot(
        dataset,
        run_name      = "p2_iot_proto",
        lambda_iot    = 1.0,
        lambda_crypt  = 0.0,
        coupling_mode = "zone_block",
        epsilon_ot    = 0.1,
        sigma_zone    = 0.8,
        lambda_unif   = 0.1,
        lambda_var    = 0.25,
        lambda_cov    = 0.1,
        n_epochs      = 200,
    )
    all_results["Phase 2 (prototype zone-block, cross-donor)"] = evaluate_model(model_proto, dataset)

    # --- Phase 2 prototype zone-block IOT + MSE hybrid ---
    logger.info("\n=== Phase 2: prototype zone-block IOT + MSE hybrid ===")
    model_proto_hybrid, _ = train_phase2_iot(
        dataset,
        run_name      = "p2_iot_proto_hybrid",
        lambda_iot    = 1.0,
        lambda_crypt  = 0.3,
        coupling_mode = "zone_block",
        epsilon_ot    = 0.1,
        sigma_zone    = 0.8,
        lambda_unif   = 0.1,
        lambda_var    = 0.25,
        lambda_cov    = 0.1,
        n_epochs      = 200,
    )
    all_results["Phase 2 (proto zone-block + MSE hybrid)"] = evaluate_model(model_proto_hybrid, dataset)

    print_summary(all_results)
    make_plots(all_results, OUT_DIR)

    # Save metrics
    metrics_out = {}
    for name, res in all_results.items():
        metrics_out[name] = {k: float(v) for k, v in res.items()
                             if isinstance(v, (int, float, np.floating))}
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)
    logger.info("Metrics saved to %s/metrics.json", OUT_DIR)


if __name__ == "__main__":
    main()
