"""
Experiment script: compare three models on real colon crypt snRNA-seq data.

Models compared
---------------
1. PCA baseline      : Euclidean PCA, no learning. Reports max abs(Spearman ρ)
                       across the top-10 PCs vs. crypt depth.  Upper bound for
                       linear methods.
2. Hyperbolic, unsup : MLP encoder + transcript triplet loss only (λ_crypt=0).
                       Spirit of Poincaré Maps — no axis supervision.
3. Hyperbolic, sup   : Same encoder + transcript triplet + crypt-axis MSE loss
                       (λ_crypt=1.0).  Our Phase 1 method.

Primary metric: Spearman ρ between ||z||_2 and crypt_depth label.
Secondary: neighbourhood preservation and silhouette score.

Usage
-----
    python scripts/run_experiments.py
"""

from __future__ import annotations

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

DATA_PATH = Path("data/processed/colon_epithelial.h5ad")
OUT_DIR = Path("outputs/real_data_experiments")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_dataset():
    import anndata as ad
    from src.data.dataset import CryptDataset
    from src.data.preprocessing import normalize_crypt_labels

    logger.info("Loading %s ...", DATA_PATH)
    adata = ad.read_h5ad(DATA_PATH)
    logger.info("  %d cells × %d HVGs  PCA dim: %s",
                adata.n_obs, adata.n_vars, adata.obsm["X_pca"].shape)

    labels_raw = adata.obs["crypt_depth"].values.astype(np.float32)
    labels_norm = normalize_crypt_labels(labels_raw)

    dataset = CryptDataset(
        expression=adata.obsm["X_pca"].astype(np.float32),
        crypt_labels=labels_norm,
        cell_ids=np.array(adata.obs_names),
        metadata={"cell_type": adata.obs["pheno_cell_types"].values,
                  "sample_id": adata.obs["sample_id"].values},
    )
    return dataset, adata


def train_model(
    dataset,
    run_name: str,
    lambda_crypt: float,
    label_type: str = "continuous",
    n_epochs: int = 150,
    device: str = "mps",
):
    import torch
    from src.models.encoder import MLPHyperbolicEncoder
    from src.train.config import Phase1Config
    from src.train.trainer import Trainer

    torch.manual_seed(42)
    np.random.seed(42)

    model = MLPHyperbolicEncoder(
        input_dim=dataset.input_dim,
        hidden_dims=[256, 128],
        latent_dim=32,
        dropout=0.1,
        curvature=1.0,
        max_norm=0.95,
    )

    config = Phase1Config(
        batch_size=256,
        n_epochs=n_epochs,
        lr=1e-3,
        weight_decay=1e-4,
        grad_clip=1.0,
        lambda_transcript=1.0,
        lambda_spatial=0.0,     # no spatial coords in snRNA-seq
        lambda_crypt=lambda_crypt,
        label_type=label_type,
        k_pos=10,
        triplet_margin=0.5,
        n_neg=10,
        eval_every=n_epochs,    # only eval at end
        save_every=0,
        device=device,
        output_dir="outputs/real_data_experiments",
        run_name=run_name,
        seed=42,
    )

    trainer = Trainer(config, model, dataset)
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    logger.info("Training finished in %.1f s", elapsed)
    return model, trainer


def pca_baseline(dataset, adata) -> dict:
    """Spearman ρ between each of the top-10 PCA dims and crypt depth."""
    from scipy.stats import spearmanr

    pca = adata.obsm["X_pca"]  # (N, 50)
    depth = dataset.crypt_labels.numpy()

    results = {}
    for i in range(10):
        rho, pval = spearmanr(pca[:, i], depth)
        results[f"PC{i+1}"] = (abs(rho), pval)
        logger.info("  PC%d  |ρ|=%.3f  p=%.2g", i+1, abs(rho), pval)

    best_pc = max(results, key=lambda k: results[k][0])
    best_rho = results[best_pc][0]
    logger.info("Best PCA component: %s  |ρ|=%.3f", best_pc, best_rho)
    return {"pca_best_rho": best_rho, "pca_best_pc": best_pc, "pca_all": results}


def evaluate_model(model, dataset) -> dict:
    """Compute all metrics for a trained model."""
    import torch
    from scipy.stats import spearmanr
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors

    model.eval()
    with torch.no_grad():
        z = model.encode_dataset(dataset.expression, batch_size=512, device="cpu")
    z_np = z.numpy()
    depth = dataset.crypt_labels.numpy()
    radius = np.linalg.norm(z_np, axis=1)

    # Spearman ρ: radius vs crypt depth
    rho, pval = spearmanr(radius, depth)

    # Neighbourhood preservation (k=15)
    pca = dataset.expression.numpy()
    def knn(X, k=15):
        nn = NearestNeighbors(n_neighbors=k+1, n_jobs=-1).fit(X)
        _, idx = nn.kneighbors(X)
        return idx[:, 1:]
    ref_nb = knn(pca)
    lat_nb = knn(z_np)
    np_score = np.mean([
        len(set(ref_nb[i]) & set(lat_nb[i])) / 15
        for i in range(len(pca))
    ])

    # Silhouette on depth bins (4 bins)
    bins = np.array([0, 0.15, 0.45, 0.6, 1.01])
    labels_binned = np.digitize(depth, bins) - 1
    sil = silhouette_score(z_np, labels_binned, metric="euclidean")

    logger.info("  Spearman ρ = %.4f  (p=%.2g)", rho, pval)
    logger.info("  Neighbourhood preservation = %.4f", np_score)
    logger.info("  Silhouette (4 depth bins) = %.4f", sil)
    logger.info("  Radius: mean=%.4f  std=%.4f  max=%.4f",
                radius.mean(), radius.std(), radius.max())

    return {
        "spearman_rho": rho,
        "spearman_pval": pval,
        "neighborhood_preservation": np_score,
        "silhouette": sil,
        "radius_mean": radius.mean(),
        "radius_std": radius.std(),
        "z": z_np,
        "radius": radius,
        "depth": depth,
    }


def make_plots(results: dict, out_dir: Path) -> None:
    """Save Poincaré disk + radius-vs-depth plots for each model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    out_dir.mkdir(parents=True, exist_ok=True)
    colors = plt.cm.viridis

    for name, res in results.items():
        if "z" not in res:
            continue
        z = res["z"]
        depth = res["depth"]
        radius = res["radius"]
        rho = res["spearman_rho"]

        # --- Poincaré disk ---
        fig, ax = plt.subplots(figsize=(6, 6))
        theta = np.linspace(0, 2*np.pi, 300)
        ax.plot(np.cos(theta), np.sin(theta), 'k-', lw=0.6, alpha=0.3)
        sc = ax.scatter(z[:, 0], z[:, 1], c=depth, cmap="viridis",
                        s=4, alpha=0.5, linewidths=0)
        plt.colorbar(sc, ax=ax, label="crypt depth", shrink=0.8)
        ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal")
        ax.set_title(f"{name}\nSpearman ρ={rho:.3f}")
        ax.set_xlabel("z₀"); ax.set_ylabel("z₁")
        plt.tight_layout()
        fig.savefig(out_dir / f"{name}_poincare.png", dpi=150)
        plt.close(fig)

        # --- Radius vs depth (violin by cell type) ---
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].scatter(depth, radius, s=2, alpha=0.3, c=depth, cmap="viridis")
        axes[0].set_xlabel("crypt depth label")
        axes[0].set_ylabel("hyperbolic radius ||z||")
        axes[0].set_title(f"Radius vs depth  ρ={rho:.3f}")

        # Violin by depth bin (bins are in normalized [0,1] space)
        # Normalized depths: Stem=0, TA≈0.23-0.43, Tuft+EEC≈0.63-0.77, Apex≈0.91-1.0
        bins = [0.0, 0.10, 0.55, 0.80, 1.01]
        bin_labels = ["Stem\n(≈0)", "TA\n(0.23-0.43)", "Mid\n(0.63-0.77)", "Apex\n(0.91-1.0)"]
        grouped = [radius[(depth >= bins[i]) & (depth < bins[i+1])] for i in range(len(bins)-1)]
        vp = axes[1].violinplot(grouped, positions=range(len(grouped)), showmedians=True)
        axes[1].set_xticks(range(len(grouped)))
        axes[1].set_xticklabels(bin_labels, fontsize=8)
        axes[1].set_ylabel("hyperbolic radius ||z||")
        axes[1].set_title("Radius distribution by crypt zone")
        plt.tight_layout()
        fig.savefig(out_dir / f"{name}_radius_depth.png", dpi=150)
        plt.close(fig)

    logger.info("Plots saved to %s", out_dir)


def print_summary(pca_res: dict, results: dict) -> None:
    """Print a clean comparison table."""
    print("\n" + "="*70)
    print("PHASE 1 RESULTS — Colon Crypt snRNA-seq (GSE285985, n=8810 epithelial)")
    print("="*70)
    print(f"\nDataset: 3 adult mouse samples (6w, 8w, 6m), 3 donors")
    print(f"         Crypt-axis labels derived from cell-type annotations")
    print(f"         (Stem→0.0, TA→0.20-0.38, Tuft→0.55, EEC→0.65-0.68,")
    print(f"          Colonocyte→0.80, Goblet→0.88)")

    print(f"\n{'Model':<35} {'Spearman ρ':>12} {'NP@15':>8} {'Silhouette':>12}")
    print("-"*70)

    best_pc = pca_res["pca_best_pc"]
    best_rho = pca_res["pca_best_rho"]
    print(f"{'PCA (best of top-10 PCs)':<35} {best_rho:>12.3f}  {'—':>8}  {'—':>12}")

    for name, res in results.items():
        if "spearman_rho" not in res:
            continue
        rho = res["spearman_rho"]
        np_ = res["neighborhood_preservation"]
        sil = res["silhouette"]
        marker = " ◄ our method" if "sup" in name else ""
        print(f"{name:<35} {rho:>12.3f}  {np_:>8.3f}  {sil:>12.3f}{marker}")
    print("="*70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run scripts/prepare_real_data.py first."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset, adata = load_dataset()

    # --- PCA baseline ---
    logger.info("\n=== PCA baseline ===")
    pca_res = pca_baseline(dataset, adata)

    all_results = {}

    # --- Hyperbolic, transcript-only (unsupervised, Poincaré Maps spirit) ---
    logger.info("\n=== Hyperbolic encoder, transcript-only (unsupervised) ===")
    model_unsup, _ = train_model(
        dataset, run_name="hyperbolic_unsup",
        lambda_crypt=0.0, n_epochs=150, device="mps",
    )
    logger.info("Evaluating unsupervised model ...")
    res_unsup = evaluate_model(model_unsup, dataset)
    all_results["Hyperbolic (transcript-only, no sup)"] = res_unsup

    # --- Hyperbolic, with crypt-axis supervision ---
    logger.info("\n=== Hyperbolic encoder, crypt-axis supervised ===")
    model_sup, _ = train_model(
        dataset, run_name="hyperbolic_sup",
        lambda_crypt=1.0, n_epochs=150, device="mps",
    )
    logger.info("Evaluating supervised model ...")
    res_sup = evaluate_model(model_sup, dataset)
    all_results["Hyperbolic (crypt-axis sup, ours)"] = res_sup

    # --- Summary ---
    print_summary(pca_res, all_results)

    # --- Plots ---
    make_plots(all_results, OUT_DIR)

    # --- Save metrics to file ---
    import json
    metrics_out = {}
    metrics_out["pca_best_rho"] = float(pca_res["pca_best_rho"])
    metrics_out["pca_best_pc"] = pca_res["pca_best_pc"]
    for name, res in all_results.items():
        metrics_out[name] = {k: float(v) for k, v in res.items()
                             if isinstance(v, (int, float, np.floating))}
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)
    logger.info("Metrics saved to %s/metrics.json", OUT_DIR)


if __name__ == "__main__":
    main()
