"""
Experiment script: compare three models on SCP2595 spatial transcriptomics data.

Models compared
---------------
1. PCA baseline      : Euclidean PCA, no learning. Reports max abs(Spearman ρ)
                       across top-10 PCs vs. crypt depth.
2. Hyperbolic, unsup : MLP encoder + transcript triplet loss only (λ_crypt=0, λ_spatial=0).
3. Hyperbolic, sup   : Same encoder + transcript triplet + crypt-axis MSE + spatial
                       smoothness loss (λ_crypt=1.0, λ_spatial=0.5).  Our Phase 1 method.

Key difference from snRNA-seq experiments: supervision is from true MROI labels
(crypt apex/mid/base/sub-crypt assigned by histologist from H&E), not cell-type
proxies.  Spatial coordinates also enable the smoothness loss.

Usage
-----
    python scripts/run_mroi_experiments.py
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

DATA_PATH = Path("data/processed/colon_spatial_mroi.h5ad")
OUT_DIR = Path("outputs/mroi_experiments")


def load_dataset():
    import anndata as ad
    from src.data.dataset import CryptDataset
    from src.data.preprocessing import normalize_crypt_labels

    logger.info("Loading %s ...", DATA_PATH)
    adata = ad.read_h5ad(DATA_PATH)
    logger.info("  %d spots × %d genes  PCA dim: %s  spatial: %s",
                adata.n_obs, adata.n_vars,
                adata.obsm["X_pca"].shape, adata.obsm["spatial"].shape)

    labels_raw = adata.obs["crypt_depth"].values.astype(np.float32)
    labels_norm = normalize_crypt_labels(labels_raw)

    dataset = CryptDataset(
        expression=adata.obsm["X_pca"].astype(np.float32),
        crypt_labels=labels_norm,
        cell_ids=np.array(adata.obs_names),
        spatial_coords=adata.obsm["spatial"].astype(np.float32),
        metadata={
            "mroi": adata.obs["mroi_norm"].values,
            "age": adata.obs["Age"].values,
            "region": adata.obs["Region"].values,
        },
    )
    return dataset, adata


def train_model(
    dataset,
    run_name: str,
    lambda_crypt: float,
    lambda_spatial: float = 0.0,
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
        lambda_spatial=lambda_spatial,
        lambda_crypt=lambda_crypt,
        label_type=label_type,
        k_pos=10,
        triplet_margin=0.5,
        n_neg=10,
        eval_every=n_epochs,
        save_every=0,
        device=device,
        output_dir=str(OUT_DIR),
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
    from scipy.stats import spearmanr

    pca = adata.obsm["X_pca"]
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

    rho, pval = spearmanr(radius, depth)

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

    # 4 MROI bins: sub-crypt(0), base(0.25), mid(0.5), apex(1.0)
    bins = np.array([0.0, 0.125, 0.375, 0.75, 1.01])
    labels_binned = np.digitize(depth, bins) - 1
    sil = silhouette_score(z_np, labels_binned, metric="euclidean")

    logger.info("  Spearman ρ = %.4f  (p=%.2g)", rho, pval)
    logger.info("  Neighbourhood preservation = %.4f", np_score)
    logger.info("  Silhouette (4 MROI bins) = %.4f", sil)
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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    for name, res in results.items():
        if "z" not in res:
            continue
        z = res["z"]
        depth = res["depth"]
        radius = res["radius"]
        rho = res["spearman_rho"]

        # Poincaré disk
        fig, ax = plt.subplots(figsize=(6, 6))
        theta = np.linspace(0, 2*np.pi, 300)
        ax.plot(np.cos(theta), np.sin(theta), 'k-', lw=0.6, alpha=0.3)
        sc = ax.scatter(z[:, 0], z[:, 1], c=depth, cmap="viridis",
                        s=4, alpha=0.5, linewidths=0)
        plt.colorbar(sc, ax=ax, label="crypt depth (MROI)", shrink=0.8)
        ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal")
        ax.set_title(f"{name}\nSpearman ρ={rho:.3f}")
        ax.set_xlabel("z₀"); ax.set_ylabel("z₁")
        plt.tight_layout()
        slug = name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
        fig.savefig(out_dir / f"{slug}_poincare.png", dpi=150)
        plt.close(fig)

        # Violin by MROI zone
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].scatter(depth, radius, s=2, alpha=0.3, c=depth, cmap="viridis")
        axes[0].set_xlabel("crypt depth label (MROI)")
        axes[0].set_ylabel("hyperbolic radius ||z||")
        axes[0].set_title(f"Radius vs depth  ρ={rho:.3f}")

        bins = [0.0, 0.125, 0.375, 0.75, 1.01]
        bin_labels = ["Sub-crypt\n(0.0)", "Crypt base\n(0.25)", "Crypt mid\n(0.5)", "Crypt apex\n(1.0)"]
        grouped = [radius[(depth >= bins[i]) & (depth < bins[i+1])] for i in range(len(bins)-1)]
        counts = [len(g) for g in grouped]
        valid_groups = [g for g in grouped if len(g) > 1]
        valid_labels = [f"{bin_labels[i]}\n(n={counts[i]})" for i in range(len(bins)-1) if counts[i] > 1]
        if valid_groups:
            vp = axes[1].violinplot(valid_groups, positions=range(len(valid_groups)), showmedians=True)
            for body in vp['bodies']:
                body.set_alpha(0.6)
        axes[1].set_xticks(range(len(valid_labels)))
        axes[1].set_xticklabels(valid_labels, fontsize=8)
        axes[1].set_ylabel("hyperbolic radius ||z||")
        axes[1].set_title("Radius by MROI zone")
        plt.tight_layout()
        fig.savefig(out_dir / f"{slug}_radius_depth.png", dpi=150)
        plt.close(fig)

    logger.info("Plots saved to %s", out_dir)


def print_summary(pca_res: dict, results: dict) -> None:
    print("\n" + "="*72)
    print("PHASE 1 RESULTS — SCP2595 Spatial Transcriptomics (n=7,180 crypt spots)")
    print("="*72)
    print(f"\nDataset: ~1,500 tissue sections, 11 age groups (0w–2yr), 3 colon regions")
    print(f"         Crypt-axis labels from MROI annotation (histologist-assigned):")
    print(f"         sub-crypt→0.0, crypt base→0.25, crypt mid→0.5, crypt apex→1.0")

    print(f"\n{'Model':<40} {'Spearman ρ':>12} {'NP@15':>8} {'Silhouette':>12}")
    print("-"*72)

    best_pc = pca_res["pca_best_pc"]
    best_rho = pca_res["pca_best_rho"]
    print(f"{'PCA (best of top-10 PCs)':<40} {best_rho:>12.3f}  {'—':>8}  {'—':>12}")

    for name, res in results.items():
        if "spearman_rho" not in res:
            continue
        rho = res["spearman_rho"]
        np_ = res["neighborhood_preservation"]
        sil = res["silhouette"]
        marker = " ◄ our method" if "ours" in name.lower() else ""
        print(f"{name:<40} {rho:>12.3f}  {np_:>8.3f}  {sil:>12.3f}{marker}")
    print("="*72)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run scripts/prepare_scp2595.py first."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset, adata = load_dataset()

    logger.info("\n=== PCA baseline ===")
    pca_res = pca_baseline(dataset, adata)

    all_results = {}

    logger.info("\n=== Hyperbolic encoder, transcript-only (unsupervised) ===")
    model_unsup, _ = train_model(
        dataset, run_name="hyperbolic_unsup",
        lambda_crypt=0.0, lambda_spatial=0.0, n_epochs=150, device="mps",
    )
    res_unsup = evaluate_model(model_unsup, dataset)
    all_results["Hyperbolic (transcript-only, no sup)"] = res_unsup

    logger.info("\n=== Hyperbolic encoder, MROI-supervised (no spatial loss) ===")
    model_sup, _ = train_model(
        dataset, run_name="hyperbolic_sup",
        lambda_crypt=1.0, lambda_spatial=0.0, n_epochs=150, device="mps",
    )
    res_sup = evaluate_model(model_sup, dataset)
    all_results["Hyperbolic (MROI-sup, ours)"] = res_sup

    print_summary(pca_res, all_results)
    make_plots(all_results, OUT_DIR)

    import json
    metrics_out = {"pca_best_rho": float(pca_res["pca_best_rho"]),
                   "pca_best_pc": pca_res["pca_best_pc"]}
    for name, res in all_results.items():
        metrics_out[name] = {k: float(v) for k, v in res.items()
                             if isinstance(v, (int, float, np.floating))}
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)
    logger.info("Metrics saved to %s/metrics.json", OUT_DIR)


if __name__ == "__main__":
    main()
