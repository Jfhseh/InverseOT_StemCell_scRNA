"""Re-generate plots from saved embeddings (no retraining)."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path("outputs/real_data_experiments")
DATA_H5 = Path("data/processed/colon_epithelial.h5ad")


def load_depth():
    import anndata as ad
    from src.data.preprocessing import normalize_crypt_labels
    adata = ad.read_h5ad(DATA_H5)
    raw = adata.obs["crypt_depth"].values.astype(np.float32)
    return normalize_crypt_labels(raw), adata.obs["pheno_cell_types"].values


def make_panel(z_unsup, z_sup, depth, cell_types, out_dir):
    """4-panel figure: Poincaré disk + violin for each model."""
    from scipy.stats import spearmanr

    r_unsup = np.linalg.norm(z_unsup, axis=1)
    r_sup   = np.linalg.norm(z_sup,   axis=1)
    rho_unsup, _ = spearmanr(r_unsup, depth)
    rho_sup,   _ = spearmanr(r_sup,   depth)

    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    theta = np.linspace(0, 2*np.pi, 300)

    for col, (z, r, rho, label) in enumerate([
        (z_unsup, r_unsup, rho_unsup, f"Transcript-only (unsup)\nρ={rho_unsup:.3f}"),
        (z_sup,   r_sup,   rho_sup,   f"Crypt-axis supervised (ours)\nρ={rho_sup:.3f}"),
    ]):
        # --- Poincaré disk ---
        ax = axes[0, col]
        ax.plot(np.cos(theta), np.sin(theta), 'k-', lw=0.6, alpha=0.3)
        sc = ax.scatter(z[:, 0], z[:, 1], c=depth, cmap="viridis",
                        s=3, alpha=0.5, linewidths=0)
        fig.colorbar(sc, ax=ax, label="crypt depth (norm.)", shrink=0.85)
        ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal")
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("z₀"); ax.set_ylabel("z₁")

        # --- Violin by crypt zone ---
        ax = axes[1, col]
        bins = [0.0, 0.10, 0.55, 0.80, 1.01]
        bin_labels = ["Stem\n(≈0)", "TA\n(0.23-0.43)", "Mid\n(0.63-0.77)", "Apex\n(0.91-1.0)"]
        groups = [r[(depth >= bins[i]) & (depth < bins[i+1])] for i in range(len(bins)-1)]
        counts = [len(g) for g in groups]
        groups = [g for g in groups if len(g) > 1]   # skip empty
        valid_labels = [f"{bin_labels[i]}\n(n={counts[i]})"
                        for i in range(len(bins)-1) if counts[i] > 1]
        if groups:
            vp = ax.violinplot(groups, positions=range(len(groups)), showmedians=True)
            for body in vp['bodies']:
                body.set_alpha(0.6)
        ax.set_xticks(range(len(valid_labels)))
        ax.set_xticklabels(valid_labels, fontsize=8)
        ax.set_ylabel("Hyperbolic radius ||z||")
        ax.set_title(f"Radius distribution by crypt zone\n(Spearman ρ={rho:.3f})")

    plt.tight_layout()
    path = out_dir / "comparison_panel.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    depth, cell_types = load_depth()

    z_unsup = np.load(OUT_DIR / "hyperbolic_unsup/embeddings.npy")
    z_sup   = np.load(OUT_DIR / "hyperbolic_sup/embeddings.npy")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    make_panel(z_unsup, z_sup, depth, cell_types, OUT_DIR)

    from scipy.stats import spearmanr
    r_u = np.linalg.norm(z_unsup, axis=1)
    r_s = np.linalg.norm(z_sup, axis=1)
    rho_u, pu = spearmanr(r_u, depth)
    rho_s, ps = spearmanr(r_s, depth)
    print(f"\nFinal Spearman ρ:")
    print(f"  Unsupervised : {rho_u:.4f}  (p={pu:.2g})")
    print(f"  Supervised   : {rho_s:.4f}  (p={ps:.2g})")

if __name__ == "__main__":
    main()
