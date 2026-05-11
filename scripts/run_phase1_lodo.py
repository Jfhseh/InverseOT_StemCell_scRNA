"""
Phase 1 LODO (Leave-One-Donor-Out) Evaluation.

Evaluates three Phase 1 baselines across 4 LODO folds:
  1. PCA (best of top-10 PCs, selected on train)
  2. Hyperbolic encoder — unsupervised (transcript triplet only)
  3. Hyperbolic encoder — MROI-supervised (+ crypt-axis MSE)

For each fold, the model is trained on the 10 remaining age groups and
evaluated exclusively on the held-out age group.

Hold-outs: 4w (n=1320), 12w (n=373), 52w (n=287), 104w (n=1374).
0w (n=10) excluded — too few cells for stable metrics.

Key metric: cross_dms — fraction of each held-out cell's 15-NN in the
full embedding (train + test) that come from training donors.
Low cross_dms = test cells cluster within their own donor (bad generalization).
High cross_dms = test cells integrate with same-zone training cells (good).

Usage:
    cd /Users/gavinye/classes/cs2212_project
    source .venv/bin/activate
    python scripts/run_phase1_lodo.py
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
OUT_DIR   = Path("outputs/phase1_lodo")

# 0w (n=10) excluded: too small for silhouette / Spearman
HOLD_OUTS = ["4w", "12w", "52w", "104w"]

# SCP2595 depths are already in [0, 1] — fix normalization range globally
# so train and test subsets share the same scale.
_DEPTH_MIN = 0.0
_DEPTH_MAX = 1.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_split_dataset(hold_out_age: str):
    import anndata as ad
    from src.data.dataset import CryptDataset
    from src.data.preprocessing import normalize_crypt_labels

    adata = ad.read_h5ad(DATA_PATH)
    age_col = adata.obs["Age"].values
    train_mask = age_col != hold_out_age
    test_mask  = age_col == hold_out_age

    logger.info(
        "LODO split: hold-out=%s  train=%d  test=%d",
        hold_out_age, train_mask.sum(), test_mask.sum(),
    )

    def make_dataset(mask):
        labels_raw = adata[mask].obs["crypt_depth"].values.astype(np.float32)
        # Fixed global range — prevents per-split rescaling that creates NaN
        labels_norm = normalize_crypt_labels(
            labels_raw, min_val=_DEPTH_MIN, max_val=_DEPTH_MAX
        )
        return CryptDataset(
            expression     = adata[mask].obsm["X_pca"].astype(np.float32),
            crypt_labels   = labels_norm,
            cell_ids       = np.array(adata[mask].obs_names),
            spatial_coords = adata[mask].obsm["spatial"].astype(np.float32),
            metadata       = {
                "mroi":   adata[mask].obs["mroi_norm"].values,
                "age":    adata[mask].obs["Age"].values,
                "region": adata[mask].obs["Region"].values,
            },
        )

    return make_dataset(train_mask), make_dataset(test_mask)


# ---------------------------------------------------------------------------
# PCA baseline
# ---------------------------------------------------------------------------

def pca_baseline(train_dataset, test_dataset) -> dict:
    from scipy.stats import spearmanr

    pca_train  = train_dataset.expression.numpy()
    depth_train = train_dataset.crypt_labels.numpy()
    pca_test   = test_dataset.expression.numpy()
    depth_test  = test_dataset.crypt_labels.numpy()

    # Select best PC on training set
    best_idx, best_rho = 0, 0.0
    for i in range(min(10, pca_train.shape[1])):
        if np.std(pca_train[:, i]) < 1e-10:
            continue
        rho, _ = spearmanr(pca_train[:, i], depth_train)
        if not np.isnan(rho) and abs(rho) > best_rho:
            best_rho, best_idx = abs(rho), i

    # Evaluate on test; guard against constant arrays
    if np.std(depth_test) < 1e-10 or np.std(pca_test[:, best_idx]) < 1e-10:
        test_rho, pval = float("nan"), float("nan")
    else:
        test_rho, pval = spearmanr(pca_test[:, best_idx], depth_test)

    logger.info(
        "  Best Train PC: %d (train |ρ|=%.3f) -> Test ρ=%.4f",
        best_idx + 1, best_rho, test_rho if not np.isnan(test_rho) else float("nan"),
    )
    return {"spearman_rho": float(abs(test_rho)) if not np.isnan(test_rho) else float("nan"),
            "spearman_pval": pval, "best_pc_idx": best_idx}


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_model(
    train_dataset,
    run_name: str,
    lambda_crypt: float,
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
        input_dim   = train_dataset.input_dim,
        hidden_dims = [256, 128],
        latent_dim  = 32,
        dropout     = 0.1,
        curvature   = 1.0,
        max_norm    = 0.95,
    )
    config = Phase1Config(
        batch_size        = 256,
        n_epochs          = n_epochs,
        lr                = 1e-3,
        weight_decay      = 1e-4,
        grad_clip         = 1.0,
        lambda_transcript = 1.0,
        lambda_spatial    = 0.0,
        lambda_crypt      = lambda_crypt,
        label_type        = "continuous",
        k_pos             = 10,
        triplet_margin    = 0.5,
        n_neg             = 10,
        eval_every        = n_epochs,
        save_every        = 0,
        device            = device,
        output_dir        = str(OUT_DIR),
        run_name          = run_name,
        seed              = 42,
    )
    trainer = Trainer(config, model, train_dataset)
    t0 = time.time()
    trainer.train()
    logger.info("Training finished in %.1f s", time.time() - t0)
    return model


# ---------------------------------------------------------------------------
# Evaluation on held-out test set
# ---------------------------------------------------------------------------

def evaluate_model(model, train_dataset, test_dataset, hold_out: str) -> dict:
    """
    Evaluate model on held-out cells.

    Metrics computed purely on test cells (no train cells contaminate):
      ρ, NP@15, Silhouette

    cross_dms: fraction of each test cell's 15-NN in the FULL embedding
      (train + test concatenated) that come from a training donor.
      High = test cells integrate with training same-zone cells (good).
      Low  = test cells cluster with their own (held-out) donor (bad).
    """
    import torch
    from scipy.stats import spearmanr
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors

    model.eval()
    with torch.no_grad():
        z_test  = model.encode_dataset(
            test_dataset.expression, batch_size=512, device="cpu"
        ).numpy()
        z_train = model.encode_dataset(
            train_dataset.expression, batch_size=512, device="cpu"
        ).numpy()

    depth  = test_dataset.crypt_labels.numpy()
    radius = np.linalg.norm(z_test, axis=1)

    # Spearman ρ (guard against constant arrays)
    if np.std(radius) < 1e-10 or np.std(depth) < 1e-10:
        rho, pval = float("nan"), float("nan")
    else:
        rho, pval = spearmanr(radius, depth)

    # NP@15: compare within-test PCA and embedding neighbourhoods
    k = min(15, len(z_test) - 1)
    def knn(X):
        nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(X)
        _, idx = nn.kneighbors(X)
        return idx[:, 1:]

    ref_nb = knn(test_dataset.expression.numpy())
    lat_nb = knn(z_test)
    np15 = float(np.mean([
        len(set(ref_nb[i]) & set(lat_nb[i])) / k
        for i in range(len(z_test))
    ]))

    # Silhouette (guard against single-label test set)
    bins   = np.array([0.0, 0.125, 0.375, 0.75, 1.01])
    zbin   = np.digitize(depth, bins) - 1
    n_uniq = len(np.unique(zbin))
    if n_uniq < 2:
        sil = float("nan")
        logger.info("  Silhouette skipped — only %d unique zone(s) in test set", n_uniq)
    else:
        sil = float(silhouette_score(z_test, zbin, metric="euclidean"))

    # cross_dms: fraction of test cell's k-NN (in full space) from training donors
    z_all    = np.vstack([z_train, z_test])
    is_train = np.array([True] * len(z_train) + [False] * len(z_test))
    nn_full  = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(z_all)
    _, idx_full = nn_full.kneighbors(z_test)   # query = test cells only
    # exclude self: test cells are NOT in z_all, so no self-index issue
    cross_dms = float(np.mean([
        is_train[idx_full[i, :k]].mean()
        for i in range(len(z_test))
    ]))

    logger.info(
        "  [%s held-out n=%d]  ρ=%.4f  NP@15=%.4f  Sil=%.4f  "
        "r_mean=%.3f  r_std=%.3f  cross_dms=%.4f",
        hold_out, len(z_test), rho, np15, sil,
        radius.mean(), radius.std(), cross_dms,
    )
    return {
        "spearman_rho": float(rho),
        "spearman_pval": float(pval) if not np.isnan(pval) else float("nan"),
        "np15": np15,
        "silhouette": sil,
        "radius_mean": float(radius.mean()),
        "radius_std":  float(radius.std()),
        "cross_dms":   cross_dms,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results_per_fold: dict[str, list] = {
        "PCA": [], "Hyperbolic_Unsup": [], "Hyperbolic_MROI_Sup": []
    }

    for hold_out in HOLD_OUTS:
        logger.info("\n" + "=" * 55)
        logger.info("FOLD: hold-out age group = %s", hold_out)
        logger.info("=" * 55)

        train_dataset, test_dataset = load_split_dataset(hold_out)

        # 1. PCA baseline
        logger.info("\n--- PCA baseline (%s held-out) ---", hold_out)
        pca_res = pca_baseline(train_dataset, test_dataset)
        results_per_fold["PCA"].append({**pca_res, "hold_out": hold_out})

        # 2. Hyperbolic unsupervised
        logger.info("\n--- Hyperbolic Unsupervised (%s held-out) ---", hold_out)
        model_unsup = train_model(
            train_dataset,
            run_name     = f"unsup_lodo_{hold_out}",
            lambda_crypt = 0.0,
            n_epochs     = 150,
        )
        res_unsup = evaluate_model(model_unsup, train_dataset, test_dataset, hold_out)
        results_per_fold["Hyperbolic_Unsup"].append({**res_unsup, "hold_out": hold_out})

        # 3. Hyperbolic MROI-supervised
        logger.info("\n--- Hyperbolic MROI-Supervised (%s held-out) ---", hold_out)
        model_sup = train_model(
            train_dataset,
            run_name     = f"sup_lodo_{hold_out}",
            lambda_crypt = 1.0,
            n_epochs     = 150,
        )
        res_sup = evaluate_model(model_sup, train_dataset, test_dataset, hold_out)
        results_per_fold["Hyperbolic_MROI_Sup"].append({**res_sup, "hold_out": hold_out})

    # Summary table
    print("\n" + "=" * 76)
    print(f"PHASE 1 LODO RESULTS  ({', '.join(HOLD_OUTS)} held-out)")
    print("=" * 76)
    print(f"{'Model':<30} {'ρ (mean±std)':>16} {'NP@15':>7} {'Sil':>7} {'cross_dms':>10}")
    print("-" * 76)

    aggregated = {}
    for model_name, folds in results_per_fold.items():
        rhos  = [abs(f["spearman_rho"]) for f in folds if not np.isnan(f["spearman_rho"])]
        np15s = [f.get("np15", float("nan")) for f in folds]
        sils  = [f.get("silhouette", float("nan")) for f in folds]
        cdms  = [f.get("cross_dms", float("nan")) for f in folds]

        rho_mean = float(np.nanmean(rhos))
        rho_std  = float(np.nanstd(rhos))
        np_mean  = float(np.nanmean(np15s))
        sil_mean = float(np.nanmean(sils))
        dms_mean = float(np.nanmean(cdms))

        aggregated[model_name] = {
            "spearman_rho_mean": rho_mean, "spearman_rho_std": rho_std,
            "np15_mean": np_mean, "silhouette_mean": sil_mean,
            "cross_dms_mean": dms_mean,
        }
        print(
            f"{model_name:<30} {rho_mean:>7.4f} ± {rho_std:.4f}"
            f" {np_mean:>7.4f} {sil_mean:>7.4f} {dms_mean:>10.4f}"
        )

    print("=" * 76)

    # Per-fold detail
    print("\nPer-fold breakdown:")
    for model_name, folds in results_per_fold.items():
        print(f"\n  {model_name}:")
        for f in folds:
            print(
                f"    {f['hold_out']:>5}  ρ={f['spearman_rho']:>7.4f}"
                f"  NP@15={f.get('np15', float('nan')):>6.4f}"
                f"  Sil={f.get('silhouette', float('nan')):>7.4f}"
                f"  cross_dms={f.get('cross_dms', float('nan')):>6.4f}"
            )

    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump({"aggregated": aggregated, "per_fold": results_per_fold},
                  f, indent=2, default=str)
    logger.info("Saved to %s/metrics.json", OUT_DIR)


if __name__ == "__main__":
    main()
