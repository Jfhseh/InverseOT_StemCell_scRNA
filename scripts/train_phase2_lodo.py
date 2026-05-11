"""
Leave-one-age-group-out (LODO) evaluation for Phase 2.

Matches the Phase 1 LODO protocol. For each held-out age group, trains on
the remaining donors and reports all metrics on held-out cells only.

Folds: 4w, 12w, 52w, 104w  (0w excluded — only 10 test spots)

Per fold:
  1. Phase 1 baseline (MSE sup) trained on training ages
  2. Phase 2 Zone-block IOT warm-started from Phase 1 checkpoint

Usage:
    source .venv/bin/activate
    python scripts/train_phase2_lodo.py 2>&1 | tee outputs/phase2_lodo_run.log
"""

from __future__ import annotations

import json
import logging
import sys
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
OUT_DIR   = Path("outputs/phase2_lodo")
FOLDS     = ["4w", "12w", "52w", "104w"]


# ---------------------------------------------------------------------------
# Dataset splitting
# ---------------------------------------------------------------------------

def load_split_dataset(hold_out_age: str):
    """
    Load SCP2595 and split by donor (age group).

    Returns (train_dataset, test_dataset, full_adata).
    Training set excludes all cells from hold_out_age.
    Test set contains only cells from hold_out_age.
    """
    import anndata as ad
    from src.data.dataset import CryptDataset
    from src.data.preprocessing import normalize_crypt_labels

    logger.info("Loading %s ...", DATA_PATH)
    adata = ad.read_h5ad(DATA_PATH)

    age_col = adata.obs["Age"].values
    train_mask = age_col != hold_out_age
    test_mask  = age_col == hold_out_age

    logger.info(
        "LODO split: hold-out=%s  train=%d  test=%d",
        hold_out_age, train_mask.sum(), test_mask.sum(),
    )

    def make_dataset(mask):
        labels_raw  = adata[mask].obs["crypt_depth"].values.astype(np.float32)
        labels_norm = normalize_crypt_labels(labels_raw)
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

    return make_dataset(train_mask), make_dataset(test_mask), adata


# ---------------------------------------------------------------------------
# Held-out evaluation
# ---------------------------------------------------------------------------

def evaluate_held_out(model, train_dataset, test_dataset, hold_out: str) -> dict:
    """
    Evaluate model on held-out cells.
    Also computes cross-split DMS: fraction of test cell's k-NN (in embedding)
    that come from the training set (different donors by definition).
    """
    import torch
    from scipy.stats import spearmanr
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors

    model.eval()
    with torch.no_grad():
        z_train = model.encode_dataset(
            train_dataset.expression, batch_size=512, device="cpu"
        ).numpy()
        z_test  = model.encode_dataset(
            test_dataset.expression,  batch_size=512, device="cpu"
        ).numpy()

    depth_test  = test_dataset.crypt_labels.numpy()
    radius_test = np.linalg.norm(z_test, axis=1)

    rho, pval = spearmanr(radius_test, depth_test)

    # NP@15: compare held-out PCA vs held-out embedding neighbourhoods
    def knn_indices(X, k=15):
        nn = NearestNeighbors(n_neighbors=k+1, n_jobs=-1).fit(X)
        _, idx = nn.kneighbors(X)
        return idx[:, 1:]

    ref_nb = knn_indices(test_dataset.expression.numpy())
    lat_nb = knn_indices(z_test)
    np15 = np.mean([
        len(set(ref_nb[i]) & set(lat_nb[i])) / 15
        for i in range(len(z_test))
    ])

    bins   = np.array([0.0, 0.125, 0.375, 0.75, 1.01])
    binned = np.digitize(depth_test, bins) - 1
    n_uniq = len(np.unique(binned))
    if n_uniq >= 2:
        sil = silhouette_score(z_test, binned, metric="euclidean")
    else:
        sil = float("nan")
        logger.info("  Silhouette skipped — only %d unique zone(s) in test set", n_uniq)

    # Cross-split DMS: for each test cell, fraction of its 15 nearest
    # neighbours (in the FULL embedding space = train + test) that come from
    # training donors (i.e. a different age group from the test donor).
    z_all    = np.vstack([z_train, z_test])
    is_train = np.array([True] * len(z_train) + [False] * len(z_test))
    nn_all   = NearestNeighbors(n_neighbors=16, n_jobs=-1).fit(z_all)
    n_test   = len(z_test)
    _, idx_all = nn_all.kneighbors(z_test)
    idx_all = idx_all[:, 1:]  # exclude self if present — test cells are the query
    cross_dms = float(np.mean([
        is_train[idx_all[i]].mean() for i in range(n_test)
    ]))

    logger.info(
        "  [held-out %s n=%d]  ρ=%.4f  NP@15=%.4f  Sil=%.4f  r_mean=%.3f  r_std=%.3f  cross-DMS=%.4f",
        hold_out, len(z_test), rho, np15, sil, radius_test.mean(), radius_test.std(), cross_dms,
    )
    return {
        "spearman_rho": rho, "spearman_pval": pval,
        "np15": np15, "silhouette": sil,
        "radius_mean": radius_test.mean(), "radius_std": radius_test.std(),
        "cross_dms": cross_dms,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _make_model(dataset):
    from src.models.encoder import MLPHyperbolicEncoder
    return MLPHyperbolicEncoder(
        input_dim=dataset.input_dim, hidden_dims=[256, 128], latent_dim=32,
        dropout=0.1, curvature=1.0, max_norm=0.95,
    )


def _train_p1(dataset, run_name: str, fold_dir: str) -> str:
    """Train Phase 1 and return (model, checkpoint_path). Skips training if checkpoint exists."""
    import time, torch
    from src.train.config import Phase1Config
    from src.train.trainer import Trainer
    ckpt_path = str(Path(fold_dir) / run_name / "checkpoint_final.pt")
    model = _make_model(dataset)
    if Path(ckpt_path).exists():
        logger.info("Reusing existing P1 checkpoint: %s", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        return model, ckpt_path
    torch.manual_seed(42); np.random.seed(42)
    config = Phase1Config(
        batch_size=256, n_epochs=150, lr=1e-3, weight_decay=1e-4, grad_clip=1.0,
        lambda_transcript=1.0, lambda_spatial=0.0, lambda_crypt=1.0,
        label_type="continuous", k_pos=10, triplet_margin=0.5, n_neg=10,
        eval_every=150, save_every=0, device="mps",
        output_dir=fold_dir, run_name=run_name, seed=42,
    )
    t0 = time.time()
    Trainer(config, model, dataset).train()
    logger.info("%s finished in %.1f s", run_name, time.time() - t0)
    return model, ckpt_path


def _train_p2(dataset, run_name: str, fold_dir: str, warm_start_ckpt: str):
    """Train Phase 2 with cross-donor zone triplet loss warm-started from Phase 1."""
    import time, torch
    from src.train.config import Phase2Config
    from src.train.trainer import Phase2Trainer
    torch.manual_seed(42); np.random.seed(42)
    model = _make_model(dataset)
    ckpt  = torch.load(warm_start_ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    lr = 1e-3 * 0.3
    logger.info("Warm-started from %s  (lr=%.4f)", warm_start_ckpt, lr)
    config = Phase2Config(
        batch_size=256, n_epochs=150, lr=lr, weight_decay=1e-4, grad_clip=1.0,
        lambda_transcript=1.0, lambda_spatial=0.0, lambda_crypt=1.0,
        lambda_iot=0.0,
        lambda_cross_donor=0.3, margin_cross_donor=0.5,
        coupling_mode="zone_block", epsilon_ot=0.1, n_sink_iter=20,
        sigma_depth=0.15, sigma_zone=0.8,
        lambda_unif=0.0, lambda_var=0.0, lambda_cov=0.0, gamma_var=1.0,
        label_type="continuous", k_pos=10, triplet_margin=0.5, n_neg=10,
        eval_every=150, save_every=0, device="mps",
        output_dir=fold_dir, run_name=run_name, seed=42,
    )
    t0 = time.time()
    Phase2Trainer(config, model, dataset).train()
    logger.info("%s finished in %.1f s", run_name, time.time() - t0)
    return model


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    per_fold_p1: list[dict] = []
    per_fold_p2: list[dict] = []

    for fold in FOLDS:
        logger.info("\n%s", "=" * 60)
        logger.info("FOLD: hold-out age group = %s", fold)
        logger.info("%s", "=" * 60)

        fold_dir = str(OUT_DIR / f"fold_{fold}")
        train_ds, test_ds, _ = load_split_dataset(fold)

        logger.info("\n--- Phase 1 baseline (hold-out=%s) ---", fold)
        model_p1, p1_ckpt = _train_p1(train_ds, run_name=f"p1_{fold}", fold_dir=fold_dir)
        res_p1 = evaluate_held_out(model_p1, train_ds, test_ds, fold)
        res_p1["hold_out"] = fold
        per_fold_p1.append(res_p1)

        logger.info("\n--- Phase 2 cross-donor zone triplet warm-start (hold-out=%s) ---", fold)
        model_p2 = _train_p2(train_ds, run_name=f"p2_crossdonor_{fold}",
                              fold_dir=fold_dir, warm_start_ckpt=p1_ckpt)
        res_p2 = evaluate_held_out(model_p2, train_ds, test_ds, fold)
        res_p2["hold_out"] = fold
        per_fold_p2.append(res_p2)

    # Aggregate
    def agg(results, key):
        vals = [r[key] for r in results
                if not (isinstance(r[key], float) and np.isnan(r[key]))]
        return float(np.mean(vals)), float(np.std(vals))

    rho_p1_m, rho_p1_s = agg(per_fold_p1, "spearman_rho")
    rho_p2_m, rho_p2_s = agg(per_fold_p2, "spearman_rho")
    np_p1_m,  _        = agg(per_fold_p1, "np15")
    np_p2_m,  _        = agg(per_fold_p2, "np15")
    sil_p1_m, _        = agg(per_fold_p1, "silhouette")
    sil_p2_m, _        = agg(per_fold_p2, "silhouette")
    dms_p1_m, _        = agg(per_fold_p1, "cross_dms")
    dms_p2_m, _        = agg(per_fold_p2, "cross_dms")

    print("\n" + "=" * 72)
    print("PHASE 2 LODO RESULTS — SCP2595 (folds: 4w, 12w, 52w, 104w)")
    print("=" * 72)
    print(f"{'Model':<38} {'ρ mean±std':>13} {'NP@15':>7} {'Sil':>7} {'DMS':>7}")
    print("-" * 72)
    print(f"{'Phase 1 baseline (MSE sup)':<38} "
          f"{rho_p1_m:>6.3f}±{rho_p1_s:.3f}  "
          f"{np_p1_m:>7.3f}  {sil_p1_m:>7.3f}  {dms_p1_m:>7.3f}")
    print(f"{'Phase 2 ZBlock-IOT + MSE (no reg)':<38} "
          f"{rho_p2_m:>6.3f}±{rho_p2_s:.3f}  "
          f"{np_p2_m:>7.3f}  {sil_p2_m:>7.3f}  {dms_p2_m:>7.3f}  ◄")
    print("=" * 72)
    print("Per-fold P1: " + "  ".join(
        f"{r['hold_out']}={r['spearman_rho']:.3f}" for r in per_fold_p1))
    print("Per-fold P2: " + "  ".join(
        f"{r['hold_out']}={r['spearman_rho']:.3f}" for r in per_fold_p2))

    metrics = {
        "aggregated": {
            "Phase1_baseline": {"rho_mean": rho_p1_m, "rho_std": rho_p1_s,
                                "np15_mean": np_p1_m, "silhouette_mean": sil_p1_m,
                                "cross_dms_mean": dms_p1_m},
            "Phase2_ZBlockIOT_noReg": {"rho_mean": rho_p2_m, "rho_std": rho_p2_s,
                                       "np15_mean": np_p2_m, "silhouette_mean": sil_p2_m,
                                       "cross_dms_mean": dms_p2_m},
        },
        "per_fold": {
            "Phase1_baseline": [{k: float(v) if isinstance(v, (int, float, np.floating))
                                 else v for k, v in r.items()} for r in per_fold_p1],
            "Phase2_ZBlockIOT_noReg": [{k: float(v) if isinstance(v, (int, float, np.floating))
                                        else v for k, v in r.items()} for r in per_fold_p2],
        },
    }
    out_path = OUT_DIR / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
