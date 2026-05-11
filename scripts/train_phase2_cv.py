"""
Standard cross-validation evaluation for Phase 2 IOT (zone_block).

All donors are seen during training, but a random 20% of cells are held out
(stratified by age group). Tests whether zone_block IOT generalises to
unseen cells from seen donors — isolating the "unseen donor" failure mode.

Usage:
    .venv/bin/python scripts/train_phase2_cv.py 2>&1 | tee outputs/phase2_cv.log
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
OUT_DIR   = Path("outputs/phase2_cv_zblock")
N_SPLITS  = 3
TEST_FRAC = 0.20
SEEDS     = [42, 7, 123]


def load_and_split(seed: int):
    """80/20 cell split stratified by Age — all donors in both splits."""
    import anndata as ad
    from src.data.dataset import CryptDataset
    from src.data.preprocessing import normalize_crypt_labels

    logger.info("Loading %s ...", DATA_PATH)
    adata = ad.read_h5ad(DATA_PATH)
    age_groups = adata.obs["Age"].values
    rng = np.random.default_rng(seed)

    train_idx, test_idx = [], []
    for age in np.unique(age_groups):
        idx = np.where(age_groups == age)[0]
        rng.shuffle(idx)
        cut = int(len(idx) * (1 - TEST_FRAC))
        train_idx.append(idx[:cut])
        test_idx.append(idx[cut:])

    train_idx = np.concatenate(train_idx)
    test_idx  = np.concatenate(test_idx)
    logger.info("CV split seed=%d  train=%d  test=%d", seed, len(train_idx), len(test_idx))

    def make_ds(idx):
        labels_raw  = adata[idx].obs["crypt_depth"].values.astype(np.float32)
        from src.data.preprocessing import normalize_crypt_labels
        labels_norm = normalize_crypt_labels(labels_raw)
        return CryptDataset(
            expression     = adata[idx].obsm["X_pca"].astype(np.float32),
            crypt_labels   = labels_norm,
            cell_ids       = np.array(adata[idx].obs_names),
            spatial_coords = adata[idx].obsm["spatial"].astype(np.float32),
            metadata       = {
                "mroi":   adata[idx].obs["mroi_norm"].values,
                "age":    adata[idx].obs["Age"].values,
                "region": adata[idx].obs["Region"].values,
            },
        )
    return make_ds(train_idx), make_ds(test_idx)


def evaluate_held_out(model, test_dataset, label: str) -> dict:
    import torch
    from scipy.stats import spearmanr
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors

    model.eval()
    with torch.no_grad():
        z_test = model.encode_dataset(
            test_dataset.expression, batch_size=512, device="cpu"
        ).numpy()

    depth_test  = test_dataset.crypt_labels.numpy()
    radius_test = np.linalg.norm(z_test, axis=1)
    rho, pval   = spearmanr(radius_test, depth_test)

    def knn_indices(X, k=15):
        nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(X)
        _, idx = nn.kneighbors(X)
        return idx[:, 1:]

    ref_nb = knn_indices(test_dataset.expression.numpy())
    lat_nb = knn_indices(z_test)
    np15 = float(np.mean([
        len(set(ref_nb[i]) & set(lat_nb[i])) / 15
        for i in range(len(z_test))
    ]))

    bins   = np.array([0.0, 0.125, 0.375, 0.75, 1.01])
    binned = np.digitize(depth_test, bins) - 1
    n_uniq = len(np.unique(binned))
    sil    = silhouette_score(z_test, binned, metric="euclidean") if n_uniq >= 2 else float("nan")

    logger.info(
        "  [%s n=%d]  ρ=%.4f  NP@15=%.4f  Sil=%.4f  r_mean=%.3f  r_std=%.3f",
        label, len(z_test), rho, np15, sil, radius_test.mean(), radius_test.std(),
    )
    return {
        "spearman_rho":  float(rho),
        "spearman_pval": float(pval),
        "np15":          float(np15),
        "silhouette":    float(sil),
        "radius_mean":   float(radius_test.mean()),
        "radius_std":    float(radius_test.std()),
    }


def _make_model(dataset):
    from src.models.encoder import MLPHyperbolicEncoder
    return MLPHyperbolicEncoder(
        input_dim=dataset.input_dim, hidden_dims=[256, 128], latent_dim=32,
        dropout=0.1, curvature=1.0, max_norm=0.95,
    )


def _train_p1(dataset, run_name: str, out_dir: str):
    import torch
    from src.train.config import Phase1Config
    from src.train.trainer import Trainer

    ckpt_path = str(Path(out_dir) / run_name / "checkpoint_final.pt")
    model = _make_model(dataset)
    if Path(ckpt_path).exists():
        logger.info("Reusing P1 checkpoint: %s", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        return model, ckpt_path

    torch.manual_seed(42); np.random.seed(42)
    config = Phase1Config(
        batch_size=256, n_epochs=150, lr=1e-3, weight_decay=1e-4, grad_clip=1.0,
        lambda_transcript=1.0, lambda_spatial=0.0, lambda_crypt=1.0,
        label_type="continuous", k_pos=10, triplet_margin=0.5, n_neg=10,
        eval_every=150, save_every=0, device="mps",
        output_dir=out_dir, run_name=run_name, seed=42,
    )
    t0 = time.time()
    Trainer(config, model, dataset).train()
    logger.info("%s done in %.1f s", run_name, time.time() - t0)
    return model, ckpt_path


def _train_p2_zblock(dataset, run_name: str, out_dir: str, warm_ckpt: str, **kwargs):
    """zone_block IOT warm-started from P1, mirroring the original LODO config."""
    import torch
    from src.train.config import Phase2Config
    from src.train.trainer import Phase2Trainer

    torch.manual_seed(42); np.random.seed(42)
    model = _make_model(dataset)
    ckpt  = torch.load(warm_ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])

    defaults = dict(
        batch_size=256, n_epochs=150, lr=3e-4, weight_decay=1e-4, grad_clip=1.0,
        lambda_transcript=1.0, lambda_spatial=0.0, lambda_crypt=0.0,
        lambda_iot=1.0, coupling_mode="zone_block", epsilon_ot=0.1, n_sink_iter=20,
        sigma_depth=0.15, sigma_zone=0.8,
        lambda_unif=0.0, lambda_var=0.0, lambda_cov=0.0, gamma_var=1.0,
        lambda_rad_var=0.0, gamma_rad_var=0.1, prototype_grouping="zone",
        lambda_cross_donor=0.0, margin_cross_donor=0.5,
        label_type="continuous", k_pos=10, triplet_margin=0.5, n_neg=10,
        eval_every=150, save_every=0, device="mps",
        output_dir=out_dir, run_name=run_name, seed=42,
    )
    defaults.update(kwargs)
    config = Phase2Config(**defaults)

    t0 = time.time()
    Phase2Trainer(config, model, dataset).train()
    logger.info("%s done in %.1f s", run_name, time.time() - t0)
    return model


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_p1, all_zblock, all_zblock_crypt = [], [], []

    for seed in SEEDS:
        logger.info("\n%s\nSEED=%d\n%s", "="*60, seed, "="*60)
        split_dir = str(OUT_DIR / f"seed_{seed}")
        train_ds, test_ds = load_and_split(seed)

        logger.info("--- Phase 1 baseline ---")
        m_p1, p1_ckpt = _train_p1(train_ds, f"p1_s{seed}", split_dir)
        r = evaluate_held_out(m_p1, test_ds, f"P1 seed={seed}")
        all_p1.append(r)

        logger.info("--- P2: zone_block IOT (no crypt MSE) ---")
        m = _train_p2_zblock(train_ds, f"p2_zblock_s{seed}", split_dir, p1_ckpt)
        r = evaluate_held_out(m, test_ds, f"P2 zblock seed={seed}")
        all_zblock.append(r)

        logger.info("--- P2: zone_block IOT + lambda_crypt=1.0 ---")
        m = _train_p2_zblock(train_ds, f"p2_zblock_crypt_s{seed}", split_dir, p1_ckpt,
                             lambda_crypt=1.0)
        r = evaluate_held_out(m, test_ds, f"P2 zblock+crypt seed={seed}")
        all_zblock_crypt.append(r)

    def agg(results, key):
        vals = [r[key] for r in results if not (isinstance(r[key], float) and np.isnan(r[key]))]
        return float(np.mean(vals)), float(np.std(vals))

    print("\n" + "=" * 80)
    print("PHASE 2 CV RESULTS — zone_block IOT, all-donor 80/20 split, 3 seeds")
    print("=" * 80)
    print(f"{'Model':<40} {'ρ mean±std':>13} {'NP@15':>7} {'Sil':>7} {'r_std':>7}")
    print("-" * 80)
    for name, res_list in [
        ("Phase 1 baseline (MSE sup)",        all_p1),
        ("P2: zone_block IOT",                all_zblock),
        ("P2: zone_block IOT + crypt=1.0",    all_zblock_crypt),
    ]:
        rho_m, rho_s = agg(res_list, "spearman_rho")
        np_m, _      = agg(res_list, "np15")
        sil_m, _     = agg(res_list, "silhouette")
        rstd_m, _    = agg(res_list, "radius_std")
        print(f"{name:<40} {rho_m:>6.3f}±{rho_s:.3f}  {np_m:>7.3f}  {sil_m:>7.3f}  {rstd_m:>7.3f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
