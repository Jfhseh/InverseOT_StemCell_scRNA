import json
import logging
import sys
import time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.encoder import MLPHyperbolicEncoder
from src.train.config import Phase1Config, Phase2Config
from src.train.trainer import Trainer, Phase2Trainer
from scripts.train_phase2_lodo import load_split_dataset, evaluate_held_out

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

DATA_PATH = Path("data/processed/colon_spatial_mroi.h5ad")
OUT_DIR   = Path("outputs/phase3_iot_experiments")
FOLDS     = ["4w"] # Just 4w for rapid testing first

def _make_model(dataset):
    return MLPHyperbolicEncoder(
        input_dim=dataset.input_dim, hidden_dims=[256, 128], latent_dim=32,
        dropout=0.1, curvature=1.0, max_norm=0.95,
    )

def _get_p1_ckpt(fold: str) -> str:
    ckpt_path = Path("outputs/phase2_lodo") / f"fold_{fold}" / f"p1_{fold}" / "checkpoint_final.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing P1 checkpoint: {ckpt_path}")
    return str(ckpt_path)

def _train_p2_variant(dataset, run_name: str, fold_dir: str, warm_start_ckpt: str, **kwargs):
    torch.manual_seed(42); np.random.seed(42)
    model = _make_model(dataset)
    ckpt  = torch.load(warm_start_ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    lr = 1e-3 * 0.3
    
    config_dict = dict(
        batch_size=256, n_epochs=150, lr=lr, weight_decay=1e-4, grad_clip=1.0,
        lambda_transcript=1.0, lambda_spatial=0.0, lambda_crypt=0.0,
        lambda_iot=1.0, coupling_mode="zone_proto", epsilon_ot=0.1, n_sink_iter=20,
        sigma_depth=0.15, sigma_zone=0.8,
        lambda_unif=0.0, lambda_var=0.0, lambda_cov=0.0, gamma_var=1.0,
        lambda_rad_var=0.0, gamma_rad_var=0.1, prototype_grouping="zone",
        lambda_cross_donor=0.0, margin_cross_donor=0.5,
        label_type="continuous", k_pos=10, triplet_margin=0.5, n_neg=10,
        eval_every=150, save_every=0, device="mps",
        output_dir=fold_dir, run_name=run_name, seed=42,
    )
    config_dict.update(kwargs)
    config = Phase2Config(**config_dict)
    
    t0 = time.time()
    Phase2Trainer(config, model, dataset).train()
    logger.info("%s finished in %.1f s", run_name, time.time() - t0)
    return model

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    results_p1 = []
    results_crypt = []
    results_radvar = []
    results_age = []

    for fold in FOLDS:
        logger.info("\n%s\nFOLD: %s\n%s", "="*60, fold, "="*60)
        fold_dir = str(OUT_DIR / f"fold_{fold}")
        train_ds, test_ds, _ = load_split_dataset(fold)

        p1_ckpt = _get_p1_ckpt(fold)
        
        # We need a model instance loaded with p1_ckpt to evaluate P1
        m_p1 = _make_model(train_ds)
        m_p1.load_state_dict(torch.load(p1_ckpt, map_location="cpu", weights_only=True)["model_state_dict"])
        res1 = evaluate_held_out(m_p1, train_ds, test_ds, fold)
        res1["hold_out"] = fold
        results_p1.append(res1)

        # Variant 1: Stronger lambda_crypt (2.0)
        logger.info("\n--- Phase 2: zone_proto + lambda_crypt=2.0 ---")
        m_crypt = _train_p2_variant(train_ds, f"p2_crypt_{fold}", fold_dir, p1_ckpt, lambda_crypt=2.0)
        res_crypt = evaluate_held_out(m_crypt, train_ds, test_ds, fold)
        res_crypt["hold_out"] = fold
        results_crypt.append(res_crypt)

        # Variant 2: Explicit radial variance (1.0)
        logger.info("\n--- Phase 2: zone_proto + lambda_rad_var=1.0 ---")
        m_radvar = _train_p2_variant(train_ds, f"p2_radvar_{fold}", fold_dir, p1_ckpt, lambda_rad_var=1.0, gamma_rad_var=0.15)
        res_radvar = evaluate_held_out(m_radvar, train_ds, test_ds, fold)
        res_radvar["hold_out"] = fold
        results_radvar.append(res_radvar)

        # Variant 3: Finer grouping (zone_age)
        logger.info("\n--- Phase 2: zone_age_proto ---")
        m_age = _train_p2_variant(train_ds, f"p2_age_{fold}", fold_dir, p1_ckpt, prototype_grouping="zone_age")
        res_age = evaluate_held_out(m_age, train_ds, test_ds, fold)
        res_age["hold_out"] = fold
        results_age.append(res_age)

    def agg(results, key):
        vals = [r[key] for r in results if not (isinstance(r[key], float) and np.isnan(r[key]))]
        return float(np.mean(vals)), float(np.std(vals))

    print("\n" + "=" * 80)
    print("PHASE 3 IOT FIXES EXPERIMENTS — SCP2595 LODO")
    print("=" * 80)
    print(f"{'Model':<35} {'ρ mean±std':>13} {'NP@15':>7} {'Sil':>7} {'DMS':>7} {'r_std':>7}")
    print("-" * 80)
    
    for name, res_list in [
        ("Phase 1 baseline", results_p1),
        ("P2: zone_proto + crypt=2.0", results_crypt),
        ("P2: zone_proto + radvar=1.0", results_radvar),
        ("P2: zone_age_proto", results_age)
    ]:
        rho_m, rho_s = agg(res_list, "spearman_rho")
        np_m, _      = agg(res_list, "np15")
        sil_m, _     = agg(res_list, "silhouette")
        dms_m, _     = agg(res_list, "cross_dms")
        rstd_m, _    = agg(res_list, "radius_std")
        print(f"{name:<35} {rho_m:>6.3f}±{rho_s:.3f}  {np_m:>7.3f}  {sil_m:>7.3f}  {dms_m:>7.3f}  {rstd_m:>7.3f}")

if __name__ == "__main__":
    main()
