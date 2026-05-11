"""
Phase 2 follow-up: warm-started IOT experiments.

Runs after train_phase2.py has completed and saved the Phase 1 baseline
checkpoint at outputs/phase2_experiments/p1_baseline/checkpoint_final.pt.

Why warm-start?
    The IOT loss shapes RELATIVE geometry (which cells are near which) but
    has no signal about ABSOLUTE radius.  From random init, the transcript
    triplet loss drives all points to the boundary (same collapse as Phase 1
    unsupervised).  Warm-starting from Phase 1 preserves the radial gradient
    (ρ=0.911) while IOT refines the coupling geometry on top of it.

Experiments (all warm-started from Phase 1 checkpoint):
    1. Gaussian IOT, no MSE         -- tests if IOT alone can maintain ρ
    2. Gaussian IOT + MSE           -- hybrid with warm-start (vs cold-start hybrid)
    3. Zone-block IOT, no MSE       -- prototype cross-donor, no MSE anchor
    4. Zone-block IOT + MSE         -- prototype cross-donor with MSE anchor

Usage:
    cd /Users/gavinye/classes/cs2212_project
    source .venv/bin/activate
    python scripts/train_phase2_warmstart.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# reuse all helpers from train_phase2
from scripts.train_phase2 import (
    load_dataset,
    train_phase1_baseline,
    train_phase2_iot,
    evaluate_model,
    make_plots,
    print_summary,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_PATH   = Path("data/processed/colon_spatial_mroi.h5ad")
OUT_DIR     = Path("outputs/phase2_warmstart")
P1_CKPT     = Path("outputs/phase2_experiments/p1_baseline/checkpoint_final.pt")


def _ensure_p1_checkpoint(dataset):
    """Train Phase 1 if its checkpoint is missing."""
    if P1_CKPT.exists():
        logger.info("Phase 1 checkpoint found at %s", P1_CKPT)
        return
    logger.info("Phase 1 checkpoint not found — training Phase 1 baseline first.")
    train_phase1_baseline(dataset, n_epochs=150)
    if not P1_CKPT.exists():
        raise RuntimeError(
            f"Phase 1 checkpoint not saved to expected path {P1_CKPT}. "
            "Check output_dir / run_name in train_phase1_baseline()."
        )


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run scripts/prepare_scp2595.py first."
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset, adata = load_dataset()
    _ensure_p1_checkpoint(dataset)

    p1_ckpt = str(P1_CKPT)
    all_results: dict = {}

    # ------------------------------------------------------------------
    # 1. Gaussian IOT, warm-start, no MSE
    #    Tests whether IOT alone can maintain the Phase 1 radial gradient
    #    when the model starts from a good initialisation.
    # ------------------------------------------------------------------
    logger.info("\n=== Warm-start + Gaussian IOT (λ_crypt=0) ===")
    m, _ = train_phase2_iot(
        dataset,
        run_name         = "ws_gaussian_iot",
        lambda_iot       = 1.0,
        lambda_crypt     = 0.0,
        coupling_mode    = "gaussian",
        epsilon_ot       = 0.1,
        sigma_depth      = 0.15,
        lambda_unif      = 0.1,
        lambda_var       = 0.25,
        lambda_cov       = 0.1,
        n_epochs         = 150,
        warm_start_ckpt  = p1_ckpt,
        warm_start_lr_scale = 0.3,
    )
    all_results["Warm-start: Gaussian IOT (no MSE)"] = evaluate_model(m, dataset)

    # ------------------------------------------------------------------
    # 2. Gaussian IOT + MSE, warm-start
    #    Adds a light MSE anchor (λ_crypt=0.3) on top of the warm-start.
    # ------------------------------------------------------------------
    logger.info("\n=== Warm-start + Gaussian IOT + MSE (λ_crypt=0.3) ===")
    m, _ = train_phase2_iot(
        dataset,
        run_name         = "ws_gaussian_iot_mse",
        lambda_iot       = 1.0,
        lambda_crypt     = 0.3,
        coupling_mode    = "gaussian",
        epsilon_ot       = 0.1,
        sigma_depth      = 0.15,
        lambda_unif      = 0.1,
        lambda_var       = 0.25,
        lambda_cov       = 0.1,
        n_epochs         = 150,
        warm_start_ckpt  = p1_ckpt,
        warm_start_lr_scale = 0.3,
    )
    all_results["Warm-start: Gaussian IOT + MSE"] = evaluate_model(m, dataset)

    # ------------------------------------------------------------------
    # 3. Zone-block IOT, warm-start, no MSE
    #    Prototype cross-donor coupling; no MSE.  Main novel experiment.
    # ------------------------------------------------------------------
    logger.info("\n=== Warm-start + Zone-block IOT (cross-donor, λ_crypt=0) ===")
    m, _ = train_phase2_iot(
        dataset,
        run_name         = "ws_zoneblock_iot",
        lambda_iot       = 1.0,
        lambda_crypt     = 0.0,
        coupling_mode    = "zone_block",
        epsilon_ot       = 0.1,
        sigma_zone       = 0.8,
        lambda_unif      = 0.1,
        lambda_var       = 0.25,
        lambda_cov       = 0.1,
        n_epochs         = 150,
        warm_start_ckpt  = p1_ckpt,
        warm_start_lr_scale = 0.3,
    )
    all_results["Warm-start: Zone-block IOT (cross-donor, no MSE)"] = evaluate_model(m, dataset)

    # ------------------------------------------------------------------
    # 4. Zone-block IOT + MSE, warm-start
    #    Prototype cross-donor coupling with light MSE anchor.
    # ------------------------------------------------------------------
    logger.info("\n=== Warm-start + Zone-block IOT + MSE (λ_crypt=0.3) ===")
    m, _ = train_phase2_iot(
        dataset,
        run_name         = "ws_zoneblock_iot_mse",
        lambda_iot       = 1.0,
        lambda_crypt     = 0.3,
        coupling_mode    = "zone_block",
        epsilon_ot       = 0.1,
        sigma_zone       = 0.8,
        lambda_unif      = 0.1,
        lambda_var       = 0.25,
        lambda_cov       = 0.1,
        n_epochs         = 150,
        warm_start_ckpt  = p1_ckpt,
        warm_start_lr_scale = 0.3,
    )
    all_results["Warm-start: Zone-block IOT + MSE"] = evaluate_model(m, dataset)

    # ------------------------------------------------------------------
    # 5. Zone-prototype IOT, warm-start, no MSE
    #    The principled cross-distribution alignment mode: K×K (4×4) OT
    #    between zone prototype means, not individual cells.
    #    Cross-donor alignment is implicit: all donors' zone-k cells
    #    contribute to z_proto_k, forcing them to co-locate in hyperbolic space.
    # ------------------------------------------------------------------
    logger.info("\n=== Warm-start + Zone-prototype IOT (K×K, λ_crypt=0) ===")
    m, _ = train_phase2_iot(
        dataset,
        run_name         = "ws_zoneprot_iot",
        lambda_iot       = 1.0,
        lambda_crypt     = 0.0,
        coupling_mode    = "zone_proto",
        epsilon_ot       = 0.1,
        sigma_zone       = 0.8,
        lambda_unif      = 0.1,
        lambda_var       = 0.25,
        lambda_cov       = 0.1,
        n_epochs         = 150,
        warm_start_ckpt  = p1_ckpt,
        warm_start_lr_scale = 0.3,
    )
    all_results["Warm-start: Zone-proto IOT (K×K, no MSE)"] = evaluate_model(m, dataset)

    # ------------------------------------------------------------------
    # 6. Zone-prototype IOT + MSE, warm-start
    #    K×K distribution alignment with a light MSE depth anchor.
    # ------------------------------------------------------------------
    logger.info("\n=== Warm-start + Zone-prototype IOT + MSE (K×K, λ_crypt=0.3) ===")
    m, _ = train_phase2_iot(
        dataset,
        run_name         = "ws_zoneprot_iot_mse",
        lambda_iot       = 1.0,
        lambda_crypt     = 0.3,
        coupling_mode    = "zone_proto",
        epsilon_ot       = 0.1,
        sigma_zone       = 0.8,
        lambda_unif      = 0.1,
        lambda_var       = 0.25,
        lambda_cov       = 0.1,
        n_epochs         = 150,
        warm_start_ckpt  = p1_ckpt,
        warm_start_lr_scale = 0.3,
    )
    all_results["Warm-start: Zone-proto IOT + MSE (K×K)"] = evaluate_model(m, dataset)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print_summary(all_results)
    make_plots(all_results, OUT_DIR)

    metrics_out = {}
    for name, res in all_results.items():
        metrics_out[name] = {k: float(v) for k, v in res.items()
                             if isinstance(v, (int, float, np.floating))}
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)
    logger.info("Metrics saved to %s/metrics.json", OUT_DIR)


if __name__ == "__main__":
    main()
