# InverseOT_StemCell_scRNA

Hyperbolic representation learning for colon crypt axis structure in spatial transcriptomics data.

## Phases

**Phase 1 — Spatially Anchored Hyperbolic Encoder:** Trains an MLP encoder that maps gene expression into a Poincaré ball, supervised by ordered crypt-axis position (sub-crypt → apex). A transcript neighborhood triplet loss preserves local expression structure while a crypt-axis MSE/ranking loss aligns embedding radius to anatomical depth.

**Phase 2 — Inverse Optimal Transport (IOT) Loss:** Extends Phase 1 by adding a Sinkhorn-based IOT term that shapes the hyperbolic latent geometry so crypt-zone populations soft-align according to a biologically informed target coupling. This improves both crypt-axis Spearman ρ and distribution matching score (DMS) over Phase 1.

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.10+ and a CUDA-capable GPU (CPU works but is slower).

---

## Running the Methods

### Phase 1

**Prepare SCP2595 MROI data** (run once):
```bash
python scripts/prepare_scp2595.py
```

**Train (full data):**
```bash
python scripts/train_phase1.py \
    --data_path data/processed/colon_spatial_mroi.h5ad \
    --crypt_label_key crypt_depth \
    --label_type continuous \
    --n_epochs 150 \
    --run_name phase1_run
```

**Leave-one-age-group-out evaluation:**
```bash
python scripts/run_phase1_lodo.py
```

Outputs written to `outputs/<run_name>/`: config, checkpoint, embeddings, and diagnostic plots.

---

### Phase 2

**Train single experiment:**
```bash
python scripts/train_phase2.py \
    --run_name phase2_run \
    --lambda_crypt 1.0 \
    --n_epochs 150
```

**Cross-validation:**
```bash
python scripts/train_phase2_cv.py
```

**Leave-one-out generalization:**
```bash
python scripts/train_phase2_lodo.py
```

**Phase 2B (extended IOT experiments):**
```bash
python scripts/train_phase2b_experiments.py
```

Outputs written to `outputs/phase2*/`.