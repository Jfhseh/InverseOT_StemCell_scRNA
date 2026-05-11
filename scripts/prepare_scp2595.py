"""
Prepare training data from SCP2595 processed AnnData.

Prerequisites
-------------
1. Create a free account at https://singlecell.broadinstitute.org/
2. Navigate to SCP2595 (Daly et al. 2025 colon aging atlas)
3. Download the processed cSplotch output file — expected filename:
       adata_csplotch_lambdas.h5ad
   Place it at:  data/scp2595/adata_csplotch_lambdas.h5ad

Why this file is better than the GSE285985 snRNA-seq data
---------------------------------------------------------
The snRNA-seq GEO deposit (GSE285985) carries only transcriptomic cell-type
labels (pheno_cell_types).  The SCP2595 cSplotch output has:

  * True MROI labels per spatial spot (obs['annotation']):
      'crypt apex', 'crypt mid', 'crypt base', 'sub-crypt', ...
  * 2-D spatial coordinates (obsm['spatial'])
  * Per-spot gene expression (posterior mean from cSplotch model)

These enable direct supervision on measured spatial position — not a
cell-type proxy — and also activate the spatial smoothness loss.

Usage
-----
    python scripts/prepare_scp2595.py
    python scripts/prepare_scp2595.py --input data/scp2595/my_file.h5ad

Then train with true MROI supervision:
    python scripts/train_phase1.py \
        --data_path data/processed/colon_spatial_mroi.h5ad \
        --use_rep X_pca \
        --crypt_label_key crypt_depth \
        --label_type continuous \
        --spatial_key spatial \
        --lambda_spatial 0.5 \
        --lambda_crypt 1.0 \
        --run_name mroi_supervised
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MROI → continuous crypt-axis depth
# ---------------------------------------------------------------------------

# Direct spatial measurements; order confirmed by Daly et al. Figure 1 and
# Extended Data.  'sub-crypt' is the stem-cell niche below the base.
MROI_TO_DEPTH: dict[str, float] = {
    "sub-crypt":   0.00,
    "crypt base":  0.25,
    "crypt mid":   0.50,
    "crypt apex":  1.00,
}

# MROI labels that do NOT correspond to a crypt-axis position.
# Spots carrying these labels are excluded from supervised training but
# can optionally be kept as unlabelled cells.
NON_CRYPT_MROI = {
    "epithelium",
    "epithelium and muscle and submucosa",
    "epithelium and mucosae",
    "epithelium and mucosae and submucosa",
    "muscle and submucosa",
    "interna",
    "externa",
    "externa and interna",
    "interna and mucosae",
    "peyer's patch",
    "cross-mucosa",
}

# Aliases — cSplotch annotation strings vary slightly across releases
MROI_ALIASES: dict[str, str] = {
    "sub_crypt":   "sub-crypt",
    "subcrypt":    "sub-crypt",
    "crypt_base":  "crypt base",
    "crypt_mid":   "crypt mid",
    "crypt_apex":  "crypt apex",
    # Short-form labels used in some cSplotch versions
    "SUB-CRYPT":   "sub-crypt",
    "BASE":        "crypt base",
    "MID":         "crypt mid",
    "APEX":        "crypt apex",
}


def normalise_mroi(label: str) -> str:
    """Normalise MROI label string to canonical form."""
    return MROI_ALIASES.get(label, label)


def explore_h5ad(path: Path) -> None:
    """Print a summary of an AnnData object to help identify the MROI column."""
    import anndata as ad
    adata = ad.read_h5ad(path)
    logger.info("Shape: %d obs × %d vars", adata.n_obs, adata.n_vars)
    logger.info("obsm keys: %s", list(adata.obsm.keys()))
    logger.info("uns keys:  %s", list(adata.uns.keys()))
    logger.info("")
    logger.info("obs columns (first 40):")
    for col in adata.obs.columns[:40]:
        n_unique = adata.obs[col].nunique()
        sample = adata.obs[col].dropna().unique()[:5].tolist()
        logger.info("  %-35s  nuniq=%-5d  sample=%s", col, n_unique, sample)


def find_mroi_column(adata) -> str | None:
    """Auto-detect the MROI annotation column."""
    crypt_keywords = {"crypt", "apex", "base", "mid", "sub-crypt", "subcrypt", "mroi", "annotation"}
    for col in adata.obs.columns:
        col_lower = col.lower()
        # Check if the column name looks like an annotation column
        if any(kw in col_lower for kw in crypt_keywords):
            vals = set(adata.obs[col].dropna().astype(str).unique())
            # Check if any of the values match known MROI labels
            if vals & (set(MROI_TO_DEPTH.keys()) | set(MROI_ALIASES.keys())):
                return col
    return None


def prepare(
    input_path: Path,
    output_path: Path,
    annotation_col: str | None = None,
    spatial_key: str = "spatial",
    n_pca: int = 50,
    keep_non_crypt: bool = False,
) -> None:
    import anndata as ad
    import pandas as pd
    import scanpy as sc
    from src.data.preprocessing import normalize_crypt_labels

    logger.info("Loading %s ...", input_path)
    adata = ad.read_h5ad(input_path)
    logger.info("Loaded: %d spots × %d genes/features", adata.n_obs, adata.n_vars)

    # --- Explore if no annotation column provided ---
    if annotation_col is None:
        annotation_col = find_mroi_column(adata)
        if annotation_col is None:
            logger.warning(
                "Could not auto-detect MROI column.  Run with --explore to "
                "see all obs columns, then re-run with --annotation_col <name>."
            )
            return
        logger.info("Auto-detected annotation column: '%s'", annotation_col)

    if annotation_col not in adata.obs.columns:
        raise ValueError(
            f"Column '{annotation_col}' not found.  "
            f"Available: {list(adata.obs.columns)}"
        )

    # --- Normalise and map MROI labels ---
    raw_labels = adata.obs[annotation_col].astype(str).values
    normalised = np.array([normalise_mroi(l) for l in raw_labels])
    depths = np.array([MROI_TO_DEPTH.get(l, np.nan) for l in normalised], dtype=np.float32)

    n_crypt = (~np.isnan(depths)).sum()
    logger.info("Spots with crypt-axis MROI label: %d / %d", n_crypt, len(depths))

    for mroi, depth in sorted(MROI_TO_DEPTH.items(), key=lambda x: x[1]):
        n = (normalised == mroi).sum()
        logger.info("  depth=%.2f  %-15s  n=%d", depth, mroi, n)

    # Attach depth to obs
    adata.obs["crypt_depth"] = depths
    adata.obs["mroi_norm"] = normalised

    # --- Optionally keep non-crypt spots (will have NaN crypt_depth) ---
    if not keep_non_crypt:
        crypt_mask = ~np.isnan(depths)
        logger.info(
            "Keeping %d crypt spots, dropping %d non-crypt spots",
            crypt_mask.sum(), (~crypt_mask).sum(),
        )
        adata = adata[crypt_mask].copy()
    else:
        logger.info("Keeping all %d spots (non-crypt have NaN crypt_depth)", len(adata))

    # --- Spatial coordinates ---
    if spatial_key in adata.obsm:
        logger.info("Spatial coordinates found in obsm['%s']", spatial_key)
    elif "x" in adata.obs.columns and "y" in adata.obs.columns:
        coords = np.stack([adata.obs["x"].values, adata.obs["y"].values], axis=1).astype(np.float32)
        adata.obsm[spatial_key] = coords
        logger.info("Built obsm['%s'] from obs['x'] / obs['y']: shape %s", spatial_key, coords.shape)
    else:
        logger.warning(
            "Spatial key '%s' not found in obsm and no obs['x']/obs['y'] columns; "
            "spatial loss will be disabled. Available obsm keys: %s",
            spatial_key, list(adata.obsm.keys())
        )

    # --- PCA if not already present ---
    if "X_pca" not in adata.obsm:
        logger.info("Running PCA (n=%d components) ...", n_pca)
        if hasattr(adata.X, "toarray"):
            # Sparse — scanpy handles this
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="cell_ranger")
            adata_hvg = adata[:, adata.var.highly_variable].copy()
            sc.tl.pca(adata_hvg, n_comps=n_pca)
            adata.obsm["X_pca"] = adata_hvg.obsm["X_pca"]
        else:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            X = np.array(adata.X)
            X = StandardScaler().fit_transform(X)
            pca = PCA(n_components=min(n_pca, min(X.shape) - 1))
            adata.obsm["X_pca"] = pca.fit_transform(X).astype(np.float32)
        logger.info("PCA done: shape %s", adata.obsm["X_pca"].shape)
    else:
        logger.info("Using existing X_pca: shape %s", adata.obsm["X_pca"].shape)

    # --- Save ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_path)

    depths_valid = adata.obs["crypt_depth"].dropna()
    logger.info(
        "Saved %d spots to %s\n"
        "  crypt_depth: min=%.2f  max=%.2f  mean=%.2f\n"
        "  spatial available: %s",
        len(adata), output_path,
        depths_valid.min(), depths_valid.max(), depths_valid.mean(),
        spatial_key in adata.obsm,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SCP2595 data for Phase 1 training")
    parser.add_argument(
        "--input", type=Path,
        default=Path("data/scp2595/adata_csplotch_lambdas.h5ad"),
        help="Path to downloaded SCP2595 h5ad file",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/processed/colon_spatial_mroi.h5ad"),
        help="Output path for processed AnnData",
    )
    parser.add_argument(
        "--annotation_col", type=str, default=None,
        help="obs column holding MROI label strings (auto-detected if not given)",
    )
    parser.add_argument(
        "--spatial_key", type=str, default="spatial",
        help="obsm key for 2-D spatial coordinates",
    )
    parser.add_argument(
        "--n_pca", type=int, default=50,
        help="PCA components (only used if X_pca absent)",
    )
    parser.add_argument(
        "--keep_non_crypt", action="store_true",
        help="Keep non-crypt spots with NaN crypt_depth (excluded from axis loss)",
    )
    parser.add_argument(
        "--explore", action="store_true",
        help="Print a column summary to identify the annotation column, then exit",
    )
    args = parser.parse_args()

    if not args.input.exists():
        logger.error(
            "Input file not found: %s\n"
            "Download adata_csplotch_lambdas.h5ad from SCP2595:\n"
            "  https://singlecell.broadinstitute.org/single_cell/study/SCP2595",
            args.input,
        )
        sys.exit(1)

    if args.explore:
        explore_h5ad(args.input)
        return

    prepare(
        input_path=args.input,
        output_path=args.output,
        annotation_col=args.annotation_col,
        spatial_key=args.spatial_key,
        n_pca=args.n_pca,
        keep_non_crypt=args.keep_non_crypt,
    )


if __name__ == "__main__":
    main()
