"""
Load GSE285985 snRNA-seq samples, filter to epithelial cells,
assign crypt-axis depth labels from cell-type annotations, and save
a processed AnnData (.h5ad) ready for training.

Biology reference for crypt-axis depth mapping
-----------------------------------------------
Mouse colonic crypt (base=0, apex=1):
  0.00  Intestinal stem cells (ISC / Stem)
  0.25  Transit amplifying (TA) - early, highly proliferative
  0.35  Cycling TA
  0.55  Tuft cells (scattered, biased mid-crypt)
  0.65  Enteroendocrine cells (distributed but apex-biased)
  0.80  Mature colonocytes (dominant apex population)
  0.88  Goblet cells (mucus-secreting, apex-biased)

Non-epithelial cell types are excluded because they do not have a
biologically meaningful crypt-axis position.

Usage
-----
    python scripts/prepare_real_data.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Crypt-axis depth label map
# ---------------------------------------------------------------------------

CRYPT_DEPTH: dict[str, float] = {
    # --- Stem / progenitor (base) ---
    "Stem_1":          0.00,
    "Stem":            0.00,
    # --- Transit amplifying ---
    "TA_1":            0.20,
    "TA_2":            0.25,
    "TA_3":            0.25,
    "TA_4":            0.30,
    "TA_5":            0.30,
    "TA_6":            0.30,
    "Cycling_TA_1":    0.35,
    "Cycling_TA_2":    0.38,
    # --- Secretory lineage, mid-crypt ---
    "Tuft":            0.55,
    "Enteroendocrine_1": 0.65,
    "Enteroendocrine_2": 0.68,
    # --- Mature absorptive / secretory (apex) ---
    "Colonocyte":      0.80,
    "Goblet":          0.88,
}

# Any cell type NOT listed here is non-epithelial and will be excluded
EPITHELIAL_TYPES = set(CRYPT_DEPTH.keys())

SAMPLE_IDS = {
    "GSM8714511": "6w_All_M_M1",
    "GSM8714512": "8w_All_F_M1",
    "GSM8714520": "6m_All_M_M1",
}


def load_sample(acc: str, tag: str, data_dir: Path) -> "anndata.AnnData":
    """Load one GSE285985 sample.

    The features.tsv.gz in this dataset is NON-STANDARD: it has a custom
    header line and extra per-gene QC columns appended after the usual 3
    (gene_name, ensembl_id, feature_type).  We skip that header and read
    only the first two columns (gene symbol, Ensembl ID).
    """
    import anndata as ad
    import pandas as pd
    import scipy.io

    sample_dir = data_dir / acc
    logger.info("Loading %s (%s) ...", acc, tag)

    # --- Barcodes ---
    # This dataset's barcodes.tsv.gz has a spurious '0' on the first line
    # (a leftover column-index artifact).  We read all lines, then drop
    # any entry that doesn't look like a 10x barcode (must contain '-').
    barcodes_raw = pd.read_csv(
        sample_dir / "barcodes.tsv.gz", header=None, names=["barcode"]
    )["barcode"].values
    barcodes = np.array([b for b in barcodes_raw if "-" in str(b)])

    # --- Features: skip the custom header line ---
    features = pd.read_csv(
        sample_dir / "features.tsv.gz",
        sep="\t",
        header=0,       # consume (and discard) the non-standard first line
        usecols=[0, 1], # gene symbol, Ensembl ID
    )
    features.columns = ["gene_symbol", "ensembl_id"]
    # Make unique gene names
    gene_names = features["gene_symbol"].values
    seen: dict[str, int] = {}
    unique_names = []
    for g in gene_names:
        if g in seen:
            seen[g] += 1
            unique_names.append(f"{g}-{seen[g]}")
        else:
            seen[g] = 0
            unique_names.append(g)
    features["gene_unique"] = unique_names

    # --- Matrix ---
    # This dataset stores the matrix as (cells × genes) — opposite of the
    # standard 10x convention (genes × cells).  No transpose needed.
    mat = scipy.io.mmread(sample_dir / "matrix.mtx.gz").tocsr()  # cells × genes

    # --- Construct AnnData ---
    adata = ad.AnnData(
        X=mat,
        obs=pd.DataFrame(index=barcodes),
        var=features.set_index("gene_unique"),
    )

    # --- Attach metadata ---
    meta = pd.read_csv(sample_dir / "metadata.tsv.gz", sep="\t", index_col=0)
    shared = adata.obs_names.intersection(meta.index)
    adata = adata[shared].copy()
    adata.obs = meta.loc[shared].copy()

    adata.obs["sample_id"] = acc
    adata.obs["age_label"] = tag.split("_")[0]
    logger.info("  %d cells, %d genes", adata.n_obs, adata.n_vars)
    return adata


def assign_crypt_depth(adata: "anndata.AnnData") -> "anndata.AnnData":
    import pandas as pd

    ct_col = "pheno_cell_types"
    if ct_col not in adata.obs.columns:
        raise ValueError(f"Column '{ct_col}' not found in obs")

    depths = adata.obs[ct_col].map(CRYPT_DEPTH)  # NaN for non-epithelial

    adata.obs["crypt_depth"] = depths.astype("float32")

    # Epithelial mask
    is_epi = adata.obs[ct_col].isin(EPITHELIAL_TYPES)
    logger.info(
        "Epithelial cells: %d / %d (%.1f%%)",
        is_epi.sum(), len(adata), 100 * is_epi.mean(),
    )
    return adata[is_epi].copy()


def preprocess(adata: "anndata.AnnData", n_hvg: int = 2000, n_pca: int = 50) -> "anndata.AnnData":
    import scanpy as sc

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor="cell_ranger")
    adata = adata[:, adata.var.highly_variable].copy()
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=n_pca, svd_solver="arpack")
    logger.info("After QC + HVG + PCA: %d cells × %d HVGs, PCA=%d", adata.n_obs, adata.n_vars, n_pca)
    return adata


def main(data_dir: Path, out_path: Path) -> None:
    import anndata as ad

    adatas = []
    for acc, tag in SAMPLE_IDS.items():
        try:
            adata = load_sample(acc, tag, data_dir)
            adata = assign_crypt_depth(adata)
            adatas.append(adata)
        except Exception as exc:
            logger.warning("Skipping %s: %s", acc, exc)

    if not adatas:
        raise RuntimeError("No samples loaded successfully")

    combined = ad.concat(adatas, label="sample", keys=list(SAMPLE_IDS.keys()))
    logger.info("Combined: %d cells", combined.n_obs)

    combined = preprocess(combined)

    # Depth distribution
    depths = combined.obs["crypt_depth"].dropna()
    logger.info(
        "Crypt depth: min=%.2f  max=%.2f  mean=%.2f  n_valid=%d",
        depths.min(), depths.max(), depths.mean(), len(depths),
    )
    for ct, d in sorted(CRYPT_DEPTH.items(), key=lambda x: x[1]):
        n = (combined.obs["pheno_cell_types"] == ct).sum()
        if n > 0:
            logger.info("  depth=%.2f  %s  n=%d", d, ct, n)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.write_h5ad(out_path)
    logger.info("Saved to %s", out_path)


if __name__ == "__main__":
    data_dir = Path("data/raw")
    out_path = Path("data/processed/colon_epithelial.h5ad")
    main(data_dir, out_path)
