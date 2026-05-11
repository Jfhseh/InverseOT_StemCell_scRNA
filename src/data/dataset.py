"""
CryptDataset: loads expression, optional spatial coords, crypt-axis labels,
and donor/section metadata for colon crypt cells/spots.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import torch
from torch.utils.data import Dataset


class CryptDataset(Dataset):
    """
    Dataset for colon crypt cells or spatial spots.

    Parameters
    ----------
    expression:
        (N, D) float32 array — gene expression, PCA embedding, or any other
        pre-computed per-cell representation.
    spatial_coords:
        (N, 2) float32 array of 2-D spatial coordinates (e.g. Visium spot XY).
        Set to None when spatial data are unavailable (transcript-only fallback).
    crypt_labels:
        (N,) float32 array of crypt-axis position.  Can be ordinal integers
        (0=subcrypt, 1=base, 2=mid, 3=apex) or a continuous depth score in
        [0, 1].  None when labels are absent.
    cell_ids:
        (N,) string array of cell / spot identifiers.
    metadata:
        Dict mapping obs-key names to (N,) arrays (e.g. donor, section).
    """

    def __init__(
        self,
        expression: np.ndarray,
        spatial_coords: Optional[np.ndarray] = None,
        crypt_labels: Optional[np.ndarray] = None,
        cell_ids: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        self.expression = torch.as_tensor(expression, dtype=torch.float32)
        self.n_cells: int = len(expression)

        self.spatial_coords = (
            torch.as_tensor(spatial_coords, dtype=torch.float32)
            if spatial_coords is not None
            else None
        )
        self.crypt_labels = (
            torch.as_tensor(crypt_labels, dtype=torch.float32)
            if crypt_labels is not None
            else None
        )
        self.cell_ids = cell_ids
        self.metadata: Dict[str, np.ndarray] = metadata or {}

    # ------------------------------------------------------------------
    # Standard Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.n_cells

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "idx": idx,
            "expression": self.expression[idx],
        }
        if self.spatial_coords is not None:
            item["spatial_coords"] = self.spatial_coords[idx]
        if self.crypt_labels is not None:
            item["crypt_label"] = self.crypt_labels[idx]
        return item

    # ------------------------------------------------------------------
    # Factory constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_anndata(
        cls,
        adata: Any,  # anndata.AnnData; typed as Any to avoid hard import
        use_rep: str = "X_pca",
        spatial_key: Optional[str] = "spatial",
        crypt_label_key: Optional[str] = None,
        metadata_keys: Optional[List[str]] = None,
    ) -> "CryptDataset":
        """
        Build a CryptDataset from an AnnData object.

        Parameters
        ----------
        use_rep:
            Key in ``adata.obsm`` to use as the expression representation, or
            ``"X"`` to use the raw count / normalised matrix directly.
        spatial_key:
            Key in ``adata.obsm`` holding 2-D spatial coordinates.
        crypt_label_key:
            Column in ``adata.obs`` holding crypt-axis labels.  String columns
            are ordinally encoded (alphabetical sort).
        metadata_keys:
            Additional columns in ``adata.obs`` to store as metadata.
        """
        # --- expression ---
        if use_rep == "X":
            X = adata.X
            expression: np.ndarray = X.toarray() if hasattr(X, "toarray") else np.array(X)
        elif use_rep in adata.obsm:
            expression = np.array(adata.obsm[use_rep])
        else:
            raise ValueError(
                f"Representation '{use_rep}' not found in adata.obsm; "
                f"available keys: {list(adata.obsm.keys())}"
            )

        # --- spatial coords ---
        spatial_coords: Optional[np.ndarray] = None
        if spatial_key is not None and spatial_key in adata.obsm:
            spatial_coords = np.array(adata.obsm[spatial_key], dtype=np.float32)

        # --- crypt labels ---
        crypt_labels: Optional[np.ndarray] = None
        if crypt_label_key is not None:
            raw = adata.obs[crypt_label_key].values
            if raw.dtype.kind in ("U", "S", "O"):
                # Ordinal string labels → integer codes (sorted alphabetically)
                categories = sorted(set(raw))
                label_map = {c: float(i) for i, c in enumerate(categories)}
                crypt_labels = np.array([label_map[v] for v in raw], dtype=np.float32)
            else:
                crypt_labels = raw.astype(np.float32)

        # --- metadata ---
        metadata: Dict[str, np.ndarray] = {}
        if metadata_keys:
            for key in metadata_keys:
                if key in adata.obs:
                    metadata[key] = adata.obs[key].values

        return cls(
            expression=expression.astype(np.float32),
            spatial_coords=spatial_coords,
            crypt_labels=crypt_labels,
            cell_ids=np.array(adata.obs_names),
            metadata=metadata,
        )

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        expression_key: str = "expression",
        spatial_key: str = "spatial_coords",
        label_key: str = "crypt_labels",
    ) -> "CryptDataset":
        """Load from a NumPy .npz archive produced by ``save_npz``."""
        data = np.load(path, allow_pickle=False)
        expression = data[expression_key]
        spatial_coords = data[spatial_key] if spatial_key in data else None
        crypt_labels = data[label_key] if label_key in data else None
        return cls(expression=expression, spatial_coords=spatial_coords, crypt_labels=crypt_labels)

    def save_npz(self, path: str | Path) -> None:
        """Save expression, spatial, and label arrays to a .npz archive."""
        arrays: Dict[str, np.ndarray] = {
            "expression": self.expression.numpy(),
        }
        if self.spatial_coords is not None:
            arrays["spatial_coords"] = self.spatial_coords.numpy()
        if self.crypt_labels is not None:
            arrays["crypt_labels"] = self.crypt_labels.numpy()
        np.savez(path, **arrays)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def has_spatial(self) -> bool:
        return self.spatial_coords is not None

    @property
    def has_labels(self) -> bool:
        return self.crypt_labels is not None

    @property
    def input_dim(self) -> int:
        return self.expression.shape[1]
