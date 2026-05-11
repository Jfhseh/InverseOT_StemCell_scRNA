"""
Visualisation utilities for Phase 1 embeddings.

Two main plots:
  1. Poincaré disk: 2-D scatter of the first two latent dimensions, coloured
     by crypt-axis label (or hyperbolic radius).
  2. Radius vs. depth: scatter / violin of ||z||_2 vs. crypt position.

Both functions return the matplotlib Figure so the caller can save or display.
All plots are written to disk only if ``save_path`` is provided.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def plot_poincare_disk(
    z: np.ndarray,
    labels: Optional[np.ndarray] = None,
    label_name: str = "crypt depth",
    title: str = "Poincaré Disk (dims 0-1)",
    save_path: Optional[str | Path] = None,
):
    """
    2-D scatter of the first two latent dimensions on the Poincaré disk.

    Parameters
    ----------
    z          : (N, D) hyperbolic embeddings (uses first 2 dims).
    labels     : (N,) continuous or integer labels for colouring.
    label_name : colour-bar label.
    save_path  : if given, saves the figure to this path.

    Returns
    -------
    matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    fig, ax = plt.subplots(figsize=(6, 6))

    # Draw the unit circle boundary
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), "k-", lw=0.8, alpha=0.4)

    x, y = z[:, 0], z[:, 1]
    scatter_kwargs = dict(s=6, alpha=0.6, linewidths=0)

    if labels is not None:
        sc = ax.scatter(x, y, c=labels, cmap="viridis", **scatter_kwargs)
        plt.colorbar(sc, ax=ax, label=label_name, shrink=0.8)
    else:
        ax.scatter(x, y, **scatter_kwargs)

    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("z₀")
    ax.set_ylabel("z₁")
    ax.axhline(0, lw=0.4, color="gray", alpha=0.3)
    ax.axvline(0, lw=0.4, color="gray", alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_radius_vs_depth(
    z: np.ndarray,
    labels: np.ndarray,
    label_name: str = "crypt depth",
    title: str = "Hyperbolic radius vs. crypt axis",
    save_path: Optional[str | Path] = None,
):
    """
    Scatter (or binned violin) of ||z||_2 vs. crypt-axis label.

    Parameters
    ----------
    z          : (N, D) embeddings.
    labels     : (N,) crypt-axis labels.
    save_path  : optional output path.

    Returns
    -------
    matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    radius = np.linalg.norm(z, axis=1)
    valid = ~np.isnan(labels)
    r_v, l_v = radius[valid], labels[valid]

    rho, pval = spearmanr(r_v, l_v)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(l_v, r_v, s=4, alpha=0.4, linewidths=0, color="steelblue")
    ax.set_xlabel(label_name)
    ax.set_ylabel("Hyperbolic radius ||z||")
    ax.set_title(f"{title}\nSpearman ρ = {rho:.3f}  (p = {pval:.2g})")

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def save_all_plots(
    z: np.ndarray,
    labels: Optional[np.ndarray],
    out_dir: str | Path,
    prefix: str = "",
) -> None:
    """Convenience wrapper: save both standard plots to ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    p = f"{prefix}_" if prefix else ""
    plot_poincare_disk(z, labels, save_path=out_dir / f"{p}poincare_disk.png")
    if labels is not None:
        plot_radius_vs_depth(z, labels, save_path=out_dir / f"{p}radius_vs_depth.png")
