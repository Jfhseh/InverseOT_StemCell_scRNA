"""
Phase 1 training loop.

Responsibilities
----------------
* Set up optimizer and DataLoader.
* Iterate minibatches, combine all three losses with configured weights.
* Skip losses whose pre-requisites are absent (no spatial → skip spatial
  loss; no labels → skip crypt-axis loss).
* Evaluate every ``config.eval_every`` epochs and log Spearman ρ.
* Save checkpoints and the final embeddings.

Loss combination::

    L = λ_t · L_transcript
      + λ_s · L_spatial       (if spatial coords present)
      + λ_c · L_crypt         (if crypt labels present)
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..data.dataset import CryptDataset
from ..eval.metrics import evaluate_embeddings
from ..losses.cross_donor import cross_donor_zone_triplet_loss
from ..losses.crypt_axis import crypt_axis_loss
from ..losses.inverse_ot import inverse_ot_loss, multiscale_inverse_ot_loss
from ..losses.iot_targets import depth_to_zone
from ..losses.neighborhood import transcript_neighbor_loss, within_zone_triplet_loss
from ..losses.radial_geometry import (
    zone_radial_center_loss,
    adjacent_radial_margin_loss,
    within_zone_radius_variance_floor,
    boundary_penalty,
)
from ..losses.spatial import spatial_smoothness_loss
from .config import Phase1Config, Phase2Config, Phase2V2Config

logger = logging.getLogger(__name__)


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


class Trainer:
    """
    Minimal training loop for Phase 1 hyperbolic crypt encoder.

    Parameters
    ----------
    config  : Phase1Config instance.
    model   : MLPHyperbolicEncoder.
    dataset : CryptDataset.
    """

    def __init__(
        self,
        config: Phase1Config,
        model: nn.Module,
        dataset: CryptDataset,
    ) -> None:
        self.config = config
        self.model = model
        self.dataset = dataset

        self.device = _resolve_device(config.device)
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        # Output directory
        self.out_dir = Path(config.output_dir) / config.run_name
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Training history (loss per epoch)
        self.history: Dict[str, list] = {
            "epoch": [],
            "loss_total": [],
            "loss_transcript": [],
            "loss_spatial": [],
            "loss_crypt": [],
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, n_epochs: Optional[int] = None) -> None:
        """Run the training loop for ``n_epochs`` (or config.n_epochs)."""
        n_epochs = n_epochs if n_epochs is not None else self.config.n_epochs

        torch.manual_seed(self.config.seed)

        loader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,  # keep it simple; set >0 for large datasets
            pin_memory=(self.device.type == "cuda"),
        )

        logger.info(
            "Training: %d epochs · %d cells · batch %d · device %s",
            n_epochs,
            len(self.dataset),
            self.config.batch_size,
            self.device,
        )

        for epoch in range(1, n_epochs + 1):
            epoch_losses = self._train_epoch(loader)

            # Append to history
            self.history["epoch"].append(epoch)
            for k, v in epoch_losses.items():
                self.history[k].append(v)

            # Logging
            if epoch % max(1, self.config.eval_every // 5) == 0 or epoch == 1:
                parts = [f"epoch {epoch:4d}/{n_epochs}"] + [
                    f"{k.replace('loss_', '')}={v:.4f}" for k, v in epoch_losses.items()
                ]
                logger.info("  ".join(parts))

            # Evaluation
            if epoch % self.config.eval_every == 0 or epoch == n_epochs:
                self._evaluate(epoch)

            # Checkpoint
            if self.config.save_every > 0 and epoch % self.config.save_every == 0:
                self.save_checkpoint(epoch)

        # Final save
        self.save_checkpoint("final")
        self._save_embeddings()
        logger.info("Training complete. Outputs saved to %s", self.out_dir)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        cfg = self.config
        manifold = self.model._get_manifold()

        sum_total = sum_t = sum_s = sum_c = 0.0
        n_batches = 0

        for batch in loader:
            x = batch["expression"].to(self.device)
            spatial = batch.get("spatial_coords")
            labels = batch.get("crypt_label")

            if spatial is not None:
                spatial = spatial.to(self.device)
            if labels is not None:
                labels = labels.to(self.device)

            # Forward pass
            z = self.model(x)

            # --- Loss 1: transcript neighborhood ---
            l_t = transcript_neighbor_loss(
                z, x, manifold,
                k=cfg.k_pos,
                margin=cfg.triplet_margin,
                n_neg=cfg.n_neg,
            )

            # --- Loss 2: spatial smoothness ---
            l_s = torch.zeros(1, device=self.device)
            if spatial is not None and cfg.lambda_spatial > 0:
                l_s = spatial_smoothness_loss(
                    z, spatial, manifold, k=cfg.k_spatial
                )

            # --- Loss 3: crypt-axis supervision ---
            l_c = torch.zeros(1, device=self.device)
            if labels is not None and cfg.lambda_crypt > 0:
                l_c = crypt_axis_loss(
                    z, labels,
                    label_type=cfg.label_type,
                    margin=cfg.rank_margin,
                )

            loss = (
                cfg.lambda_transcript * l_t
                + cfg.lambda_spatial * l_s
                + cfg.lambda_crypt * l_c
            )

            self.optimizer.zero_grad()
            loss.backward()

            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)

            self.optimizer.step()

            sum_total += loss.detach().item()
            sum_t += l_t.detach().item()
            sum_s += l_s.detach().item()
            sum_c += l_c.detach().item()
            n_batches += 1

        denom = max(n_batches, 1)
        return {
            "loss_total": sum_total / denom,
            "loss_transcript": sum_t / denom,
            "loss_spatial": sum_s / denom,
            "loss_crypt": sum_c / denom,
        }

    def _evaluate(self, epoch: int | str) -> None:
        self.model.eval()
        with torch.no_grad():
            z_all = self.model.encode_dataset(
                self.dataset.expression,
                batch_size=self.config.batch_size,
                device=str(self.device),
            )
        labels_np = (
            self.dataset.crypt_labels.numpy()
            if self.dataset.crypt_labels is not None
            else None
        )
        metrics = evaluate_embeddings(
            z_all.numpy(),
            crypt_labels=labels_np,
            expression=self.dataset.expression.numpy(),
        )
        parts = [f"[eval epoch={epoch}]"]
        for k, v in metrics.items():
            parts.append(f"{k}={v:.4f}")
        logger.info("  ".join(parts))

    def save_checkpoint(self, tag: int | str) -> None:
        path = self.out_dir / f"checkpoint_{tag}.pt"
        torch.save(
            {
                "epoch": tag,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config.to_dict(),
            },
            path,
        )
        logger.info("Checkpoint saved: %s", path)

    def _save_embeddings(self) -> None:
        self.model.eval()
        with torch.no_grad():
            z_all = self.model.encode_dataset(
                self.dataset.expression,
                batch_size=self.config.batch_size,
                device=str(self.device),
            )
        out_path = self.out_dir / "embeddings.npy"
        np.save(out_path, z_all.numpy())
        logger.info("Embeddings saved: %s  shape=%s", out_path, z_all.shape)


class Phase2Trainer(Trainer):
    """
    Phase 2 training loop: replaces the hard crypt-axis MSE loss with a
    bilevel inverse-OT loss.

    Loss::

        L = λ_t · L_transcript
          + λ_iot · L_IOT           (KL + uniformity + var + cov)

    All Phase1 losses (spatial, crypt MSE) are still available via config
    if lambda values are set > 0.
    """

    def __init__(
        self,
        config: Phase2Config,
        model: nn.Module,
        dataset: CryptDataset,
    ) -> None:
        super().__init__(config, model, dataset)
        self.p2config: Phase2Config = config
        # Extend history to track IOT sub-components and cross-donor loss
        self.history["loss_iot"]        = []
        self.history["iot_kl"]          = []
        self.history["iot_var"]         = []
        self.history["iot_rad_var"]     = []
        self.history["loss_cross_donor"] = []

        # Precompute zone labels (0-3) from crypt labels for cross-donor triplet
        _ZONE_BOUNDS = [0.125, 0.375, 0.75]
        if dataset.crypt_labels is not None:
            lbl = dataset.crypt_labels
            zones = torch.zeros(len(lbl), dtype=torch.long)
            for k, b in enumerate(_ZONE_BOUNDS):
                zones[lbl >= b] = k + 1
            self._zone_labels = zones   # (N,)
        else:
            self._zone_labels = None

        # Precompute integer age labels from metadata
        age_raw = dataset.metadata.get("age", None)
        if age_raw is not None:
            unique_ages = sorted(set(age_raw))
            age_map = {a: i for i, a in enumerate(unique_ages)}
            self._age_labels = torch.tensor(
                [age_map[a] for a in age_raw], dtype=torch.long
            )
        else:
            self._age_labels = None

    def _train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        cfg = self.p2config
        manifold = self.model._get_manifold()

        sum_total = sum_t = sum_s = sum_c = sum_iot = sum_kl = sum_var = sum_rad_var = sum_cd = 0.0
        n_batches = 0

        for batch in loader:
            x       = batch["expression"].to(self.device)
            spatial = batch.get("spatial_coords")
            labels  = batch.get("crypt_label")
            idx     = batch.get("idx")

            if spatial is not None:
                spatial = spatial.to(self.device)
            if labels is not None:
                labels = labels.to(self.device)

            z = self.model(x)

            # --- transcript neighborhood loss ---
            l_t = transcript_neighbor_loss(
                z, x, manifold,
                k=cfg.k_pos,
                margin=cfg.triplet_margin,
                n_neg=cfg.n_neg,
            )

            # --- spatial smoothness loss (usually disabled in Phase 2) ---
            l_s = torch.zeros(1, device=self.device)
            if spatial is not None and cfg.lambda_spatial > 0:
                l_s = spatial_smoothness_loss(z, spatial, manifold, k=cfg.k_spatial)

            # --- crypt MSE loss ---
            l_c = torch.zeros(1, device=self.device)
            if labels is not None and cfg.lambda_crypt > 0:
                l_c = crypt_axis_loss(z, labels, label_type=cfg.label_type,
                                      margin=cfg.rank_margin)

            # Extract age labels for sub_labels if needed
            ages_b = None
            if idx is not None and self._age_labels is not None:
                ages_b = self._age_labels[idx].to(self.device)

            # --- bilevel inverse-OT loss ---
            l_iot = torch.zeros(1, device=self.device)
            iot_components: Dict[str, float] = {}
            if labels is not None and cfg.lambda_iot > 0:
                l_iot, iot_components = inverse_ot_loss(
                    z=z,
                    labels=labels,
                    manifold=manifold,
                    epsilon=cfg.epsilon_ot,
                    n_sink_iter=cfg.n_sink_iter,
                    coupling_mode=cfg.coupling_mode,
                    sigma_depth=cfg.sigma_depth,
                    sigma_zone=cfg.sigma_zone,
                    lambda_unif=cfg.lambda_unif,
                    lambda_var=cfg.lambda_var,
                    lambda_cov=cfg.lambda_cov,
                    gamma_var=cfg.gamma_var,
                    lambda_rad_var=cfg.lambda_rad_var,
                    gamma_rad_var=cfg.gamma_rad_var,
                    prototype_grouping=cfg.prototype_grouping,
                    sub_labels=ages_b,
                )

            # --- cross-donor zone triplet loss ---
            l_cd = torch.zeros(1, device=self.device)
            if (cfg.lambda_cross_donor > 0
                    and idx is not None
                    and self._zone_labels is not None
                    and ages_b is not None):
                zones_b = self._zone_labels[idx].to(self.device)
                l_cd = cross_donor_zone_triplet_loss(
                    z, zones_b, ages_b, manifold,
                    margin=cfg.margin_cross_donor,
                )

            loss = (
                cfg.lambda_transcript  * l_t
                + cfg.lambda_spatial   * l_s
                + cfg.lambda_crypt     * l_c
                + cfg.lambda_iot       * l_iot
                + cfg.lambda_cross_donor * l_cd
            )

            self.optimizer.zero_grad()
            loss.backward()

            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)

            self.optimizer.step()

            sum_total += loss.detach().item()
            sum_t     += l_t.detach().item()
            sum_s     += l_s.detach().item()
            sum_c     += l_c.detach().item()
            sum_iot   += l_iot.detach().item()
            sum_kl    += iot_components.get("iot_kl",  0.0)
            sum_var   += iot_components.get("iot_var", 0.0)
            sum_rad_var += iot_components.get("iot_rad_var", 0.0)
            sum_cd    += l_cd.detach().item()
            n_batches += 1

        d = max(n_batches, 1)
        return {
            "loss_total":        sum_total / d,
            "loss_transcript":   sum_t     / d,
            "loss_spatial":      sum_s     / d,
            "loss_crypt":        sum_c     / d,
            "loss_iot":          sum_iot   / d,
            "iot_kl":            sum_kl    / d,
            "iot_var":           sum_var   / d,
            "iot_rad_var":       sum_rad_var / d,
            "loss_cross_donor":  sum_cd    / d,
        }


class Phase2V2Trainer(Trainer):
    """Phase 2B training loop: multi-scale IOT + radial geometry + within-zone local.

    Loss::

        L = λ_t   · L_transcript
          + λ_iot  · L_multiscale_iot
          + λ_rad  · (L_radial_center + η · L_radial_margin)
          + λ_loc  · L_within_zone_local
          + λ_bnd  · L_boundary
          + λ_c    · L_crypt   [optional MSE anchor]
    """

    def __init__(
        self,
        config: Phase2V2Config,
        model,
        dataset: CryptDataset,
        donor_ids=None,
    ) -> None:
        super().__init__(config, model, dataset)
        self.cfg2v2: Phase2V2Config = config
        # (N,) integer donor IDs aligned with dataset indices; None = donor-agnostic
        self._donor_ids = np.array(donor_ids) if donor_ids is not None else None

        extra_keys = [
            "loss_iot", "iot_kl",
            "loss_radial", "loss_local", "loss_boundary", "loss_crypt",
        ]
        for k in extra_keys:
            if k not in self.history:
                self.history[k] = []

    def _train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        cfg = self.cfg2v2
        manifold = self.model._get_manifold()

        sums: Dict[str, float] = {
            "loss_total": 0.0, "loss_transcript": 0.0,
            "loss_iot": 0.0, "iot_kl": 0.0,
            "loss_radial": 0.0, "loss_local": 0.0,
            "loss_boundary": 0.0, "loss_crypt": 0.0,
        }
        n_batches = 0

        for batch in loader:
            x      = batch["expression"].to(self.device)
            labels = batch.get("crypt_label")
            if labels is not None:
                labels = labels.to(self.device)

            # Donor IDs for this batch (integer tensor on device, or None)
            donor_batch = None
            if self._donor_ids is not None:
                idx_np = batch["idx"].numpy()
                donor_batch = torch.tensor(
                    self._donor_ids[idx_np], dtype=torch.long, device=self.device
                )

            z = self.model(x)

            # Derive zone labels for this batch
            zones_b = None
            if labels is not None:
                valid_mask = ~torch.isnan(labels)
                if valid_mask.sum() > 0:
                    zones_b = depth_to_zone(labels.clamp(0.0, 1.0))
                    zones_b[~valid_mask] = -1  # sentinel for NaN entries

            # --- transcript neighborhood ---
            l_t = transcript_neighbor_loss(
                z, x, manifold,
                k=cfg.k_pos, margin=cfg.triplet_margin, n_neg=cfg.n_neg,
            )

            # --- optional crypt MSE anchor ---
            l_c = z.new_zeros(1).squeeze()
            if labels is not None and cfg.lambda_crypt > 0:
                l_c = crypt_axis_loss(z, labels, label_type=cfg.label_type,
                                      margin=cfg.rank_margin)

            # --- multi-scale bilevel IOT ---
            l_iot = z.new_zeros(1).squeeze()
            iot_kl = 0.0
            if labels is not None and cfg.lambda_iot > 0:
                l_iot, iot_comps = multiscale_inverse_ot_loss(
                    z=z, labels=labels, expr_batch=x, manifold=manifold,
                    epsilon=cfg.epsilon_ot, n_sink_iter=cfg.n_sink_iter,
                    tau_zone=cfg.tau_zone, tau_expr=cfg.tau_expr,
                    w_zone=cfg.w_zone, w_adjacent=cfg.w_adjacent, w_expr=cfg.w_expr,
                    max_zone_gap=cfg.max_zone_gap,
                    n_star_sink_iter=cfg.n_star_sink_iter,
                    donor_ids=donor_batch,
                )
                iot_kl = iot_comps["iot_kl"]

            # --- radial geometry: center + margin ---
            l_radial = z.new_zeros(1).squeeze()
            if zones_b is not None and cfg.lambda_radial > 0:
                valid_zone_mask = zones_b >= 0
                if valid_zone_mask.sum() >= 2:
                    z_val = z[valid_zone_mask]
                    zn_val = zones_b[valid_zone_mask]
                    l_center = zone_radial_center_loss(
                        z_val, zn_val, cfg.zone_target_centers
                    )
                    l_margin = adjacent_radial_margin_loss(
                        z_val, zn_val, cfg.radial_margin
                    )
                    l_radial = l_center + cfg.eta_margin * l_margin
                    if cfg.lambda_radius_var_floor > 0:
                        l_radial = l_radial + cfg.lambda_radius_var_floor * \
                            within_zone_radius_variance_floor(z_val, zn_val, cfg.sigma_min)

            # --- within-zone local transcript ---
            l_local = z.new_zeros(1).squeeze()
            if zones_b is not None and cfg.lambda_local > 0:
                valid_zone_mask = zones_b >= 0
                if valid_zone_mask.sum() >= cfg.k_local + 2:
                    l_local = within_zone_triplet_loss(
                        z[valid_zone_mask], x[valid_zone_mask],
                        zones_b[valid_zone_mask], manifold,
                        k=cfg.k_local, margin=cfg.triplet_margin,
                        n_neg=cfg.n_neg_local,
                    )

            # --- boundary penalty ---
            l_bnd = boundary_penalty(z, cfg.boundary_threshold)

            loss = (
                cfg.lambda_transcript * l_t
                + cfg.lambda_crypt     * l_c
                + cfg.lambda_iot       * l_iot
                + cfg.lambda_radial    * l_radial
                + cfg.lambda_local     * l_local
                + cfg.lambda_boundary  * l_bnd
            )

            self.optimizer.zero_grad()
            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.optimizer.step()

            sums["loss_total"]      += loss.detach().item()
            sums["loss_transcript"] += l_t.detach().item()
            sums["loss_crypt"]      += l_c.detach().item()
            sums["loss_iot"]        += l_iot.detach().item()
            sums["iot_kl"]          += iot_kl
            sums["loss_radial"]     += l_radial.detach().item()
            sums["loss_local"]      += l_local.detach().item()
            sums["loss_boundary"]   += l_bnd.detach().item()
            n_batches += 1

        d = max(n_batches, 1)
        return {k: v / d for k, v in sums.items()}
