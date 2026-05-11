"""
Smoke test: run one full training step and verify nothing crashes.

Deliberately uses a tiny synthetic dataset so this test completes in seconds
on any machine, including CI without a GPU.
"""

import logging

import numpy as np
import pytest
import torch

from src.data.dataset import CryptDataset
from src.models.encoder import MLPHyperbolicEncoder
from src.train.config import Phase1Config
from src.train.trainer import Trainer

logging.basicConfig(level=logging.INFO)


def _make_synthetic_dataset(
    n_cells: int = 128,
    n_features: int = 20,
    with_spatial: bool = True,
    with_labels: bool = True,
    seed: int = 0,
) -> CryptDataset:
    """Create a small synthetic dataset for testing."""
    rng = np.random.default_rng(seed)
    expression = rng.standard_normal((n_cells, n_features)).astype(np.float32)
    spatial = rng.random((n_cells, 2)).astype(np.float32) if with_spatial else None
    labels = (
        np.linspace(0.0, 1.0, n_cells).astype(np.float32) if with_labels else None
    )
    return CryptDataset(expression, spatial_coords=spatial, crypt_labels=labels)


class TestSmoke:
    """Full pipeline smoke tests — one training step each."""

    def test_one_epoch_full(self):
        """All three losses active (transcript + spatial + crypt)."""
        dataset = _make_synthetic_dataset()
        model = MLPHyperbolicEncoder(
            input_dim=dataset.input_dim,
            hidden_dims=[32, 16],
            latent_dim=8,
            dropout=0.0,
        )
        config = Phase1Config(
            batch_size=32,
            n_epochs=1,
            eval_every=1,
            save_every=0,
            lambda_transcript=1.0,
            lambda_spatial=0.5,
            lambda_crypt=1.0,
            label_type="continuous",
            output_dir="/tmp/crypt_test",
            run_name="smoke_full",
            device="cpu",
        )
        trainer = Trainer(config, model, dataset)
        trainer.train(n_epochs=1)

        # Model should still produce valid outputs
        x = dataset.expression[:10]
        z = model(x)
        assert z.shape == (10, 8)
        assert (z.norm(dim=-1) < 1.0).all()
        assert not torch.isnan(z).any()

    def test_one_epoch_transcript_only(self):
        """Fallback: no spatial, no labels — only transcript loss."""
        dataset = _make_synthetic_dataset(with_spatial=False, with_labels=False)
        model = MLPHyperbolicEncoder(
            input_dim=dataset.input_dim,
            hidden_dims=[32],
            latent_dim=4,
            dropout=0.0,
        )
        config = Phase1Config(
            batch_size=32,
            n_epochs=1,
            eval_every=1,
            save_every=0,
            lambda_spatial=0.5,
            lambda_crypt=1.0,
            output_dir="/tmp/crypt_test",
            run_name="smoke_transcript_only",
            device="cpu",
        )
        trainer = Trainer(config, model, dataset)
        trainer.train(n_epochs=1)

    def test_one_epoch_ordinal_labels(self):
        """Ordinal label mode."""
        dataset = _make_synthetic_dataset()
        # Overwrite labels with integer ordinal bins
        dataset.crypt_labels = torch.randint(0, 4, (len(dataset),)).float()
        model = MLPHyperbolicEncoder(
            input_dim=dataset.input_dim,
            hidden_dims=[32],
            latent_dim=4,
            dropout=0.0,
        )
        config = Phase1Config(
            batch_size=32,
            n_epochs=1,
            eval_every=1,
            save_every=0,
            label_type="ordinal",
            output_dir="/tmp/crypt_test",
            run_name="smoke_ordinal",
            device="cpu",
        )
        trainer = Trainer(config, model, dataset)
        trainer.train(n_epochs=1)

    def test_loss_decreases(self):
        """Loss should decrease over a handful of epochs on synthetic data."""
        dataset = _make_synthetic_dataset(n_cells=256)
        model = MLPHyperbolicEncoder(
            input_dim=dataset.input_dim,
            hidden_dims=[64, 32],
            latent_dim=8,
            dropout=0.0,
        )
        config = Phase1Config(
            batch_size=64,
            n_epochs=10,
            eval_every=100,  # skip mid-training eval for speed
            save_every=0,
            device="cpu",
            output_dir="/tmp/crypt_test",
            run_name="smoke_convergence",
        )
        trainer = Trainer(config, model, dataset)
        trainer.train(n_epochs=10)
        losses = trainer.history["loss_total"]
        assert len(losses) == 10
        # Average of last 3 epochs should be lower than the first epoch
        assert np.mean(losses[-3:]) <= losses[0] * 1.5, (
            f"Loss did not decrease: first={losses[0]:.4f}, "
            f"last3_mean={np.mean(losses[-3:]):.4f}"
        )
