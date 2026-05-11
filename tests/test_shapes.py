"""
Shape and numerical-stability tests.

These tests run without real data and complete in seconds.  They verify:
  * encoder output shapes,
  * all points remain strictly inside the Poincaré ball,
  * pairwise hyperbolic distance matrix is symmetric and non-negative,
  * each loss returns a finite scalar.
"""

import numpy as np
import pytest
import torch

from src.models.encoder import MLPHyperbolicEncoder
from src.losses.neighborhood import pairwise_hyp_dist, transcript_neighbor_loss
from src.losses.spatial import spatial_smoothness_loss
from src.losses.crypt_axis import crypt_axis_loss
from src.data.dataset import CryptDataset
from src.data.preprocessing import normalize_crypt_labels, pca_reduce


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

@pytest.fixture
def encoder():
    return MLPHyperbolicEncoder(
        input_dim=16, hidden_dims=[32, 16], latent_dim=4,
        dropout=0.0, curvature=1.0, max_norm=0.95,
    )


@pytest.fixture
def batch():
    torch.manual_seed(0)
    B, D = 32, 16
    x = torch.randn(B, D)
    spatial = torch.rand(B, 2)
    labels_cont = torch.linspace(0.0, 1.0, B)
    labels_ord = torch.randint(0, 4, (B,)).float()
    return x, spatial, labels_cont, labels_ord


# --------------------------------------------------------------------------
# Encoder
# --------------------------------------------------------------------------

class TestEncoder:
    def test_output_shape(self, encoder, batch):
        x, _, _, _ = batch
        z = encoder(x)
        assert z.shape == (x.shape[0], encoder.latent_dim), "Wrong output shape"

    def test_points_inside_ball(self, encoder, batch):
        x, _, _, _ = batch
        z = encoder(x)
        radii = z.norm(dim=-1)
        assert (radii < 1.0).all(), f"Points outside ball: max radius = {radii.max():.4f}"
        assert (radii < encoder.max_norm + 1e-4).all(), "Radius exceeds max_norm"

    def test_radius_helper(self, encoder, batch):
        x, _, _, _ = batch
        z = encoder(x)
        r = encoder.hyperbolic_radius(z)
        assert r.shape == (x.shape[0],)

    def test_euclidean_encode_shape(self, encoder, batch):
        x, _, _, _ = batch
        v = encoder.encode_euclidean(x)
        assert v.shape == (x.shape[0], encoder.latent_dim)

    def test_no_nan(self, encoder, batch):
        x, _, _, _ = batch
        z = encoder(x)
        assert not torch.isnan(z).any(), "NaN in encoder output"
        assert not torch.isinf(z).any(), "Inf in encoder output"

    def test_gradients_flow(self, encoder, batch):
        x, _, _, _ = batch
        z = encoder(x)
        loss = z.norm(dim=-1).mean()
        loss.backward()
        for name, p in encoder.named_parameters():
            if p.requires_grad and p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"NaN grad in {name}"


# --------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------

class TestNeighborhoodLoss:
    def test_scalar_finite(self, encoder, batch):
        x, _, _, _ = batch
        z = encoder(x)
        manifold = encoder._get_manifold()
        loss = transcript_neighbor_loss(z, x, manifold, k=4, margin=0.5)
        assert loss.shape == torch.Size([]), "Should be scalar"
        assert torch.isfinite(loss), f"Loss not finite: {loss}"

    def test_non_negative(self, encoder, batch):
        x, _, _, _ = batch
        z = encoder(x)
        manifold = encoder._get_manifold()
        loss = transcript_neighbor_loss(z, x, manifold, k=4, margin=0.5)
        assert loss >= 0, "Triplet loss must be non-negative"

    def test_pairwise_dist_shape(self, encoder, batch):
        x, _, _, _ = batch
        z = encoder(x)
        manifold = encoder._get_manifold()
        D = pairwise_hyp_dist(z, manifold)
        B = x.shape[0]
        assert D.shape == (B, B), f"Expected ({B},{B}), got {D.shape}"

    def test_pairwise_dist_symmetric(self, encoder, batch):
        x, _, _, _ = batch
        z = encoder(x)
        manifold = encoder._get_manifold()
        D = pairwise_hyp_dist(z, manifold)
        # Hyperbolic distances at large radii accumulate float32 rounding errors
        # in the Möbius addition; 1e-2 absolute tolerance is appropriate.
        assert torch.allclose(D, D.T, atol=1e-2), "Distance matrix not symmetric"

    def test_pairwise_dist_non_negative(self, encoder, batch):
        x, _, _, _ = batch
        z = encoder(x)
        manifold = encoder._get_manifold()
        D = pairwise_hyp_dist(z, manifold)
        assert (D >= -1e-6).all(), "Distances should be non-negative"


class TestSpatialLoss:
    def test_scalar_finite(self, encoder, batch):
        x, spatial, _, _ = batch
        z = encoder(x)
        manifold = encoder._get_manifold()
        loss = spatial_smoothness_loss(z, spatial, manifold, k=4)
        assert loss.shape == torch.Size([])
        assert torch.isfinite(loss)

    def test_non_negative(self, encoder, batch):
        x, spatial, _, _ = batch
        z = encoder(x)
        manifold = encoder._get_manifold()
        loss = spatial_smoothness_loss(z, spatial, manifold, k=4)
        assert loss >= 0


class TestCryptAxisLoss:
    def test_continuous_scalar_finite(self, encoder, batch):
        x, _, labels_cont, _ = batch
        z = encoder(x)
        loss = crypt_axis_loss(z, labels_cont, label_type="continuous")
        assert loss.shape == torch.Size([])
        assert torch.isfinite(loss)

    def test_ordinal_scalar_finite(self, encoder, batch):
        x, _, _, labels_ord = batch
        z = encoder(x)
        loss = crypt_axis_loss(z, labels_ord, label_type="ordinal")
        assert loss.shape == torch.Size([])
        assert torch.isfinite(loss)

    def test_all_nan_labels(self, encoder, batch):
        x, _, _, _ = batch
        z = encoder(x)
        nan_labels = torch.full((x.shape[0],), float("nan"))
        loss = crypt_axis_loss(z, nan_labels, label_type="continuous")
        assert float(loss) == 0.0

    def test_ordinal_non_negative(self, encoder, batch):
        x, _, _, labels_ord = batch
        z = encoder(x)
        loss = crypt_axis_loss(z, labels_ord, label_type="ordinal")
        assert loss >= 0


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

class TestDataset:
    def test_basic_construction(self):
        N, D = 50, 10
        expr = np.random.randn(N, D).astype(np.float32)
        ds = CryptDataset(expr)
        assert len(ds) == N
        assert ds.input_dim == D
        assert not ds.has_spatial
        assert not ds.has_labels

    def test_with_all_fields(self):
        N, D = 50, 10
        expr = np.random.randn(N, D).astype(np.float32)
        spatial = np.random.randn(N, 2).astype(np.float32)
        labels = np.linspace(0, 1, N).astype(np.float32)
        ds = CryptDataset(expr, spatial_coords=spatial, crypt_labels=labels)
        assert ds.has_spatial
        assert ds.has_labels
        item = ds[0]
        assert "expression" in item
        assert "spatial_coords" in item
        assert "crypt_label" in item

    def test_getitem_shapes(self):
        N, D = 20, 8
        expr = np.random.randn(N, D).astype(np.float32)
        ds = CryptDataset(expr)
        item = ds[0]
        assert item["expression"].shape == (D,)


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------

class TestPreprocessing:
    def test_normalize_labels(self):
        labels = np.array([0.0, 1.0, 2.0, 3.0])
        normed = normalize_crypt_labels(labels)
        assert np.allclose(normed.min(), 0.0)
        assert np.allclose(normed.max(), 1.0)

    def test_pca_reduce_shape(self):
        X = np.random.randn(100, 50).astype(np.float32)
        emb, _ = pca_reduce(X, n_components=10)
        assert emb.shape == (100, 10)
        assert emb.dtype == np.float32
