"""Tests for deeplens/optics/diffraclens.py — DiffractiveLens."""

import pytest
import torch


class TestDiffractiveLensInit:
    """Tests for DiffractiveLens initialization."""

    def test_init_empty(self):
        """DiffractiveLens can be created without a file."""
        from deeplens import DiffractiveLens

        old_dtype = torch.get_default_dtype()
        lens = DiffractiveLens()
        torch.set_default_dtype(old_dtype)
        assert lens.surfaces == []
        assert lens.sensor_size == (8.0, 8.0)
        assert lens.sensor_res == (2000, 2000)

    def test_init_with_surfaces(self, sample_diffraclens):
        """sample_diffraclens fixture creates a valid lens."""
        lens = sample_diffraclens
        assert len(lens.surfaces) == 1
        assert lens.d_sensor is not None


class TestDiffractiveLensPSF:
    """Tests for PSF computation."""

    def test_psf_shape(self, sample_diffraclens):
        """psf() returns [ks, ks] tensor."""
        lens = sample_diffraclens
        ks = 64
        psf = lens.psf(points=[0.0, 0.0, float("-inf")], ks=ks)
        assert psf.shape == (ks, ks)
        assert (psf >= 0).all()

    def test_psf_finite_depth(self, sample_diffraclens):
        """psf() works with finite depth."""
        lens = sample_diffraclens
        ks = 64
        psf = lens.psf(points=[0.0, 0.0, -500.0], ks=ks)
        assert psf.shape == (ks, ks)

    def test_psf_off_axis(self, sample_diffraclens):
        """psf() supports off-axis point sources."""
        lens = sample_diffraclens
        ks = 64
        psf = lens.psf(points=[0.3, 0.0, float("-inf")], ks=ks)
        assert psf.shape == (ks, ks)
        assert torch.isfinite(psf).all()
        assert abs(float(psf.sum()) - 1.0) < 1e-3

    def test_psf_batch(self, sample_diffraclens):
        """psf() supports a batch of points -> [N, ks, ks]."""
        lens = sample_diffraclens
        ks = 64
        points = [[0.0, 0.0, float("-inf")], [0.3, 0.0, float("-inf")]]
        psf = lens.psf(points=points, ks=ks)
        assert psf.shape == (2, ks, ks)


class TestDiffractiveLensDeviceTransfer:
    """Tests for device transfer."""

    def test_to_cpu(self, sample_diffraclens):
        """to(cpu) moves all tensors to CPU."""
        lens = sample_diffraclens
        lens.to(torch.device("cpu"))
        assert lens.d_sensor.device.type == "cpu"
