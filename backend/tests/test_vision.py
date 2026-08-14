import numpy as np

from app.vision import compression_stats, compute_svd


def test_svd_reconstruction_shape_and_metrics():
    image = np.tile(np.arange(32, dtype=np.uint8), (32, 1))

    reconstructed, mse, psnr, elapsed_ms = compute_svd(image, rank=4)

    assert reconstructed.shape == image.shape
    assert mse >= 0
    assert psnr > 0
    assert elapsed_ms >= 0


def test_compression_stats():
    ratio, reduction = compression_stats(1000, 250)

    assert ratio == 4
    assert reduction == 75
