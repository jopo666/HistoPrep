import numpy as np

from histoprep import functional as F


def test_grayscale_metrics():
    # Create image.
    image = np.zeros((100, 100), dtype=np.uint8) + 255
    image[:20, :20] = 60
    image[50:60, 20:50] = 100
    image[80:, 80:] = 120

    # Detect tissue.
    __, tissue_mask = F.get_tissue_mask(image, threshold=200)
    metrics = F.calculate_metrics(image, tissue_mask)
    assert 29 == int(metrics.pop("laplacian_std"))
    assert metrics == {
        "background": 0.886,
        "black_pixels": 0.0,
        "white_pixels": 0.89,
        "gray_mean": 237.175,
        "gray_std": 51.692,
        "gray_q5": 60,
        "gray_q10": 60,
        "gray_q25": 60,
        "gray_q50": 100,
        "gray_q75": 120,
        "gray_q90": 120,
        "gray_q95": 255,
    }


def test_rgb_metrics():
    # Create image.
    image = np.zeros((100, 100, 3), dtype=np.uint8) + 255
    image[:20, :20, 1] = 0
    image[:20, :20, 2] = 0
    image[50:60, 20:50, 0] = 0
    image[50:60, 20:50, 1] = 128
    image[50:60, 20:50, 2] = 128
    image[80:, 80:, 0] = 0
    image[80:, 80:, 1] = 0
    image[80:, 80:, 2] = 255
    # Detect tissue.
    __, tissue_mask = F.get_tissue_mask(image, threshold=200)
    # Check metrics.
    metrics = F.calculate_metrics(image, tissue_mask, quantiles=[0.9])
    assert 33 == int(metrics.pop("laplacian_std"))
    assert metrics == {
        "background": 0.886,
        "black_pixels": 0.0,
        "white_pixels": 0.89,
        "gray_mean": 234.312,
        "gray_std": 59.78,
        "red_mean": 237.755,
        "red_std": 64.032,
        "green_mean": 231.39,
        "green_std": 70.251,
        "blue_mean": 240.355,
        "blue_std": 54.701,
        "hue_mean": 7.141,
        "hue_std": 26.801,
        "saturation_mean": 27.766,
        "saturation_std": 79.432,
        "brightness_mean": 250.876,
        "brightness_std": 22.51,
        "gray_q90": 90,
        "red_q90": 255,
        "green_q90": 128,
        "blue_q90": 255,
        "hue_q90": 120,
        "saturation_q90": 255,
        "brightness_q90": 255,
    }
