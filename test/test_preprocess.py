import os

import pytest
from histoprep.functional import (
    PreprocessMetrics,
    channel_quantiles,
    data_loss,
    detect_tissue,
    sharpness,
)
from histoprep.functional._functional import _gaussian_blur, _to_grayscale
from histoprep.functional._helpers import _to_array
from histoprep.helpers import read_image

from paths import DATA_PATH

IMAGE = read_image(os.path.join(DATA_PATH, "tile.jpeg"))
__, MASK = detect_tissue(IMAGE)


def test_opencv():
    from histoprep.functional._functional import (
        _gaussian_blur,
        _rgb2hsv,
        _thresholding,
        _to_grayscale,
        _to_laplacian,
    )

    img = _to_array(IMAGE)
    gray = _to_grayscale(img)
    blur = _gaussian_blur(gray, 2.0)
    _thresholding(blur)
    _thresholding(blur, 150)
    _to_laplacian(gray)
    _rgb2hsv(img)


def test_channel_quantiles():
    arr_image = _to_array(IMAGE)
    gray_image = (_to_grayscale(arr_image) * 255).astype("uint8")
    channel_quantiles(IMAGE, MASK)
    channel_quantiles(arr_image, MASK)
    channel_quantiles(gray_image, MASK)


def test_passing_bad_quantiles_to_channel_quantiles():
    try:
        channel_quantiles(IMAGE, MASK, quantiles=[-1, 0])
    except ValueError:
        pass


def test_data_loss_function():
    arr_image = _to_array(IMAGE)
    gray_image = (_to_grayscale(arr_image) * 255).astype("uint8")
    data_loss(IMAGE)
    data_loss(arr_image)
    data_loss(gray_image)


def test_data_loss_outputs():
    # No loss.
    no_loss = _to_array(IMAGE)
    assert data_loss(no_loss)["white_pixels"] == 0.0
    assert data_loss(no_loss)["black_pixels"] == 0.0
    # White loss.
    white_pixels = no_loss.copy()
    white_pixels[:64, :64] = 255
    assert data_loss(white_pixels)["white_pixels"] > 0.0
    assert data_loss(white_pixels)["black_pixels"] == 0.0
    # Black loss.
    white_pixels = no_loss.copy()
    white_pixels[:64, :64] = 0
    assert data_loss(white_pixels)["white_pixels"] == 0.0
    assert data_loss(white_pixels)["black_pixels"] > 0.0
    # Both losses.
    both_losses = no_loss.copy()
    both_losses[:64, :64] = 0
    both_losses[64:, 64:] = 255
    assert data_loss(both_losses)["white_pixels"] > 0.0
    assert data_loss(both_losses)["black_pixels"] > 0.0


def test_sharpenss_function():
    arr_image = _to_array(IMAGE)
    gray_image = (_to_grayscale(arr_image) * 255).astype("uint8")
    sharpness(IMAGE)
    sharpness(arr_image)
    sharpness(gray_image)


def test_sharpness_function_with_blurred_image():
    blur = _gaussian_blur(_to_array(IMAGE), sigma=2)
    sharpness(IMAGE)["sharpness_max"] > sharpness(blur)["sharpness_max"]


def test_PreprocessMetrics():
    # Normal usage.
    metrics = PreprocessMetrics()
    # repr
    print(metrics)
    with pytest.raises(TypeError):
        # without threshold
        metrics(IMAGE)
    res = metrics(IMAGE, 200)
    assert isinstance(res, dict)
    # Sharpness.
    metrics = PreprocessMetrics(sharpness_reduce=["max", "min", "median"])
    res = metrics(IMAGE, 200)
    assert "sharpness_max" in res
    assert "sharpness_min" in res
    assert "sharpness_median" in res
    # Custom callback
    metrics = PreprocessMetrics(custom_callback=lambda x: {"dog": "good_boi"})
    res = metrics(IMAGE, 200)
    assert res["dog"] == "good_boi"
    # Bad threshold.
    metrics = PreprocessMetrics(custom_callback=lambda x: {"dog": "good_boi"})
    assert metrics(IMAGE, 0)["background"] == 1.0
