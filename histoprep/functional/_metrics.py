from typing import Union

import cv2
import numpy as np
from PIL import Image

from ._check import check_image

ERROR_QUANTILES = "Quantiles should be between (0, 1)."

DEFAULT_QUANTILES = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
DEFAULT_SHAPE = (64, 64)
GRAYSCALE_NDIM = 2
MIN_TISSUE_PIXELS = 10
MIN_QUANTILE = 0.0
MAX_QUANTILE = 1.0
BLACK_PIXEL = 0
WHITE_PIXEL = 255


def get_image_metrics(
    image: Union[Image.Image, np.ndarray],
    tissue_mask: np.ndarray,
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    shape: tuple[int, int] = DEFAULT_SHAPE,
) -> dict[str, float]:
    """Calculate image metrics for preprocessing.

    The following metrics are computed:
        - Background percentage.
        - Percentage of black & white pixels (0 or 255).
        - Laplacian standard deviation.
        - Channel mean and std values for RGB/HSV/grayscale (if image is RGB).
        - Channel quantile values for RGB/HSV/grayscale (if image is RGB).

    Args:
        image: Input image
        tissue_mask: Tissue mask.
        quantiles: Possible quantile values to use. Defaults to
            (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95).
        shape: Resize shape for faster channel metric calculation. Defaults to (64, 64).

    Raises:
        ValueError: Quantiles are not between (0, 1).

    Returns:
        Dictionary of image metrics.
    """
    image = check_image(image)
    if not all(MIN_QUANTILE < x < MAX_QUANTILE for x in quantiles):
        raise ValueError(ERROR_QUANTILES)
    metrics = {}
    # Generate images for metric calculation
    if image.ndim > GRAYSCALE_NDIM:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    else:
        gray = image
    # Calculate metrics.
    metrics["background"] = ((tissue_mask == 0).sum() / tissue_mask.size).round(3)
    metrics.update(get_data_loss(gray))
    metrics.update(get_laplacian_std(gray))
    # Resize images for quicker channel metrics.
    tissue_mask = cv2.resize(tissue_mask, shape, interpolation=cv2.INTER_NEAREST)
    gray = cv2.resize(gray, shape, interpolation=cv2.INTER_NEAREST)
    if image.ndim > GRAYSCALE_NDIM:
        image = cv2.resize(image, shape, interpolation=cv2.INTER_NEAREST)
        hsv = cv2.resize(hsv, shape, interpolation=cv2.INTER_NEAREST)  # type: ignore
    # Check that there is tissue...
    if tissue_mask.sum() < MIN_TISSUE_PIXELS:
        tissue_mask[...] = 1
    # Channel mean and std.
    metrics.update(get_mean_and_std(gray, ["gray"]))
    if image.ndim > GRAYSCALE_NDIM:
        metrics.update(get_mean_and_std(image, ["red", "green", "blue"]))
        metrics.update(get_mean_and_std(hsv, ["hue", "saturation", "brightness"]))
    # Channel quantiles.
    metrics.update(get_quantiles(gray, tissue_mask, quantiles, ["gray"]))
    if image.ndim > GRAYSCALE_NDIM:
        metrics.update(
            get_quantiles(image, tissue_mask, quantiles, ["red", "green", "blue"])
        )
        metrics.update(
            get_quantiles(
                hsv, tissue_mask, quantiles, ["hue", "saturation", "brightness"]
            )
        )
    return metrics


def get_mean_and_std(image: np.ndarray, names: list[str]) -> dict[str, float]:
    """Collect mean and standard deviation for image."""
    if image.ndim == GRAYSCALE_NDIM:
        return {
            **_get_channel_mean(image, names[0]),
            **_get_channel_std(image, names[0]),
        }
    output = {}
    for channel_idx, name in enumerate(names):
        output.update(_get_channel_mean(image[..., channel_idx], name))
        output.update(_get_channel_std(image[..., channel_idx], name))
    return output


def get_quantiles(
    image: np.ndarray, tissue_mask: np.ndarray, quantiles: list[float], names: list[str]
) -> dict[str, float]:
    if image.ndim == GRAYSCALE_NDIM:
        return _get_channel_quantiles(image, tissue_mask, quantiles, names[0])
    output = {}
    for channel_idx, name in enumerate(names):
        output.update(
            _get_channel_quantiles(
                image[..., channel_idx], tissue_mask, quantiles, name
            )
        )
    return output


def get_data_loss(gray: np.ndarray) -> dict[str, float]:
    """Calculate percentage of black and white pixels."""
    return {
        "black_pixels": (gray == BLACK_PIXEL).sum() / gray.size,
        "white_pixels": (gray == WHITE_PIXEL).sum() / gray.size,
    }


def get_laplacian_std(gray: np.ndarray) -> dict[str, float]:
    """Calculate laplacian standard deviation for sharpness evaluation."""
    return {"laplacian_std": cv2.Laplacian(gray, cv2.CV_32F).std()}


def _get_channel_mean(channel: np.ndarray, name: str) -> dict[str, float]:
    """Calculate mean value for the channel."""
    return {f"{name}_mean": channel.mean().round(3).tolist()}


def _get_channel_std(channel: np.ndarray, name: str) -> dict[str, float]:
    """Calculate std value for the channel."""
    return {f"{name}_std": channel.std().round(3).tolist()}


def _get_channel_quantiles(
    channel: np.ndarray,
    tissue_mask: np.ndarray,
    quantiles: tuple[float, ...],
    name: str,
) -> dict[str, int]:
    """Calculate quantile values for the channel."""
    # Get quantiles.
    output = {}
    bins = np.cumsum(np.bincount(channel[tissue_mask == 1].flatten(), minlength=256))
    for q in quantiles:
        output[f"{name}_q{int(100*q)}"] = int(
            np.argwhere(bins > int(q * (tissue_mask == 1).sum()))[0]
        )
    return output
