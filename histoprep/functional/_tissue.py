__all__ = ["detect_tissue"]


from typing import Optional, Union

import cv2
import numpy as np
from PIL import Image

from ._check import GRAYSCALE_NDIM, check_image

ERROR_INVALID_THRESHOLD = "Threshold should be an integer in range [0, 255]"
ERROR_INVALID_MULTIPLIER = "Threshold multiplier should be a non-zero positive number."
ERROR_SIGMA = "Sigma should be a positive number."
MAX_THRESHOLD = 255
WHITE_PIXEL = 255
BLACK_PIXEL = 0
SIGMA_NO_OP = 0.0


def detect_tissue(
    image: Union[Image.Image, np.ndarray],
    *,
    threshold: Optional[int] = None,
    multiplier: float = 1.0,
    sigma: float = 1.0,
    ignore_white: bool = True,
    ignore_black: bool = True,
) -> tuple[int, np.ndarray]:
    """Detect tissue from image.

    Args:
        image: Input image.
        threshold: Threshold for tissue detection. If set, will detect tissue by
            global thresholding, and otherwise Otsu's method is used to find
            a threshold. Defaults to None.
        multiplier: Otsu's method is used to find an optimal threshold by
            minimizing the weighted within-class variance. This threshold is
            then multiplied with `multiplier`. Ignored if `threshold` is not None.
            Defaults to 1.0.
        sigma: Sigma for gaussian blurring. Defaults to 1.0.
        ignore_white: Does not consider white pixels with Otsu's method. Useful
            for slide images where large areas are artificially set to white.
            Defaults to True.
        ignore_white: Does not consider black pixels with Otsu's method. Useful
            for slide images where large areas are artificially set to black.
            Defaults to True.

    Returns:
        Tuple with `threshold` and `tissue_mask` (0=background and 1=tissue).
    """
    # Check image and convert to array.
    image = check_image(image)
    # Check arguments.
    if threshold is not None and not 0 <= threshold <= MAX_THRESHOLD:
        raise ValueError(ERROR_INVALID_THRESHOLD)
    if multiplier <= 0:
        raise ValueError(ERROR_INVALID_MULTIPLIER)
    if sigma < 0:
        raise ValueError(ERROR_SIGMA)
    # Convert to grayscale.
    gray = (
        image
        if image.ndim == GRAYSCALE_NDIM
        else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    )
    # Gaussian blurring.
    blur = gaussian_blur(image=gray, sigma=sigma, truncate=3.5)
    # Get threshold.
    if threshold is None:
        threshold = otsu_threshold(
            gray=blur, ignore_white=ignore_white, ignore_black=ignore_black
        )
        threshold = max(min(255, int(threshold * multiplier + 0.5)), 0)
    # Global thresholding.
    thrsh, mask = cv2.threshold(blur, threshold, 1, cv2.THRESH_BINARY_INV)
    return int(thrsh), mask


def clean_tissue_mask(
    tissue_mask: np.ndarray, min_area: float = 0.2, max_area: float = 2.0
) -> np.ndarray:
    """Remove too small/large contours from tissue mask.

    Args:
        tissue_mask: Tissue mask to be cleaned.
        min_area: Minimum contour area, calculated by `median(contour_area) * min_area`.
            Defaults to 0.2.
        max_area: Maximum contour area, calculated by `median(contour_area) * max_area`.
            Defaults to 2.0.

    Returns:
        Cleaned tissue mask.
    """
    # Detect contours and get their areas.
    contours, __ = cv2.findContours(
        tissue_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    areas = np.array([cv2.contourArea(cnt) for cnt in contours])
    # Define min and max values.
    min_area = np.median(areas) * min_area
    max_area = np.median(areas) * max_area
    # Initialize new mask.
    new_mask = np.zeros_like(tissue_mask)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Select only contours that fit into area range.
        if not min_area <= area <= max_area:
            continue
        # Draw to the new map.
        cv2.drawContours(new_mask, [cnt], -1, 1, -1)
    return new_mask


def otsu_threshold(*, gray: np.ndarray, ignore_white: bool, ignore_black: bool) -> int:
    """Helper function to calculate Otsu's thresold from a grayscale image."""
    # Collect values for Otsu's method.
    values = gray.flatten()
    if ignore_white:
        values = values[values != WHITE_PIXEL]
    if ignore_black:
        values = values[values != BLACK_PIXEL]
    # Get threshold.
    threshold, __ = cv2.threshold(
        values, None, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return threshold


def gaussian_blur(
    *, image: np.ndarray, sigma: float, truncate: float = 3.5
) -> np.ndarray:
    """Apply gaussian blurring."""
    if sigma == SIGMA_NO_OP:
        return image
    ksize = int(truncate * sigma + 0.5)
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(image, ksize=(ksize, ksize), sigmaX=sigma, sigmaY=sigma)
