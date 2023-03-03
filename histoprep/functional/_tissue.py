__all__ = ["get_tissue_mask", "clean_tissue_mask"]


from typing import Optional, Union

import cv2
import numpy as np
from PIL import Image

from ._check import check_image

ERROR_THRESHOLD = "Threshold should be an integer in range [0, 255], got {}."
ERROR_MULTIPLIER = "Threshold multiplier should be a positive float, got {}."
ERROR_SIGMA = "Sigma for gaussian blur should be a positive float, got {}."

MAX_THRESHOLD = 255
WHITE_PIXEL = 255
BLACK_PIXEL = 0
SIGMA_NO_OP = 0.0
GRAY_NDIM = 2


def get_tissue_mask(
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

    Raises:
        ValueError: Threshold not between 0 and 255.
        ValueError: Multiplier is negative.
        ValueError: Sigma is negative.

    Returns:
        Tuple with `threshold` and `tissue_mask` (0=background and 1=tissue).
    """
    # Check image and convert to array.
    image = check_image(image)
    # Check arguments.
    if threshold is not None and not 0 <= threshold <= MAX_THRESHOLD:
        raise ValueError(ERROR_THRESHOLD.format(threshold))
    if multiplier < 0:
        raise ValueError(ERROR_MULTIPLIER.format(multiplier))
    if sigma < 0:
        raise ValueError(ERROR_SIGMA.format(sigma))
    # Convert to grayscale.
    gray = image if image.ndim == GRAY_NDIM else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Gaussian blurring.
    blur = _gaussian_blur(image=gray, sigma=sigma, truncate=3.5)
    # Get threshold.
    if threshold is None:
        threshold = _otsu_threshold(
            gray=blur, ignore_white=ignore_white, ignore_black=ignore_black
        )
        threshold = max(min(255, int(threshold * multiplier + 0.5)), 0)
    # Global thresholding.
    thrsh, mask = cv2.threshold(blur, threshold, 1, cv2.THRESH_BINARY_INV)
    return int(thrsh), mask


def clean_tissue_mask(
    tissue_mask: np.ndarray,
    min_area_pixel: int = 10,
    max_area_pixel: Optional[int] = None,
    min_area_relative: float = 0.2,
    max_area_relative: Optional[float] = 2.0,
) -> np.ndarray:
    """Remove too small/large contours from tissue mask.

    Args:
        tissue_mask: Tissue mask to be cleaned.
        min_area_pixel: Minimum pixel area for contours. Defaults to 10.
        max_area_pixel: Maximum pixel area for contours. Defaults to None.
        min_area_relative: Relative minimum contour area, calculated from the median
            contour area after filtering contours with `[min,max]_pixel` arguments
            (`min_area_relative * median(contour_areas)`). Defaults to 0.2.
        max_area_relative: Relative maximum contour area, calculated from the median
            contour area after filtering contours with `[min,max]_pixel` arguments
            (`max_area_relative * median(contour_areas)`). Defaults to 2.0.

    Returns:
        Tissue mask with too small/large contours removed.
    """
    contours, __ = cv2.findContours(
        tissue_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    contour_areas = np.array([cv2.contourArea(cnt) for cnt in contours])
    # Filter based on pixel values.
    selection = contour_areas >= min_area_pixel
    if max_area_pixel is not None:
        selection = selection & (contour_areas <= max_area_pixel)
    # Define relative min/max values.
    area_median = np.median(contour_areas[selection])
    area_min = area_median * min_area_relative
    area_max = None if max_area_relative is None else area_median * max_area_relative
    # Draw new mask.
    new_mask = np.zeros_like(tissue_mask)
    for select, area, cnt in zip(selection, contour_areas, contours):
        if select and area >= area_min and (area_max is None or area <= area_max):
            cv2.drawContours(new_mask, [cnt], -1, 1, -1)
    return new_mask


def _otsu_threshold(*, gray: np.ndarray, ignore_white: bool, ignore_black: bool) -> int:
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


def _gaussian_blur(
    *, image: np.ndarray, sigma: float, truncate: float = 3.5
) -> np.ndarray:
    """Apply gaussian blurring."""
    if sigma == SIGMA_NO_OP:
        return image
    ksize = int(truncate * sigma + 0.5)
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(image, ksize=(ksize, ksize), sigmaX=sigma, sigmaY=sigma)
