from typing import Union

import numpy
from PIL import Image

from ._functional import _gaussian_blur, _thresholding, _to_grayscale
from ._helpers import _to_array

__all__ = ["detect_tissue"]


def detect_tissue(
    image: Union[numpy.ndarray, Image.Image],
    threshold: int = None,
    multiplier: float = 1.0,
    sigma: float = 1.0,
    remove_white: bool = False,
) -> numpy.ndarray:
    """Detect tissue from image.

    Args:
        image: Input image.
        threshold: Threshold for tissue detection. If set, will detect tissue by
            global thresholding, and otherwise Otsu's method is used to find
            a threshold. Defaults to None.
        multiplier: Otsu's method is used to find an optimal threshold by
            minimizing the weighted within-class variance. This threshold is
            then multiplied with `multiplier`. Used only if `threshold` is None.
            Defaults to 1.0.
        sigma: Sigma for gaussian blurring. Defaults to 1.0.
        remove_white: Does not consider white pixels with Otsu's method. Useful
            for slide images where large areas are artificially set to white.
            Defaults to False.

    Returns:
        Binary mask with 0=background and 1=tissue.

    Example:
        ```python
        import histoprep.functional as F
        from histoprep.helpers import read_image

        image = read_image("path/to/image.jpeg")
        tissue_mask = F.detect_tissue(image)
        ```
    """
    if threshold is not None and (
        not isinstance(threshold, int) or not 0 <= threshold <= 255
    ):
        raise TypeError("Threshold should be an integer in range [0, 255]")
    if not isinstance(multiplier, float) or multiplier <= 0:
        raise TypeError("Threshold multiplier should be a float over 0.")
    if not isinstance(sigma, (float, int)) or sigma < 0:
        raise TypeError("Sigma should be a positive float.")
    # Check image and convert to array.
    image = _to_array(image)
    # To grayscale.
    gray = _to_grayscale(image)
    if sigma > 0.0:
        # Gaussian blurring.
        gray = _gaussian_blur(gray, sigma)
    # Global thresholding.
    return _thresholding(
        gray=gray,
        threshold=threshold,
        multiplier=multiplier,
        remove_white=remove_white,
    )
