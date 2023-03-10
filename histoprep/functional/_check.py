from __future__ import annotations

import numpy as np
from PIL import Image

ERROR_TYPE = "Expected an array image, not {}."
ERROR_DIMENSIONS = "Image should have 2 or 3 dimensions, not {}."
ERROR_CHANNELS = "Image should have 3 colour channels, not {}."
ERROR_DTYPE = "Expected image dtype to be uint8, not {}."
GRAYSCALE_NDIM = 2
RGB_NDIM = 3


def check_image(image: np.ndarray | Image.Image) -> np.ndarray:
    """Check that input is a valid RGB/L image and convert to it to an array."""
    if isinstance(image, Image.Image):
        image = np.array(image)
    if not isinstance(image, np.ndarray):
        raise TypeError(ERROR_TYPE.format(type(image)))
    if not (image.ndim == GRAYSCALE_NDIM or image.ndim == RGB_NDIM):
        raise TypeError(ERROR_DIMENSIONS.format(image.ndim))
    if image.ndim == RGB_NDIM and image.shape[-1] != RGB_NDIM:
        raise TypeError(ERROR_CHANNELS.format(image.shape[-1]))
    if image.dtype != np.uint8:
        raise TypeError(ERROR_DTYPE.format(image.dtype))
    return image
