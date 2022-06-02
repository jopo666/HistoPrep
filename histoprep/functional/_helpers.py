from typing import List, Tuple, Union

import numpy
from PIL import Image

from ._functional import _rgb2hsv, _to_grayscale

ALLOWED_REDUCTIONS = ["max", "median", "mean", "min"]

__all__ = [
    "downsample_image",
    "resize_image",
    "arr2pil",
    "rgb2hsv",
    "rgb2gray",
]


def _to_array(image: Union[Image.Image, numpy.ndarray]) -> numpy.ndarray:
    """Convert image to array and check it's a valid image."""
    if isinstance(image, Image.Image):
        image = numpy.array(image)
    elif not isinstance(image, numpy.ndarray):
        raise TypeError(
            "Input image should be an array or image not {}".format(type(image))
        )
    # Check image.
    if image.dtype != numpy.uint8:
        raise TypeError("Image dtype should be uint8 not {}.".format(image.dtype))
    if not image.ndim >= 2:
        raise TypeError("Number of dimensions is too low for an image")
    if image.ndim != 2 and image.shape[-1] != 3:
        raise TypeError(
            "Image should have 1 or 3 channels but found: {}.".format(image.shape[-3])
        )
    return image


def _five_crop(image: numpy.ndarray) -> List[numpy.ndarray]:
    """Take five crops from the image."""
    height, width = image.shape[:2]
    crop_height, crop_width = height // 2, width // 2
    # Corners.
    tl = image[:crop_height, :crop_width]
    tr = image[:crop_height, crop_width:]
    bl = image[crop_height:, :crop_width]
    br = image[crop_height:, crop_width:]
    # Center.
    center_h = crop_height // 2
    center_w = crop_width // 2
    center = image[center_h : crop_height + center_h, center_w : crop_width + center_w]
    return tl, tr, bl, br, center


def _reduce_values(values: numpy.ndarray, method: str) -> float:
    """Reduce values of values with given method."""
    if method not in ALLOWED_REDUCTIONS:
        raise ValueError(
            "Reduction %s not recognised. Select from %s."
            % (method, ALLOWED_REDUCTIONS)
        )
    if method == "max":
        reduced = numpy.max(values)
    elif method == "median":
        reduced = numpy.median(values)
    elif method == "mean":
        reduced = numpy.mean(values)
    elif method == "min":
        reduced = numpy.min(values)
    return float(reduced)


def rgb2gray(
    image: Union[numpy.ndarray, Image.Image]
) -> Union[numpy.ndarray, Image.Image]:
    """Convert an RGB image to grayscale.

    Args:
        image: RGB image.

    Returns:
        Grayscale image.
    """
    gray = _to_grayscale(_to_array(image))
    if isinstance(image, Image.Image):
        gray = Image.fromarray(gray)
    return gray


def rgb2hsv(
    image: Union[numpy.ndarray, Image.Image]
) -> Union[numpy.ndarray, Image.Image]:
    """Convert an RGB image to HSV.

    Args:
        image: RGB image.

    Returns:
        HSV image.
    """
    if image.ndim == 2:
        raise ValueError("Image has too few dimensions to bee an RGB image.")
    hsv = _rgb2hsv(_to_array(image))
    if isinstance(image, Image.Image):
        hsv = Image.fromarray(hsv)
    return hsv


def arr2pil(
    image: Union[numpy.ndarray, Image.Image],
    equalize: bool = False,
) -> Image.Image:
    """Convert numpy array to a Pillow image.

    Args:
        image: Image array.
        equalize: Equalise image, useful for plotting masks.

    Returns:
        Pillow image.

    Example:
        ```python
        import histoprep.functional as F
        from histoprep.helpers import read_image

        image = read_image("path/to/image.jpeg")
        tissue_mask = F.detect_tissue(image)
        # Create nice mask image for visualization.
        tissue_mask_pil = F.arr2pil(1 - tissue_mask, equalize=True)
        ```
    """
    image = _to_array(image)
    if equalize and image.max() != 0:
        image = (image / image.max()) * 255
    image = Image.fromarray(image.astype(numpy.uint8))
    return image


def downsample_image(
    image: Union[numpy.ndarray, Image.Image],
    max_dimension: int = 2048,
    return_arr: bool = False,
) -> Union[Image.Image, numpy.ndarray]:
    """
    Donwsaple image until one of the dimensions is less than ``max_dimension``.

    Args:
        image: Image to be downsampled.
        max_dimension: Maximum dimension size. Defaults to 2048.

    Returns:
        Downsampled image.

    Example:
        ```python
        import histoprep.functional as F
        from histoprep.helpers import read_image

        image = read_image("path/to/image.jpeg")
        downsampled = F.downsample_image(image, max_dimension=32)
        assert all(x<=32 for x in downsampled.size)
        ```
    """
    image = _to_array(image)
    height, width = image.shape[:2]
    factor = 0
    while width > max_dimension or height > max_dimension:
        factor += 1
        height = int(height / 2**factor)
        width = int(width / 2**factor)
    image = resize_image(image, (height, width), fast_resize=True)
    if not return_arr:
        return image
    else:
        return numpy.array(image)


def resize_image(
    image: Union[Image.Image, numpy.ndarray],
    shape: Union[int, Tuple[int]],
    return_arr: bool = False,
    fast_resize: bool = False,
) -> Union[Image.Image, numpy.ndarray]:
    """Resize image to desired shape.

    Args:
        image: Input image.
        shape: Shape of the output image. If shape is an integer then the image
            is resized to a square of (shape, shape).
        return_arr: Return array. Defaults to False.
        fast_resize: Uses nearest neigbour interpolation. Defaults to False.

    Returns:
        Resized image.

    Example:
        ```python
        import histoprep.functional as F
        from histoprep.helpers import read_image

        image = read_image("path/to/image.jpeg")
        resized_32 = F.resize_image(image, (32, 32))
        resized_32 = F.resize_image(image, 32)
        resized_32_arr = F.resize_image(image, 32, return_arr=True)
        ```
    """
    if isinstance(image, numpy.ndarray):
        image = Image.fromarray(image)
    if isinstance(shape, int):
        shape = (shape, shape)
    if fast_resize:
        image = image.resize(shape[::-1], 0)
    else:
        image = image.resize(
            shape[::-1],
        )
    if not return_arr:
        return image
    else:
        return numpy.array(image)
