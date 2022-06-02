import os
from typing import Iterable, List, Union

import numpy
from PIL import Image

from ..functional import resize_image
from ._multiprocess import multiprocess_loop

__all__ = ["read_image"]


def read_image(
    path: str, return_arr: bool = False
) -> Union[Image.Image, numpy.ndarray]:
    """Read image.

    Args:
        path: Path to image
        return_arr: Return array instead of PIL image. Defaults to False.

    Returns:
        Image from path.

    Example:
        ```python
        from histoprep.helpers import read_image

        image = read_image("path/to/image")
        arr = read_image("path/to/image", return_arr=True)
        ```
    """
    image = Image.open(path)
    if return_arr:
        return numpy.array(image)
    return image


def read_and_resize(paths: List[str], px: int) -> Iterable:
    """Helper function to load tiles with many workers."""
    for tile in multiprocess_loop(
        __load_and_resize,
        iterable=paths,
        use_imap=False,
        num_workers=min(len(paths), os.cpu_count()),
        px=px,
    ):
        yield tile


def __load_and_resize(path: str, px: int) -> numpy.ndarray:
    """Helper function to load an image and resize it immediatly."""
    if path is None:
        return numpy.zeros((px, px, 3), dtype=numpy.uint8) + 255
    return resize_image(read_image(path), (px, px), return_arr=True, fast_resize=True)
