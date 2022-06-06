from typing import Dict, Tuple

import numpy
from PIL import Image

from ...functional import resize_image
from ._base import Backend

__all__ = ["PillowBackend"]

PILLOW_READABLE = ("jpeg", "jpg", "png")


class PillowBackend(Backend):
    def __init__(self, path: str):
        # Read image file.
        self.__level_images = {0: Image.open(path)}
        # Add downsamples.
        level = 1
        self.__level_dimensions = {0: self.dimensions}
        self.__level_downsamples = {0: (1.0, 1.0)}
        while True:
            if max(self.dimensions) // 2**level < 128:
                break
            h, w = tuple(x // 2**level for x in self.dimensions)
            self.__level_dimensions[level] = (h, w)
            self.__level_downsamples[level] = (
                self.dimensions[0] / h,
                self.dimensions[0] / w,
            )
            level += 1

    def get_dimensions(self) -> Dict[int, Tuple[int, int]]:
        return self.__level_images[0].size[::-1]

    def get_level_dimensions(self) -> Dict[int, Tuple[int, int]]:
        return self.__level_dimensions

    def get_level_downsamples(self) -> Dict[int, Tuple[int, int]]:
        return self.__level_downsamples

    def get_thumbnail(self, level: int) -> numpy.ndarray:
        if level not in self.__level_images.keys():
            self.__level_images[level] = resize_image(
                self.__level_images[0],
                self.__level_dimensions[level],
                fast_resize=True,
            )
        return numpy.array(self.__level_images[level])

    def read_region(self, XYWH: Tuple[int, int, int, int], level: int) -> numpy.ndarray:
        # Unpack.
        x, y, w, h = XYWH
        if level not in self.__level_images.keys():
            self.__level_images[level] = resize_image(
                self.__level_images[0],
                self.__level_dimensions[level],
                fast_resize=True,
            )
        # Read region
        return numpy.array(
            self.__level_images[level].crop(
                (
                    x,
                    y,
                    x + w,
                    y + h,
                )
            )
        )

    def __repr__(self):
        return "PILLOW"
