from typing import Dict, Tuple

import numpy

try:
    import openslide
except Exception as e:
    raise ImportError(
        "Could not import `openslide` due to error {}. Make sure you have "
        "OpenSlide installed (https://openslide.org/download/).".format(e)
    )

from ._base import Backend

__all__ = ["OpenSlideBackend"]

OPENSLIDE_READABLE = (
    "svs",
    "svslide",
    "tif",
    "tiff",
    "vms",
    "vmu",
    "ndpi",
    "bif",
    "scn",
    "mrxs",
)


class OpenSlideBackend(Backend):
    def __init__(self, path: str):
        self.reader = openslide.OpenSlide(path)
        # Init caches.
        self.__level_dimensions = None
        self.__level_downsamples = None

    def get_dimensions(self) -> Tuple[int, int]:
        return self.level_dimensions[0]

    def get_level_dimensions(self) -> Dict[int, Tuple[int, int]]:
        if self.__level_dimensions is None:
            level_dimensions = {}
            for level, __ in enumerate(self.reader.level_downsamples):
                # OpenSlide returns X, Y and we want Y, X
                level_dimensions[level] = self.reader.level_dimensions[level][::-1]
            self.__level_dimensions = level_dimensions
        return self.__level_dimensions

    def get_level_downsamples(self) -> Dict[int, Tuple[int, int]]:
        if self.__level_downsamples is None:
            level_downsamples = {}
            for level, (y, x) in self.level_dimensions.items():
                level_downsamples[level] = (
                    self.dimensions[0] / y,
                    self.dimensions[1] / x,
                )
            self.__level_downsamples = level_downsamples
        return self.__level_downsamples

    def get_thumbnail(self, level: int) -> numpy.ndarray:
        # Read thumbnail.
        thumbnail = self.reader.get_thumbnail(self.reader.level_dimensions[level])
        # Convert to array and select only RGB channels (no alpha channel).
        return numpy.array(thumbnail)[..., :3]

    def read_region(self, XYWH: Tuple[int, int, int, int], level: int) -> numpy.ndarray:
        # Unpack.
        x, y, w, h = XYWH
        # Adjust x and y based on level.
        h_d, w_d = self.level_downsamples[level]
        x = int(x * w_d)
        y = int(y * h_d)
        # Read region.
        tile = self.reader.read_region(location=(x, y), level=level, size=(w, h))
        # Convert to array and select only RGB channels (no alpha channel).
        return numpy.array(tile)[..., :3]

    def __repr__(self):
        return "OPENSLIDE"
