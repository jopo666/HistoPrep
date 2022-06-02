from typing import Dict, Tuple

import aicspylibczi
import cv2
import numpy

from ._base import Backend

__all__ = ["ZeissBackend"]

ZEISS_READABLE = ("czi",)


class ZeissBackend(Backend):
    def __init__(self, path: str):
        self.__path = path
        self.reader = aicspylibczi.CziFile(path)
        if not self.reader.is_mosaic():
            raise NotImplementedError(
                "HistoPrep does not support non-mosaic zeiss files."
            )
        # Get plane constraints.
        bbox = self.reader.get_mosaic_bounding_box()
        self.__dimensions = (bbox.h, bbox.w)
        self.__origo = (bbox.x, bbox.y)
        # Init caches.
        self.__level_dimensions = None
        self.__level_downsamples = None

    def get_dimensions(self) -> Tuple[int, int]:
        return self.__dimensions

    def get_level_dimensions(self) -> Dict[int, Tuple[int, int]]:
        if self.__level_dimensions is None:
            level_dimensions = {0: self.__dimensions}
            downsample = 1
            while True:
                if max(self.dimensions) // 2**downsample < 512:
                    break
                level_dimensions[downsample] = tuple(
                    x // 2**downsample for x in self.dimensions
                )
                downsample += 1
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
        # Calculate scale factor.
        scale_factor = 1 / 2**level
        # Read thumbnail.
        thumbnail = self.reader.read_mosaic(
            scale_factor=scale_factor,
            C=0,
            background_color=(1.0, 1.0, 1.0),
        )[0]
        # Convert BGR to RGB.
        return cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)

    def read_region(self, XYWH: Tuple[int, int, int, int], level: int) -> numpy.ndarray:
        # Unpack.
        x, y, w, h = XYWH
        # Calculate scale factor.
        scale_factor = 1 / 2**level
        # Fix coordinates.
        XYWH = (
            self.__origo[0] + int(x / scale_factor),
            self.__origo[1] + int(y / scale_factor),
            int(w / scale_factor),
            int(h / scale_factor),
        )
        # Read region.
        tile = self.reader.read_mosaic(
            region=XYWH,
            scale_factor=scale_factor,
            C=0,
            background_color=(1.0, 1.0, 1.0),
        )[0]
        # Convert BGR to RGB.
        return cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)

    def __repr__(self):
        return "ZEISS"
