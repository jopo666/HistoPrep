__all__ = ["CziReader"]

import warnings

import cv2
import numpy as np
from aicspylibczi import CziFile

from ._base import BaseReader

ERROR_NON_MOSAIC = "HistoPrep does not support reading non-mosaic czi-files."
WARN_NONZERO_LEVEL = (
    "Reading regions from non-zero slide level(s) is not stable "
    "due to a bug in the underlying libCZI implementation."
)
DOWNSAMPLES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
BACKGROUND_COLOR = (1.0, 1.0, 1.0)


class CziReader(BaseReader):
    def __init__(self, path: str) -> None:
        super().__init__(path)
        # Read slide with `aicspylibczi`.
        self.__reader = CziFile(path)
        if not self.__reader.is_mosaic():
            raise NotImplementedError(ERROR_NON_MOSAIC)
        # Get plane constraints.
        bbox = self.__reader.get_mosaic_bounding_box()
        self.__dimensions = (bbox.h, bbox.w)
        self.__origo = (bbox.x, bbox.y)
        # Define level downsamples and dimensions.
        self.__level_dimensions = {}
        self.__level_downsamples = {}
        for level, downsample in enumerate(DOWNSAMPLES):
            level_h, level_w = (round(x / downsample) for x in self.dimensions)
            self.__level_dimensions[level] = (level_h, level_w)
            self.__level_downsamples[level] = (
                self.__dimensions[0] / level_h,
                self.__dimensions[1] / level_w,
            )
        # Set warning flag.
        self.__warn_about_nonzero_level = True

    @property
    def backend(self) -> CziFile:
        """`aicspylibczi.CziFile` instance."""
        return self.__reader

    @property
    def data_bounds(self) -> tuple[int, int, int, int]:
        # Not defined.
        h, w = self.__dimensions
        return (0, 0, w, h)

    @property
    def dimensions(self) -> tuple[int, int]:
        return self.__dimensions

    @property
    def level_count(self) -> int:
        return len(self.level_dimensions)

    @property
    def level_dimensions(self) -> dict[int, tuple[int, int]]:
        return self.__level_dimensions

    @property
    def level_downsamples(self) -> dict[int, tuple[int, int]]:
        return self.__level_downsamples

    def read_level(self, level: int) -> np.ndarray:
        level = self._check_and_format_level(level)
        level_h, level_w = self.level_dimensions[level]
        return self._read_region(xywh=(0, 0, level_w, level_h), level=level)

    def _read_region(self, xywh: tuple[int, int, int, int], level: int) -> np.ndarray:
        if level > 0 and self.__warn_about_nonzero_level:
            warnings.warn(WARN_NONZERO_LEVEL)
            self.__warn_about_nonzero_level = False
        downsample = 2**level
        # Adjust coordinates.
        x, y, w, h = xywh
        adjusted_xywh = (
            self.__origo[0] + x * downsample,
            self.__origo[1] + y * downsample,
            w * downsample,
            h * downsample,
        )
        # Read level.
        region_data = self.__reader.read_mosaic(
            region=adjusted_xywh,
            scale_factor=1 / downsample,  # This does not always work as expected...
            C=0,
            background_color=BACKGROUND_COLOR,
        )[0]
        region_h, region_w = region_data.shape[:2]
        if region_h != h or region_w != w:
            # Resize to actually match (w, h) because Zeiss's libCZI is bugggggggy.
            region_data = cv2.resize(
                region_data, dsize=(w, h), interpolation=cv2.INTER_NEAREST
            )
        # Convert to RGB.
        return cv2.cvtColor(region_data, cv2.COLOR_BGR2RGB)
