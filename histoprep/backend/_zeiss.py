__all__ = ["CziReader"]

import warnings

import cv2
import numpy as np
from aicspylibczi import CziFile

from histoprep.functional import multiply_xywh

from ._base import BaseReader

ERROR_NON_MOSAIC = "HistoPrep does not support reading non-mosaic czi-files."
WARN_NONZERO_LEVEL = (
    "Reading regions from non-zero slide level(s) is not stable "
    "due to a bug in the underlying libCZI implementation."
)
BACKGROUND_COLOR = (1.0, 1.0, 1.0)
MIN_LEVEL_DIMENSION = 1024


class CziReader(BaseReader):
    def __init__(self, path: str) -> None:
        """Slide reader using a wrapper around `libCZI` as a backend
        (`aicspylibczi.CziFile` by Allen Institute for Cell Science).

        Args:
            path: Path to the slide image.
        """
        super().__init__(path)

    @property
    def backend(self) -> CziFile:
        """`aicspylibczi.CziFile` instance."""
        return self.__reader

    @property
    def data_bounds(self) -> tuple[int, int, int, int]:
        # Not defined.
        h, w = self.dimensions
        return (0, 0, w, h)

    @property
    def dimensions(self) -> tuple[int, int]:
        return self.__level_dimensions[0]

    @property
    def level_count(self) -> int:
        return len(self.level_dimensions)

    @property
    def level_dimensions(self) -> dict[int, tuple[int, int]]:
        return self.__level_dimensions

    @property
    def level_downsamples(self) -> dict[int, tuple[int, int]]:
        return self.__level_downsamples

    def _read_slide(self, path: str) -> None:
        # Read slide with `aicspylibczi`.
        self.__reader = CziFile(path)
        if not self.__reader.is_mosaic():
            raise NotImplementedError(ERROR_NON_MOSAIC)
        # Get plane constraints.
        bbox = self.__reader.get_mosaic_bounding_box()
        self.__origo = (bbox.x, bbox.y)
        # Define level downsamples and dimensions.
        self.__level_dimensions = {}
        self.__level_downsamples = {}
        level = 0
        slide_height, slide_width = (bbox.h, bbox.w)
        while (
            level == 0
            or max(slide_width, slide_height) // 2**level >= MIN_LEVEL_DIMENSION
        ):
            self.__level_dimensions[level] = (
                slide_height // 2**level,
                slide_width // 2**level,
            )
            self.__level_downsamples[level] = (
                slide_height / self.__level_dimensions[level][0],
                slide_width / self.__level_dimensions[level][1],
            )
            level += 1
        # Set warning flag.
        self.__warn_about_nonzero_level = True

    def _read_level(self, level: int) -> np.ndarray:
        return self._read_region(xywh=self.data_bounds, level=level)

    def _read_region(self, xywh: tuple[int, int, int, int], level: int) -> np.ndarray:
        if level > 0 and self.__warn_about_nonzero_level:
            warnings.warn(WARN_NONZERO_LEVEL)
            self.__warn_about_nonzero_level = False
        # Read level.
        x, y, w, h = xywh
        region_data = self.__reader.read_mosaic(
            region=(self.__origo[0] + x, self.__origo[1] + y, w, h),
            scale_factor=1 / 2**level,
            C=0,
            background_color=BACKGROUND_COLOR,
        )[0]
        region_h, region_w = region_data.shape[:2]
        expected_h, expected_w = round(h / 2**level), round(w / 2**level)
        if region_h != expected_h or region_w != expected_w:
            # Resize to actually match (w, h) because Zeiss's libCZI is bugggggggy.
            region_data = cv2.resize(
                region_data,
                dsize=(expected_w, expected_h),
                interpolation=cv2.INTER_NEAREST,
            )
        # Convert to RGB.
        return cv2.cvtColor(region_data, cv2.COLOR_BGR2RGB)
