__all__ = ["CziBackend"]

import cv2
import numpy as np
from aicspylibczi import CziFile

from histoprep import functional as F

from ._base import SlideReaderBackend

ERROR_NON_MOSAIC = "HistoPrep does not support reading non-mosaic czi-files."
BACKGROUND_COLOR = (1.0, 1.0, 1.0)
MIN_LEVEL_DIMENSION = 1024


class CziBackend(SlideReaderBackend):
    """Slide reader using `aicspylibczi.CziFile` as a backend (by Allen Institute
    for Cell Science).

    Args:
        path: Path to the slide image.
    """

    BACKEND_NAME = "CZI"

    def __init__(self, path: str) -> None:
        super().__init__(path)
        self.__reader = CziFile(path)
        if not self.__reader.is_mosaic():
            raise NotImplementedError(ERROR_NON_MOSAIC)
        # Get plane constraints.
        bbox = self.__reader.get_mosaic_bounding_box()
        self.__origo = (bbox.x, bbox.y)
        # Define dimensions and downsamples.
        slide_h, slide_w = (bbox.h, bbox.w)
        lvl = 0
        self.__level_dimensions, self.__level_downsamples = {}, {}
        while lvl == 0 or max(slide_w, slide_h) // 2**lvl >= MIN_LEVEL_DIMENSION:
            level_h, level_w = (slide_h // 2**lvl, slide_w // 2**lvl)
            self.__level_dimensions[lvl] = round(slide_h / 2**lvl), round(
                slide_w / 2**lvl
            )
            self.__level_downsamples[lvl] = slide_h / level_h, slide_w / level_w
            lvl += 1

    @property
    def reader(self) -> CziFile:
        """CziFile instance."""
        return self.__reader

    @property
    def data_bounds(self) -> tuple[int, int, int, int]:
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

    def read_level(self, level: int) -> np.ndarray:
        level = F._format_level(level, available=list(self.level_dimensions))
        return self.read_region(xywh=self.data_bounds, level=level)

    def read_region(self, xywh: tuple[int, int, int, int], level: int) -> np.ndarray:
        level = F._format_level(level, available=list(self.level_dimensions))
        x, y, w, h = xywh
        # Define allowed dims, output dims and expected dims.
        allowed_h, allowed_w = F._get_allowed_dimensions(
            xywh, dimensions=self.dimensions
        )
        output_h, output_w = round(h / 2**level), round(w / 2**level)
        # Read allowed reagion.
        scale_factor = 1 / 2**level
        if allowed_h * scale_factor < 1 or allowed_w * scale_factor < 1:
            # LibCzi crashes with zero size.
            return np.zeros((output_h, output_w, 3), dtype=np.uint8) + 255
        tile = self.__reader.read_mosaic(
            region=(self.__origo[0] + x, self.__origo[1] + y, allowed_w, allowed_h),
            scale_factor=scale_factor,
            C=0,
            background_color=BACKGROUND_COLOR,
        )[0]
        # Resize to match expected size (Zeiss's libCZI is buggy).
        excepted_h, excepted_w = (
            round(allowed_h / 2**level),
            round(allowed_w / 2**level),
        )
        tile_h, tile_w = tile.shape[:2]
        if tile_h != excepted_h or tile_w != excepted_w:
            tile = cv2.resize(
                tile, dsize=(excepted_w, excepted_h), interpolation=cv2.INTER_NEAREST
            )
        # Convert to RGB and pad.
        return F._pad_tile(
            cv2.cvtColor(tile, cv2.COLOR_BGR2RGB), shape=(excepted_h, excepted_w)
        )
