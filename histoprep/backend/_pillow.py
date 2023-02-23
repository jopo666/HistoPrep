__all__ = ["PillowReader"]

import numpy as np
from PIL import Image

from histoprep.functional import multiply_xywh

from ._base import BaseReader

MIN_LEVEL_DIMENSION = 512
Image.MAX_IMAGE_PIXELS = 15_000 * 15_000


class PillowReader(BaseReader):
    def __init__(self, path: str) -> None:
        """Slide reader using Pillow as a backend.

        Args:
            path: Path to the slide image.
        """
        super().__init__(path)

    @property
    def backend(self) -> Image.Image:
        """Pillow image at level 0."""
        return self.__pyramid[0]

    @property
    def data_bounds(self) -> tuple[int, int, int, int]:
        # This is not defined for PIL images.
        h, w = self.dimensions
        return (0, 0, w, h)

    @property
    def dimensions(self) -> tuple[int, int]:
        return self.__level_dimensions[0]

    @property
    def level_count(self) -> int:
        """Number of slide levels."""
        return len(self.level_dimensions)

    @property
    def level_dimensions(self) -> dict[int, tuple[int, int]]:
        return self.__level_dimensions

    @property
    def level_downsamples(self) -> dict[int, tuple[float, float]]:
        return self.__level_downsamples

    def _read_slide(self, path: str) -> None:
        """Read full image and generate downsamples."""
        self.__pyramid = {0: Image.open(path)}
        self.__level_dimensions = {}
        self.__level_downsamples = {}
        level = 0
        slide_width, slide_height = self.__pyramid[0].size
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

    def _read_level(self, level: int) -> np.ndarray:
        self.__lazy_load(level)
        return np.array(self.__pyramid[level])

    def _read_region(self, xywh: tuple[int, int, int, int], level: int) -> np.ndarray:
        self.__lazy_load(level)
        x, y, w, h = multiply_xywh(xywh, self.level_downsamples[level])
        return np.array(self.__pyramid[level].crop((x, y, x + w, y + h)))

    def __lazy_load(self, level: int) -> None:
        if level not in self.__pyramid:
            height, width = self.level_dimensions[level]
            self.__pyramid[level] = self.__pyramid[0].resize(
                (width, height), resample=Image.Resampling.NEAREST
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(path={self.path})"
