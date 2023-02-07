__all__ = ["PillowReader"]

import numpy as np
from PIL import Image, UnidentifiedImageError

from ._base import BaseReader
from ._exceptions import SlideReadingError

MAX_DOWNSAMPLE = 256


class PillowReader(BaseReader):
    def __init__(self, path: str) -> None:
        """Slide reader using Pillow as a backend.

        Args:
            path: Path to the slide image.
        """
        super().__init__(path)
        # Read full image and add downsample/dimensions.
        try:
            self.__pyramid = {0: Image.open(path)}
        except UnidentifiedImageError as e:
            raise SlideReadingError from e
        # Generate downsamples.
        width, height = self.__pyramid[0].size
        self.__level_dimensions = {0: (height, width)}
        self.__level_downsamples = {0: (1.0, 1.0)}
        level = 1
        while max(width, height) // 2**level >= MAX_DOWNSAMPLE:
            self.__level_dimensions[level] = (height // 2**level, width // 2**level)
            self.__level_downsamples[level] = (
                height / self.__level_dimensions[level][0],
                width / self.__level_dimensions[level][1],
            )
            level += 1

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
        return len(self.__pyramid)

    @property
    def level_dimensions(self) -> dict[int, tuple[int, int]]:
        return self.__level_dimensions

    @property
    def level_downsamples(self) -> dict[int, tuple[float, float]]:
        return self.__level_downsamples

    def read_level(self, level: int) -> np.ndarray:
        self.__lazy_load(level)
        return np.array(self.__pyramid[level])

    def _read_region(self, xywh: tuple[int, int, int, int], level: int) -> np.ndarray:
        # Lazy load pyramid.
        self.__lazy_load(level)
        # Unpack.
        (x, y, w, h) = xywh
        # Read region.
        return np.array(self.__pyramid[level].crop((x, y, x + w, y + h)))

    def __lazy_load(self, level: int) -> None:
        """Helper method to load level if it does not exist. loaded."""
        # Load levels lazily.
        if level not in self.__pyramid:
            height, width = self.level_dimensions[level]
            self.__pyramid[level] = self.__pyramid[0].resize(
                (width, height), resample=Image.Resampling.NEAREST
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(path={self.path})"
