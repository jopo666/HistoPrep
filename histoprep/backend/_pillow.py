__all__ = ["PillowBackend"]

import numpy as np
from PIL import Image

from ._base import BaseBackend
from ._functional import allowed_dimensions, divide_xywh, format_level, pad_tile

MIN_LEVEL_DIMENSION = 512
Image.MAX_IMAGE_PIXELS = 15_000 * 15_000


class PillowBackend(BaseBackend):
    def __init__(self, path: str) -> None:
        """PIL reader backend, requires that the whole image is read into memory.

        Args:
            path: Path to the slide image.
        """
        super().__init__(path)
        # Read full image.
        self.__pyramid = {0: Image.open(path)}
        # Generate downsamples.
        slide_h, slide_w = self.dimensions
        lvl = 0
        self.__level_dimensions, self.__level_downsamples = {}, {}
        while lvl == 0 or max(slide_w, slide_h) // 2**lvl >= MIN_LEVEL_DIMENSION:
            level_h, level_w = (slide_h // 2**lvl, slide_w // 2**lvl)
            self.__level_dimensions[lvl] = (slide_h // 2**lvl, slide_w // 2**lvl)
            self.__level_downsamples[lvl] = (slide_h / level_h, slide_w / level_w)
            lvl += 1

    @property
    def reader(self) -> None:
        """PIL image at level=0."""
        return self.__pyramid[0]

    @property
    def data_bounds(self) -> tuple[int, int, int, int]:
        h, w = self.dimensions
        return (0, 0, w, h)

    @property
    def dimensions(self) -> tuple[int, int]:
        # PIL has (width, height) size.
        return self.__pyramid[0].size[::-1]

    @property
    def level_count(self) -> int:
        return len(self.level_dimensions)

    @property
    def level_dimensions(self) -> dict[int, tuple[int, int]]:
        return self.__level_dimensions

    @property
    def level_downsamples(self) -> dict[int, tuple[float, float]]:
        return self.__level_downsamples

    def read_level(self, level: int) -> np.ndarray:
        level = format_level(level, available=list(self.level_dimensions))
        self.__lazy_load(level)
        return np.array(self.__pyramid[level])

    def read_region(self, xywh: tuple[int, int, int, int], level: int) -> np.ndarray:
        level = format_level(level, available=list(self.level_dimensions))
        self.__lazy_load(level)
        # Read allowed region.
        x, y, output_w, output_h = divide_xywh(xywh, self.level_downsamples[level])
        allowed_h, allowed_w = allowed_dimensions(
            xywh=(x, y, output_w, output_h), dimensions=self.level_dimensions[level]
        )
        tile = np.array(
            self.__pyramid[level].crop((x, y, x + allowed_w, y + allowed_h))
        )
        # Pad tile.
        return pad_tile(tile, shape=(output_h, output_w))

    def __lazy_load(self, level: int) -> None:
        if level not in self.__pyramid:
            height, width = self.level_dimensions[level]
            self.__pyramid[level] = self.__pyramid[0].resize(
                (width, height), resample=Image.Resampling.NEAREST
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(path={self.path})"
