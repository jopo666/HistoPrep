__all__ = ["OpenSlideReader"]

import numpy as np

from histoprep import functional as F

from ._base import BaseReader
from ._exceptions import SlideReadingError

ERROR_OPENSLIDE_IMPORT = (
    "Make sure you have OpenSlide installed (https://openslide.org/api/python/)."
)
OPENSLIDE_READABLE = (
    "svs",
    "vms",
    "vmu",
    "ndpi",
    "scn",
    "mrxs",
    "tiff",
    "svslide",
    "tif",
    "bif",
)

try:
    import openslide
except ImportError as e:
    raise ImportError(ERROR_OPENSLIDE_IMPORT) from e


class OpenSlideReader(BaseReader):
    def __init__(self, path: str) -> None:
        """Slide reader using OpenSlide as a backend.

        Args:
            path: Path to the slide image.
        """
        super().__init__(path)
        # Read slide with OpenSlide.
        try:
            self.__reader = openslide.OpenSlide(path)
        except openslide.OpenSlideError as e:
            raise SlideReadingError from e
        # Openslide has WH dimensions.
        self.__level_dimensions = {
            level: (h, w) for level, (w, h) in enumerate(self.__reader.level_dimensions)
        }
        # Calculate actual downsamples.
        self.__level_downsamples = {}
        slide_height, slide_width = self.dimensions
        for level, (level_h, level_w) in self.__level_dimensions.items():
            self.__level_downsamples[level] = (
                slide_height / level_h,
                slide_width / level_w,
            )

    @property
    def backend(self) -> openslide.OpenSlide:
        """`openslide.OpenSlide` instance."""
        return self.__reader

    @property
    def data_bounds(self) -> tuple[int, int, int, int]:
        properties = dict(self.__reader.properties)
        x_bound = int(properties.get("openslide.bounds-x", 0))
        y_bound = int(properties.get("openslide.bounds-y", 0))
        w_bound = int(properties.get("openslide.bounds-width", self.dimensions[1]))
        h_bound = int(properties.get("openslide.bounds-heigh", self.dimensions[0]))
        return (x_bound, y_bound, w_bound, h_bound)

    @property
    def dimensions(self) -> tuple[int, int]:
        return self.backend.dimensions[::-1]

    @property
    def level_count(self) -> int:
        return self.backend.level_count

    @property
    def level_dimensions(self) -> dict[int, tuple[int, int]]:
        return self.__level_dimensions

    @property
    def level_downsamples(self) -> dict[int, tuple[int, int]]:
        return self.__level_downsamples

    def read_level(self, level: int) -> np.ndarray:
        level = self._check_and_format_level(level)
        level_h, level_w = self.level_dimensions[level]
        return np.array(self.__reader.get_thumbnail(size=(level_w, level_h)))

    def _read_region(self, xywh: tuple[int, int, int, int], level: int) -> np.ndarray:
        # Unpack.
        x, y, w, h = F.downsample_xywh(xywh, downsample=self.level_downsamples[level])
        # Read region.
        tile = self.__reader.read_region(location=(x, y), level=level, size=(w, h))
        # Get only RGB channels (no alpha).
        return np.array(tile)[..., :3]
