__all__ = ["OpenSlideBackend", "OPENSLIDE_READABLE"]

import numpy as np

from histoprep import functional as F

from ._base import BaseBackend

ERROR_OPENSLIDE_IMPORT = (
    "Could not import `openslide-python`, make sure `OpenSlide` is installed "
    "(https://openslide.org/api/python/)."
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


class OpenSlideBackend(BaseBackend):
    BACKEND_NAME = "OPENSLIDE"

    def __init__(self, path: str) -> None:
        """Slide reader using OpenSlide as a backend.

        Args:
            path: Path to the slide image.
        """
        super().__init__(path)
        self.__reader = openslide.OpenSlide(path)
        # Openslide has (width, height) dimensions.
        self.__level_dimensions = {
            lvl: (h, w) for lvl, (w, h) in enumerate(self.__reader.level_dimensions)
        }
        # Calculate actual downsamples.
        slide_h, slide_w = self.dimensions
        self.__level_downsamples = {}
        for lvl, (level_h, level_w) in self.__level_dimensions.items():
            self.__level_downsamples[lvl] = (slide_h / level_h, slide_w / level_w)

    @property
    def reader(self) -> openslide.OpenSlide:
        """OpenSlide instance."""
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
        return self.level_dimensions[0]

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
        level_h, level_w = self.level_dimensions[level]
        return np.array(self.__reader.get_thumbnail(size=(level_w, level_h)))

    def read_region(self, xywh: tuple[int, int, int, int], level: int) -> np.ndarray:
        level = F._format_level(level, available=list(self.level_dimensions))
        # Only width and height have to be adjusted for the level.
        x, y, *__ = xywh
        *__, w, h = F._divide_xywh(xywh, self.level_downsamples[level])
        # Read allowed region.
        allowed_h, allowed_w = F._get_allowed_dimensions((x, y, w, h), self.dimensions)
        tile = self.__reader.read_region(
            location=(x, y), level=level, size=(allowed_w, allowed_h)
        )
        tile = np.array(tile)[..., :3]  # only rgb channels
        # Pad tile.
        return F._pad_tile(tile, shape=(h, w))
