from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from aicspylibczi import CziFile
from PIL import Image

from histoprep.functional._level import format_level
from histoprep.functional._tiles import _divide_xywh, _get_allowed_dimensions, _pad_tile

try:
    import openslide

    OPENSLIDE_ERROR = None
    HAS_OPENSLIDE = True
except ImportError as error:
    openslide = None
    OPENSLIDE_ERROR = error
    HAS_OPENSLIDE = False


Image.MAX_IMAGE_PIXELS = 20_000 * 20_000
ERROR_OPENSLIDE_IMPORT = (
    "Could not import `openslide-python`, make sure `OpenSlide` is installed "
    "(https://openslide.org/api/python/)."
)
ERROR_NON_MOSAIC = "HistoPrep does not support reading non-mosaic czi-files."
BACKGROUND_COLOR = (1.0, 1.0, 1.0)


class SlideReaderBackend(ABC):
    """Base class for all backends."""

    def __init__(self, path: Union[str, Path]) -> None:
        if not isinstance(path, Path):
            path = Path(path)
        if not path.exists():
            raise FileNotFoundError(str(path.resolve()))
        self.__path = path if isinstance(path, Path) else Path(path)
        self.__name = self.__path.name.removesuffix(self.__path.suffix)

    @property
    def path(self) -> str:
        """Full slide filepath."""
        return str(self.__path.resolve())

    @property
    def name(self) -> str:
        """Slide filename without an extension."""
        return self.__name

    @property
    def suffix(self) -> str:
        """Slide file-extension."""
        return self.__path.suffix

    @property
    @abstractmethod
    def reader(self):  # noqa
        pass

    @property
    @abstractmethod
    def data_bounds(self) -> tuple[int, int, int, int]:
        """Data bounds defined by `xywh`-coordinates at `level=0`."""

    @property
    @abstractmethod
    def dimensions(self) -> tuple[int, int]:
        """Image dimensions (height, width) at `level=0`."""

    @property
    @abstractmethod
    def level_count(self) -> int:
        """Number of slide levels."""

    @property
    @abstractmethod
    def level_dimensions(self) -> dict[int, tuple[int, int]]:
        """Image dimensions (height, width) for each level."""

    @property
    @abstractmethod
    def level_downsamples(self) -> dict[int, tuple[float, float]]:
        """Image downsample factors (height, width) for each level."""

    @abstractmethod
    def read_level(self, level: int) -> np.ndarray:
        """Read full level data.

        Args:
            level: Slide pyramid `.

        Raises:
            ValueError: Invalid level argument.

        Returns:
            Array containing image data from `level`.
        """

    @abstractmethod
    def read_region(self, xywh: tuple[int, int, int, int], level: int) -> np.ndarray:
        """Read region based on `xywh`-coordinates.

        Args:
            xywh: Coordinates for the region.
            level: Slide level to read from. Defaults to 0.

        Raises:
            ValueError: Invalid level argument.

        Returns:
            Array containing image data from `xywh`-region.
        """


class CziBackend(SlideReaderBackend):
    """Slide reader using `aicspylibczi.CziFile` as a backend (by Allen Institute
    for Cell Science).
    """

    MIN_LEVEL_DIMENSION = 1024
    BACKEND_NAME = "CZI"

    def __init__(self, path: str) -> None:
        """Initialize CziBackend class instance.

        Args:
            path: Path to slide image.

        Raises:
            NotImplementedError: Image is a non-mosaic czi-file.
        """
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
        while lvl == 0 or max(slide_w, slide_h) // 2**lvl >= self.MIN_LEVEL_DIMENSION:
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
        level = format_level(level, available=list(self.level_dimensions))
        return self.read_region(xywh=self.data_bounds, level=level)

    def read_region(self, xywh: tuple[int, int, int, int], level: int) -> np.ndarray:
        level = format_level(level, available=list(self.level_dimensions))
        x, y, w, h = xywh
        # Define allowed dims, output dims and expected dims.
        allowed_h, allowed_w = _get_allowed_dimensions(xywh, dimensions=self.dimensions)
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
        return _pad_tile(
            cv2.cvtColor(tile, cv2.COLOR_BGR2RGB), shape=(excepted_h, excepted_w)
        )


class OpenSlideBackend(SlideReaderBackend):
    """Slide reader using `OpenSlide` as a backend."""

    BACKEND_NAME = "OPENSLIDE"

    def __init__(self, path: str) -> None:
        """Initialize OpenSlideBackend class instance.

        Args:
            path: Path to the slide image.

        Raises:
            ImportError: OpenSlide could not be imported.
        """
        if not HAS_OPENSLIDE:
            raise ImportError(ERROR_OPENSLIDE_IMPORT) from OPENSLIDE_ERROR
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
        level = format_level(level, available=list(self.level_dimensions))
        level_h, level_w = self.level_dimensions[level]
        return np.array(self.__reader.get_thumbnail(size=(level_w, level_h)))

    def read_region(self, xywh: tuple[int, int, int, int], level: int) -> np.ndarray:
        level = format_level(level, available=list(self.level_dimensions))
        # Only width and height have to be adjusted for the level.
        x, y, *__ = xywh
        *__, w, h = _divide_xywh(xywh, self.level_downsamples[level])
        # Read allowed region.
        allowed_h, allowed_w = _get_allowed_dimensions((x, y, w, h), self.dimensions)
        tile = self.__reader.read_region(
            location=(x, y), level=level, size=(allowed_w, allowed_h)
        )
        tile = np.array(tile)[..., :3]  # only rgb channels
        # Pad tile.
        return _pad_tile(tile, shape=(h, w))


class PillowBackend(SlideReaderBackend):
    """Slide reader using `Pillow` as a backend.

    NOTE: `Pillow` reads the the whole slide into memory and thus isn't suitable for
    large images.
    """

    MIN_LEVEL_DIMENSION = 512
    BACKEND_NAME = "PILLOW"

    def __init__(self, path: str) -> None:
        """Initialize PillowBackend class instance.

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
        while lvl == 0 or max(slide_w, slide_h) // 2**lvl >= self.MIN_LEVEL_DIMENSION:
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
        x, y, output_w, output_h = _divide_xywh(xywh, self.level_downsamples[level])
        allowed_h, allowed_w = _get_allowed_dimensions(
            xywh=(x, y, output_w, output_h), dimensions=self.level_dimensions[level]
        )
        tile = np.array(
            self.__pyramid[level].crop((x, y, x + allowed_w, y + allowed_h))
        )
        # Pad tile.
        return _pad_tile(tile, shape=(output_h, output_w))

    def __lazy_load(self, level: int) -> None:
        if level not in self.__pyramid:
            height, width = self.level_dimensions[level]
            self.__pyramid[level] = self.__pyramid[0].resize(
                (width, height), resample=Image.Resampling.NEAREST
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(path={self.path})"
