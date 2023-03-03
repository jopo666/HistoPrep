from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import numpy as np


class BaseBackend(ABC):
    """Base class for all slide-reader backends."""

    def __init__(self, path: Union[str, Path]) -> None:
        if not isinstance(path, Path):
            path = Path(path)
        if not path.exists():
            raise FileNotFoundError(str(path))
        self.path = path if isinstance(path, Path) else Path(path)
        self.name = self.path.name.rstrip(self.path.suffix)

    @property
    @abstractmethod
    def reader(self) -> tuple[int, int, int, int]:
        """xywh-coordinates at `level=0` defining the area containing data."""

    @property
    @abstractmethod
    def data_bounds(self) -> tuple[int, int, int, int]:
        """xywh-coordinates at `level=0` defining the area containing data."""

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
        """Read level to memory.

        Args:
            level: Image level to read.

        Raises:
            ValueError: Invalid level argument.

        Returns:
            Array containing image data for the level.
        """

    @abstractmethod
    def read_region(self, xywh: tuple[int, int, int, int], level: int) -> np.ndarray:
        """Read region based on xywh-coordinates.

        Args:
            xywh: Coordinates for the region.
            level: Slide level to read from.

        Raises:
            ValueError: Invalid level argument.

        Returns:
            Array containing image data from the region.
        """
