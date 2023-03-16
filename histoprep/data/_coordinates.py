from __future__ import annotations

__all__ = ["TileCoordinates", "SpotCoordinates"]

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class TileCoordinates:
    """Data class representing a collection of tile coordinates.

    Args:
        coordinates: List of XYWH-coordinates.
        width: Tile width.
        height: Tile height.
        overlap: Overlap between neighbouring tiles.
        max_background: Maximum amount of background in each tile.
        tissue_mask: Tissue mask used to filter tiles with background.
    """

    coordinates: list[tuple[int, int, int, int]]
    width: int
    height: int
    overlap: float
    max_background: float | None = field(default=None)
    tissue_mask: np.ndarray | None = field(default=None)

    def get_properties(self, level: int, level_downsample: tuple[float, float]) -> dict:
        """Generate dictonary of properties."""
        return {
            "num_tiles": len(self),
            "level": level,
            "level_downsample": level_downsample,
            "width": self.width,
            "height": self.height,
            "overlap": self.overlap,
            "max_background": self.max_background,
        }

    def __len__(self) -> int:
        return len(self.coordinates)

    def __iter__(self) -> Iterator[tuple[int, int, int, int]]:
        return iter(self.coordinates)

    def __getitem__(self, index: int) -> tuple[int, int, int, int]:
        return self.coordinates[index]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"num_tiles={len(self)}, "
            f"shape={(self.height,self.width)}, "
            f"overlap={self.overlap}, "
            f"max_background={self.max_background},"
        )


@dataclass(frozen=True)
class SpotCoordinates:
    """Data class representing a collection of TMA spot coordinates.

    Args:
        coordinates: List of XYWH-coordinates.
        spot_names: Spot numbers.
        spot_mask: Tissue mask used to dearray spots.
    """

    coordinates: tuple[int, int, int, int] = field(repr=False)
    spot_names: list[str] = field(repr=False)
    spot_mask: np.ndarray = field(repr=False)

    def __len__(self) -> int:
        return len(self.coordinates)

    def __iter__(self) -> Iterator[str, tuple[int, int, int, int]]:
        return iter(zip(self.spot_names, self.coordinates))

    def __getitem__(self, index: int) -> tuple[int, int, int, int]:
        return self.spot_names[index], self.coordinates[index]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_spots={len(self)})"
