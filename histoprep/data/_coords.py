__all__ = ["TileCoordinates", "TMASpotCoordinates"]

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

from PIL import Image

from ._tissue import TissueMask


@dataclass(frozen=True)
class TileCoordinates:
    """Data class representing a collection of tile coordinates.

    Args:
        num_tiles: Number of tiles.
        coordinates: List of XYWH-coordinates.
        width: Tile width.
        height: Tile height.
        overlap: Overlap between neighbouring tiles.
        max_background: Maximum amount of background in each tile.
        tissue_mask: Tissue mask used to filter tiles with background.
    """

    num_tiles: int
    coordinates: list[tuple[int, int, int, int]] = field(repr=False)
    width: int
    height: int
    overlap: float
    max_background: float
    tissue_mask: TissueMask = field(repr=False)
    thumbnail: Image.Image = field(repr=False)
    thumbnail_tiles: Image.Image = field(repr=False)
    thumbnail_tissue: Image.Image = field(repr=False)

    def save_thumbnails(self, output_dir: Union[str, Path]) -> None:
        """Save thumbnail images to output directory."""
        if not isinstance(output_dir, Path):
            output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        self.thumbnail.save(output_dir / "thumbnail.jpeg")
        self.thumbnail_tiles.save(output_dir / "thumbnail_tiles.jpeg")
        self.thumbnail_tissue.save(output_dir / "thumbnail_tissue.jpeg")

    def save_properties(
        self,
        output_dir: Union[str, Path],
        level: int,
        level_downsample: tuple[float, float],
    ) -> dict:
        """Save properties to output directory."""
        if not isinstance(output_dir, Path):
            output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        properties = {
            "num_tiles": self.num_tiles,
            "level": level,
            "level_downsample": level_downsample,
            "width": self.width,
            "height": self.height,
            "overlap": self.overlap,
            "max_background": self.max_background,
            "threshold": self.tissue_mask.threshold,
            "sigma": self.tissue_mask.sigma,
        }
        with (output_dir / "properties.json").open("w") as f:
            json.dump(properties, f)

    def __len__(self) -> int:
        return self.num_tiles

    def __iter__(self) -> Iterator[tuple[int, int, int, int]]:
        return iter(self.coordinates)

    def __getitem__(self, index: int) -> tuple[int, int, int, int]:
        return self.coordinates[index]


@dataclass(frozen=True)
class TMASpotCoordinates:
    """Data class representing a collection of TMA spot coordinates.

    Args:
        num_spots: Number of spots.
        coordinates: List of XYWH-coordinates.
        tissue_mask: Tissue mask used to dearray spots.
    """

    num_spots: int
    coordinates: tuple[int, int, int, int] = field(repr=False)
    names: list[str] = field(repr=False)
    tissue_mask: TissueMask
    thumbnail: Image.Image = field(repr=False)
    thumbnail_spots: Image.Image = field(repr=False)
    thumbnail_tissue: Image.Image = field(repr=False)

    def save_thumbnails(self, output_dir: Union[str, Path]) -> None:
        """Save thumbnail images to output directory."""
        if not isinstance(output_dir, Path):
            output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        self.thumbnail.save(output_dir / "thumbnail.jpeg")
        self.thumbnail_spots.save(output_dir / "thumbnail_spots.jpeg")
        self.thumbnail_tissue.save(output_dir / "thumbnail_tissue.jpeg")

    def save_properties(
        self,
        output_dir: Union[str, Path],
        level: int,
        level_downsample: tuple[float, float],
    ) -> dict:
        """Save properties to output directory."""
        if not isinstance(output_dir, Path):
            output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        properties = {
            "num_spots": self.num_spots,
            "level": level,
            "level_downsample": level_downsample,
            "threshold": self.tissue_mask.threshold,
            "sigma": self.tissue_mask.sigma,
        }
        with (output_dir / "properties.json").open("w") as f:
            json.dump(properties, f)

    def __len__(self) -> int:
        return self.num_spots

    def __iter__(self) -> Iterator[str, tuple[int, int, int, int]]:
        return iter(zip(self.names, self.coordinates))

    def __getitem__(self, index: int) -> tuple[int, int, int, int]:
        return self.names[index], self.coordinates[index]
