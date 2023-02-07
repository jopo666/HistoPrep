__all__ = ["TileCoordinate", "TileCoordinates"]

from collections.abc import Iterator
from dataclasses import dataclass, field

from PIL import Image


@dataclass(eq=True, order=True)
class TileCoordinate:
    """Data class representing a single tile coordinate.

    Args:
        xywh: Tile coordinates (for level=0).
        level: Slide level for extracting coordinates.
        level_xywh: Tile coordinates at slide level.
        level_downsample: Slide level downsample.
        tissue_threshold: Threshold value used to generate tissue mask.
        tissue_sigma: Sigma for gaussian blurring during tissue detection.
        background_percentage: Background percentage for the tile.
    """

    xywh: tuple[int, int, int, int]
    level: int
    level_xywh: tuple[int, int, int, int] = field(repr=False)
    level_downsample: tuple[float, float] = field(repr=False)
    tissue_threshold: int = field(compare=False)
    tissue_sigma: float = field(compare=False)
    background_percentage: float = field(compare=False)


@dataclass(frozen=True)
class TileCoordinates:
    """Data class representing a collection of tile coordinates.

    Args:
        tile_coordinates: List of TileCoordinate instances.
        num_tiles: Length of the tile_coordinates list.
        width: Tile width.
        height: Tile height.
        overlap: Overlap between neighbouring tiles.
        max_background: Maximum amount of background in each tile.
        level: Slide level for extracting coordinates.
        level_xywh: Tile coordinates at slide level.
        level_downsample: Slide level downsample.
        tissue_threshold: Threshold value used to generate tissue mask.
        tissue_sigma: Sigma for gaussian blurring during tissue detection.
        thumbnail: Thumbnail image of the slide.
        thumbnail_tiles: Thumbnail image of the slide with tile annotations.
        thumbnail_tissue: Thumbnail image of the tissue mask.
    """

    tile_coordinates: list[TileCoordinate] = field(repr=False)
    num_tiles: int
    width: int
    height: int
    overlap: float
    max_background: float
    tissue_threshold: int
    tissue_sigma: float
    thumbnail: Image.Image = field(repr=False)
    thumbnail_tiles: Image.Image = field(repr=False)
    thumbnail_tissue: Image.Image = field(repr=False)

    def __len__(self) -> int:
        return len(self.tile_coordinates)

    def __iter__(self) -> Iterator[TileCoordinate]:
        return iter(self.tile_coordinates)

    def __getitem__(self, index: int) -> TileCoordinate:
        return self.tile_coordinates[index]
