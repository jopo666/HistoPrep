__all__ = ["Properties"]

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from ._tiles import TileCoordinates

ERROR_PROPRERTY_MISMATCH = "Could not resume saving due to property mismatch ({})."


@dataclass(eq=True)
class Properties:
    """Data class for generating and comparing `propreties.json`, when saving tiles.

    Args:
        num_tiles: Number of extracted tiles.
        width: Tile width.
        height: Tile height.
        overlap: Overlap between neighbouring tiles.
        max_background: Maximum amount of background in each tile.
        tissue_threshold: Threshold used to obtain tissue mask.
        tissue_sigma: Sigma for gaussian blurring during tissue detection.
    """

    num_tiles: int
    width: int
    height: int
    overlap: float
    max_background: float
    tissue_threshold: int
    tissue_sigma: float

    @classmethod
    def from_tile_coordinates(cls, tiles: TileCoordinates) -> "Properties":
        return cls(
            num_tiles=tiles.num_tiles,
            width=tiles.width,
            height=tiles.height,
            overlap=tiles.overlap,
            max_background=tiles.max_background,
            tissue_threshold=tiles.tissue_threshold,
            tissue_sigma=tiles.tissue_sigma,
        )

    @classmethod
    def from_json(cls, path: Path) -> "Properties":
        with path.open("r") as f:
            kwargs = json.load(f)
        return cls(**kwargs)

    def to_json(self, path: Path) -> None:
        with path.open("w") as f:
            d = asdict(self)
            json.dump(d, f)

    def panic_if_not_equal(self, other: "Properties") -> None:
        if self == other:
            return
        # Collect different values.
        current = asdict(self)
        other = asdict(other)
        differences = []
        for key in current.keys():
            if current[key] != other[key]:
                differences.append(f"{key}: {current[key]}!={other[key]}")
        differences = ", ".join(differences)
        raise ValueError(ERROR_PROPRERTY_MISMATCH.format(differences))
