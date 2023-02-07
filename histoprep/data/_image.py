__all__ = ["TileImage"]

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class TileImage:
    """Data class representing a single tile image.

    Args:
        xywh: Tile coordinates (for level=0).
        level: Slide level for extracting coordinates.
        level_xywh: Tile coordinates at slide level.
        level_downsample: Slide level downsample.
        image: Image data for the xywh-region.
        tissue_mask: Tissue mask for the xywh-region.
        tissue_threshold: Threshold value used to generate tissue mask.
        tissue_sigma: Sigma for gaussian blurring during tissue detection.
        image_metrics: Image metrics calculated for `image`.
    """

    xywh: tuple[int, int, int, int]
    level: int
    level_xywh: tuple[int, int, int, int] = field(repr=False)
    level_downsample: tuple[float, float] = field(repr=False)
    image: np.ndarray = field(repr=False)
    tissue_mask: np.ndarray = field(repr=False)
    tissue_threshold: int
    tissue_sigma: float
    image_metrics: dict[str, float] = field(repr=False)

    def tile_to_pil(self) -> Image.Image:
        """Convert `image` property to a pillow image."""
        return Image.fromarray(self.image)

    def mask_to_pil(self) -> Image.Image:
        """Convert `mask` property to a pillow image."""
        return Image.fromarray(255 - self.tissue_mask * 255)

    def save_tile(
        self, output_dir: Union[str, Path], suffix: str = "jpeg", **writer_kwargs
    ) -> Path:
        """Save tile image to output directory.

        Args:
            output_dir: Output directory for the image.
            suffix: File suffix. Defaults to "jpeg".
            writer_kwargs: Passed to PIL image writer.
        """
        if not isinstance(output_dir, Path):
            output_dir = Path(output_dir)
        while suffix.startswith("."):
            suffix = suffix[1:]
        filepath = output_dir / "x{}_y{}_w{}_h{}.{}".format(*self.xywh, suffix)
        self.tile_to_pil().save(filepath, **writer_kwargs)
        return filepath

    def save_mask(self, output_dir: Union[str, Path], **writer_kwargs) -> Path:
        """Save tile mask to output directory.

        Args:
            output_dir: Output directory for the mask.
            writer_kwargs: Passed to PIL image writer.
        """
        if not isinstance(output_dir, Path):
            output_dir = Path(output_dir)
        filepath = output_dir / "x{}_y{}_w{}_h{}.png".format(*self.xywh)
        self.mask_to_pil().save(filepath, **writer_kwargs)
        return filepath
