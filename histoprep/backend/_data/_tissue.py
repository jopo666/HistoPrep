from __future__ import annotations

__all__ = ["TissueMask"]

from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

from histoprep.backend._functional import allowed_dimensions, divide_xywh, pad_tile


@dataclass(frozen=True)
class TissueMask:
    """Data class for tissue mask.

    Args:
        mask: Tissue mask array (0=background, 1=tissue).
        threshold: Threshold value used to generate tissue mask.
        sigma: Sigma for gaussian blurring during tissue detection.
        level: Slide level, which was used to generate tissue mask.
        level_downsample: Slide level downsample.

    """

    mask: np.ndarray = field(repr=False)
    threshold: int
    sigma: float
    level: int
    level_downsample: tuple[float, float]

    def to_pil(self) -> Image.Image:
        """Convert `mask` property to a pillow image."""
        return Image.fromarray(255 - self.mask * 255)

    def read_region(
        self, xywh: tuple[int, int, int, int], shape: tuple[int, int] | None = None
    ) -> np.ndarray:
        """Read region from tissue mask.

        Args:
            xywh: Region coordinates from `level=0`.
            shape: Output height and width for the tile, ignored if None. Defaults to
                None.

        Returns:
            Tissue mask for thr region.
        """
        # Downsample xywh.
        xywh_d = divide_xywh(xywh, self.level_downsample)
        # Read allowed region and pad.
        x, y, output_w, output_h = xywh_d
        allowed_h, allowed_w = allowed_dimensions(
            xywh_d, dimensions=self.mask.shape[:2]
        )
        tile_mask = pad_tile(
            tile=self.mask[y : y + allowed_h, x : x + allowed_w],
            shape=(output_h, output_w),
            fill=0,
        )
        # Reshape.
        if shape is not None:
            return cv2.resize(
                tile_mask, dsize=shape[::-1], interpolation=cv2.INTER_NEAREST
            )
        return tile_mask
