__all__ = ["TissueMask"]

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from histoprep.functional import allowed_xywh, multiply_xywh, pad_tile


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
        self, xywh: tuple[int, int, int, int], shape: Optional[tuple[int, int]] = None
    ) -> np.ndarray:
        """Read region from tissue mask.

        Args:
            xywh: Region coordinates from `level=0`.
            shape: Output shape for the tile, ignored if None. Defaults to None.

        Returns:
            Tissue mask for thr region.
        """
        # Downsample xywh.
        xywh = multiply_xywh(xywh, self.level_downsample)
        # Read allowed region and pad.
        output_w, output_h = xywh[2:]
        
        x, y, w, h = allowed_xywh(xywh, self.mask.shape[:2])
        tile_mask = pad_tile(
            tile=self.mask[y : y + h, x : x + w], shape=(output_h, output_w), fill=0
        )
        # Reshape.
        if shape is not None:
            return cv2.resize(tile_mask, dsize=shape, interpolation=cv2.INTER_NEAREST)
        return tile_mask
