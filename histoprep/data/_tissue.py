__all__ = ["TissueMask"]

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from histoprep.functional import downsample_xywh


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
            Tissue mask.
        """
        x_d, y_d, w_d, h_d = downsample_xywh(xywh, self.level_downsample)
        tile_mask = self.mask[y_d : y_d + h_d, x_d : x_d + w_d]
        if shape is not None:
            return cv2.resize(
                src=tile_mask, dsize=shape, interpolation=cv2.INTER_NEAREST
            )
        return tile_mask
