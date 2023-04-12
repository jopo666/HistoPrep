from __future__ import annotations

__all__ = ["MachenkoStainNormalizer", "VahadaneStainNormalizer"]

from collections.abc import Callable
from functools import partial

import numpy as np

import histoprep.functional as F


class StainNormalizer:
    """Base class for stain normalizers."""

    def __init__(self, stain_matrix_fn: Callable, **kwargs) -> None:
        self.__stain_matrix_fn = partial(stain_matrix_fn, **kwargs)
        self.__normalize_fn = F._normalize_stains

    def fit(self, image: np.ndarray, tissue_mask: np.ndarray | None = None) -> None:
        """Fit stain normalizer with a target image.

        Args:
            image: Target image.
            tissue_mask: Tissue mask, which is ignored if empty. Defaults to None.
        """
        stain_matrix = self.__stain_matrix_fn(image=image, tissue_mask=tissue_mask)
        concentrations = F._get_stain_consentrations(image, stain_matrix)
        max_concentrations = np.percentile(concentrations, 99, axis=0).reshape((1, 2))
        self.__normalize_fn = partial(
            F._normalize_stains,
            target_stain_matrix=stain_matrix,
            target_max_concentrations=max_concentrations,
        )

    def separate_stains(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Separate haematoxylin and eosin stains into separate images.

        Args:
            image: Input image.

        Returns:
            Haematoxylin and eosin stain images.
        """
        return F.separate_stains(
            image=image,
            stain_matrix=self.__stain_matrix_fn(image=image, tissue_mask=tissue_mask),
        )

    def normalize(
        self, image: np.ndarray, tissue_mask: np.ndarray | None = None
    ) -> None:
        """Normalize image stains to match target image.

        Args:
            image: Input imag
            tissue_mask: Tissue mask, which is ignored if empty. Defaults to None.

        Returns:
            Stain normalized image.
        """
        stain_matrix = self.__stain_matrix_fn(image=image, tissue_mask=tissue_mask)
        return self.__normalize_fn(image=image, stain_matrix=stain_matrix)


class MachenkoStainNormalizer(StainNormalizer):
    """Stain normalizer based on the macenko method."""

    def __init__(self, angular_percentile: float = 0.99) -> None:
        """Initialize `MachenkoStainNormalizer` instance.

        Args:
            angular_percentile:  Hyperparameter. Defaults to 0.99.
        """
        super().__init__(
            stain_matrix_fn=F.get_macenko_stain_matrix,
            angular_percentile=angular_percentile,
        )


class VahadaneStainNormalizer(StainNormalizer):
    """Stain normalizer based on the vahadane method."""

    def __init__(self, alpha: float = 0.1, max_iter: int = 3) -> None:
        """Initialize `VahadaneStainNormalizer` instance.

        Args:
            alpha: Regulariser for lasso. Defaults to 0.1.
            max_iter: Maximum training iterations. Defaults to 3.
        """
        super().__init__(
            stain_matrix_fn=F.get_vahadane_stain_matrix, alpha=alpha, max_iter=max_iter
        )
