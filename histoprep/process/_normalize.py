__all__ = ["MachenkoStainNormalizer", "VahadaneStainNormalizer"]

from collections.abc import Callable
from functools import partial
from typing import Optional

import numpy as np

import histoprep.functional as F

ERROR_NO_TARGET = "Please call `fit()` before `normalize()`."


class StainNormalizer:
    """Base class for stain normalizers."""

    def __init__(self, stain_matrix_fn: Callable, **kwargs) -> None:
        self.__stain_matrix_fn = partial(stain_matrix_fn, **kwargs)
        self.__normalize_fn = None

    def fit(self, image: np.ndarray, tissue_mask: Optional[np.ndarray] = None) -> None:
        """Fit stain normalizer with a target image.

        Args:
            image: Target image.
            tissue_mask: Tissue mask, which is ignored if empty. Defaults to None.
        """
        stain_matrix = self.__stain_matrix_fn(image=image, tissue_mask=tissue_mask)
        stain_concentrations = F.get_stain_consentrations(
            image=image, stain_matrix=stain_matrix
        )
        self.__normalize_fn = partial(
            F.normalize_stains,
            dst_stain_matrix=stain_matrix,
            dst_stain_concentrations=stain_concentrations,
        )

    def normalize(
        self, image: np.ndarray, tissue_mask: Optional[np.ndarray] = None
    ) -> None:
        """Normalize image stains to match target image.

        Args:
            image: Input imag
            tissue_mask: Tissue mask, which is ignored if empty. Defaults to None.

        Returns:
            Stain normalized image.
        """
        if self.__normalize_fn is None:
            raise ValueError(ERROR_NO_TARGET)
        stain_matrix = self.__stain_matrix_fn(image=image, tissue_mask=tissue_mask)
        return self.__normalize_fn(image=image, src_stain_matrix=stain_matrix)


class MachenkoStainNormalizer(StainNormalizer):
    def __init__(self, angular_percentile: float = 0.99) -> None:
        """Stain normalizer based on the macenko method.

        Args:
            angular_percentile: Hyperparameter. Defaults to 0.99.
        """
        self.angular_percentile = angular_percentile
        super().__init__(
            stain_matrix_fn=F.get_macenko_stain_matrix,
            angular_percentile=angular_percentile,
        )


class VahadaneStainNormalizer(StainNormalizer):
    def __init__(self, alpha: float = 0.1, max_iter: int = 3) -> None:
        """Stain normalizer based on the vahadane method.

        Args:
            alpha: Regulariser for lasso. Defaults to 0.1.
            max_iter: Maximum training iterations. Defaults to 3.
        """
        self.alpha = alpha
        self.max_iter = max_iter
        super().__init__(
            stain_matrix_fn=F.get_vahadane_stain_matrix, alpha=alpha, max_iter=max_iter
        )
