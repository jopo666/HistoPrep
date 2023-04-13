"""Stain normalization methods."""

__all__ = ["MachenkoStainNormalizer", "VahadaneStainNormalizer"]

from collections.abc import Callable
from functools import partial
from typing import Union

import numpy as np

from histoprep.functional._normalize import (
    adjust_stain_concentrations,
    check_and_copy_image,
    get_macenko_stain_matrix,
    get_stain_consentrations,
    get_vahadane_stain_matrix,
    normalize_stains,
    separate_stains,
)


class StainNormalizer:
    """Base class for stain normalizers."""

    def __init__(self, stain_matrix_fn: Callable, **kwargs) -> None:
        self.__stain_matrix_fn = partial(stain_matrix_fn, **kwargs)
        self.__normalize_fn = normalize_stains

    def fit(
        self, image: np.ndarray, tissue_mask: Union[np.ndarray, None] = None
    ) -> None:
        """Fit stain normalizer with a target image.

        Args:
            image: Target image.
            tissue_mask: Tissue mask, ignored if empty. Defaults to None.
        """
        image = check_and_copy_image(image)
        stain_matrix = self.get_stain_matrix(image=image, tissue_mask=tissue_mask)
        concentrations = get_stain_consentrations(image, stain_matrix)
        max_concentrations = np.percentile(concentrations, 99, axis=0).reshape((1, 2))
        self.__normalize_fn = partial(
            normalize_stains,
            target_stain_matrix=stain_matrix,
            target_max_concentrations=max_concentrations,
        )

    def normalize(
        self, image: np.ndarray, tissue_mask: Union[np.ndarray, None] = None
    ) -> np.ndarray:
        """Normalize image stains to match target image. If `fit` has not been called
        the reference stain matrix and concentrations are used (from link below).

            https://github.com/mitkovetta/staining-normalization

        Args:
            image: Input image.
            tissue_mask: Tissue mask, ignored if empty. Defaults to None.

        Returns:
            Stain normalized image.
        """
        image = check_and_copy_image(image)
        stain_matrix = self.__stain_matrix_fn(image=image, tissue_mask=tissue_mask)
        return self.__normalize_fn(image=image, stain_matrix=stain_matrix)

    def get_stain_matrix(
        self, image: np.ndarray, tissue_mask: Union[np.ndarray, None] = None
    ) -> np.ndarray:
        """Calculate a stain matrix.

        Args:
            image: Input image.
            tissue_mask: Tissue mask, ignored if empty. Defaults to None.

        Returns:
            Stain matrix.
        """
        image = check_and_copy_image(image)
        return self.__stain_matrix_fn(image=image, tissue_mask=tissue_mask)

    def get_stain_concentrations(
        self, image: np.ndarray, tissue_mask: Union[np.ndarray, None] = None
    ) -> np.ndarray:
        """Calculate haematoxylin and eosin stain concentrations for each pixel.

        Args:
            image: Input image.
            tissue_mask: Tissue mask, ignored if empty. Defaults to None.

        Returns:
            Stain concentrations.
        """
        image = check_and_copy_image(image)
        return get_stain_consentrations(
            image,
            stain_matrix=self.__stain_matrix_fn(image=image, tissue_mask=tissue_mask),
        )

    def adjust_stain_concentrations(
        self,
        image: np.ndarray,
        haematoxylin_magnitude: float = 1.0,
        eosin_magnitude: float = 1.0,
        tissue_mask: Union[np.ndarray, None] = None,
    ) -> np.ndarray:
        """Adjust stain magnitudes.

        Args:
            image: Input image.
            stain_matrix: Stain matrix.
            haematoxylin_magnitude: Multiplier for haematoxylin concentrations. Defaults
                to 1.0.
            eosin_magnitude: Multiplier for eosin concentrations. Defaults to 1.0.
            tissue_mask: Tissue mask, ignored if empty. Defaults to None.

        Returns:
            Stain adjusted image.
        """
        image = check_and_copy_image(image)
        return adjust_stain_concentrations(
            image=image,
            haematoxylin_magnitude=haematoxylin_magnitude,
            eosin_magnitude=eosin_magnitude,
            stain_matrix=self.__stain_matrix_fn(image=image, tissue_mask=tissue_mask),
        )

    def separate_stains(
        self, image: np.ndarray, tissue_mask: Union[np.ndarray, None] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Separate haematoxylin and eosin stains.

        Args:
            image: Input imag
            tissue_mask: Tissue mask, ignored if empty. Defaults to None.

        Returns:
            Haematoxylin and eosin stain images.
        """
        image = check_and_copy_image(image)
        return separate_stains(
            image=image,
            stain_matrix=self.__stain_matrix_fn(image=image, tissue_mask=tissue_mask),
        )


class MachenkoStainNormalizer(StainNormalizer):
    """Stain normalizer based on the Macenko method."""

    def __init__(self, angular_percentile: float = 0.99) -> None:
        """Initialize `MachenkoStainNormalizer` instance.

        Args:
            angular_percentile: Hyperparameter. Defaults to 0.99.
        """
        super().__init__(
            stain_matrix_fn=get_macenko_stain_matrix,
            angular_percentile=angular_percentile,
        )


class VahadaneStainNormalizer(StainNormalizer):
    """Stain normalizer based on the Vahadane method."""

    def __init__(self, alpha: float = 0.1, max_iter: int = 3) -> None:
        """Initialize `VahadaneStainNormalizer` instance.

        Args:
            alpha: Regulariser for lasso. Defaults to 0.1.
            max_iter: Maximum training iterations. Defaults to 3.
        """
        super().__init__(
            stain_matrix_fn=get_vahadane_stain_matrix,
            alpha=alpha,
            max_iter=max_iter,
        )
