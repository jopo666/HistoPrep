"""Stain normalization methods."""

from __future__ import annotations

__all__ = ["MachenkoStainNormalizer", "VahadaneStainNormalizer"]

from collections.abc import Callable
from functools import partial

import numpy as np
from sklearn.decomposition import DictionaryLearning

import histoprep.functional as F

ERROR_GRAYSCALE = "Stain matrix is not defined for grayscale images."
# These are from: https://github.com/mitkovetta/staining-normalization
REFERENCE_STAIN_MATRIX = np.array([[0.5626, 0.7201, 0.4062], [0.2159, 0.8012, 0.5581]])
REFERENCE_MAX_CONCENTRATIONS = np.array([[1.9705, 1.0308]])


class StainNormalizer:
    """Base class for stain normalizers."""

    def __init__(self, stain_matrix_fn: Callable, **kwargs) -> None:
        self.__stain_matrix_fn = partial(stain_matrix_fn, **kwargs)
        self.__normalize_fn = _normalize_stains

    def fit(self, image: np.ndarray, tissue_mask: np.ndarray | None = None) -> None:
        """Fit stain normalizer with a target image.

        Args:
            image: Target image.
            tissue_mask: Tissue mask, ignored if empty. Defaults to None.
        """
        image = _check_and_copy_image(image)
        stain_matrix = self.get_stain_matrix(image=image, tissue_mask=tissue_mask)
        concentrations = _get_stain_consentrations(image, stain_matrix)
        max_concentrations = np.percentile(concentrations, 99, axis=0).reshape((1, 2))
        self.__normalize_fn = partial(
            _normalize_stains,
            target_stain_matrix=stain_matrix,
            target_max_concentrations=max_concentrations,
        )

    def normalize(
        self, image: np.ndarray, tissue_mask: np.ndarray | None = None
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
        image = _check_and_copy_image(image)
        stain_matrix = self.__stain_matrix_fn(image=image, tissue_mask=tissue_mask)
        return self.__normalize_fn(image=image, stain_matrix=stain_matrix)

    def get_stain_matrix(
        self, image: np.ndarray, tissue_mask: np.ndarray | None = None
    ) -> np.ndarray:
        """Calculate a stain matrix.

        Args:
            image: Input image.
            tissue_mask: Tissue mask, ignored if empty. Defaults to None.

        Returns:
            Stain matrix.
        """
        image = _check_and_copy_image(image)
        return self.__stain_matrix_fn(image=image, tissue_mask=tissue_mask)

    def get_stain_concentrations(
        self, image: np.ndarray, tissue_mask: np.ndarray | None = None
    ) -> np.ndarray:
        """Calculate haematoxylin and eosin stain concentrations for each pixel.

        Args:
            image: Input image.
            tissue_mask: Tissue mask, ignored if empty. Defaults to None.

        Returns:
            Stain concentrations.
        """
        image = _check_and_copy_image(image)
        return _get_stain_consentrations(
            image,
            stain_matrix=self.__stain_matrix_fn(image=image, tissue_mask=tissue_mask),
        )

    def adjust_stain_concentrations(
        self,
        image: np.ndarray,
        haematoxylin_magnitude: float = 1.0,
        eosin_magnitude: float = 1.0,
        tissue_mask: np.ndarray | None = None,
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
        image = _check_and_copy_image(image)
        return _adjust_stain_concentrations(
            image=image,
            haematoxylin_magnitude=haematoxylin_magnitude,
            eosin_magnitude=eosin_magnitude,
            stain_matrix=self.__stain_matrix_fn(image=image, tissue_mask=tissue_mask),
        )

    def separate_stains(
        self, image: np.ndarray, tissue_mask: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Separate haematoxylin and eosin stains.

        Args:
            image: Input imag
            tissue_mask: Tissue mask, ignored if empty. Defaults to None.

        Returns:
            Haematoxylin and eosin stain images.
        """
        image = _check_and_copy_image(image)
        return _separate_stains(
            image=image,
            stain_matrix=self.__stain_matrix_fn(image=image, tissue_mask=tissue_mask),
        )


class MachenkoStainNormalizer(StainNormalizer):
    """Stain normalizer based on the Macenko method."""

    def __init__(self, angular_percentile: float = 0.99) -> None:
        """Initialize `MachenkoStainNormalizer` instance.

        Args:
            angular_percentile:  Hyperparameter. Defaults to 0.99.
        """
        super().__init__(
            stain_matrix_fn=_get_macenko_stain_matrix,
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
            stain_matrix_fn=_get_vahadane_stain_matrix,
            alpha=alpha,
            max_iter=max_iter,
        )


def _adjust_stain_concentrations(
    image: np.ndarray,
    stain_matrix: np.ndarray,
    haematoxylin_magnitude: float = 1.0,
    eosin_magnitude: float = 1.0,
) -> np.ndarray:
    """Adjust stain magnitudes."""
    stain_concentrations = _get_stain_consentrations(image, stain_matrix)
    stain_concentrations[:, 0] *= haematoxylin_magnitude
    stain_concentrations[:, 1] *= eosin_magnitude
    return (
        np.clip(255 * np.exp(-1 * np.dot(stain_concentrations, stain_matrix)), 0, 255)
        .reshape(image.shape)
        .astype(np.uint8)
    )


def _separate_stains(
    image: np.ndarray, stain_matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Separate haematoxylin and eosin stains."""
    stain_concentrations = _get_stain_consentrations(image, stain_matrix)
    output = []
    for stain_idx in range(2):
        tmp = stain_concentrations.copy()
        # Set other stain to zero
        tmp[:, stain_idx] = 0.0
        output.append(
            np.clip(255 * np.exp(-1 * np.dot(tmp, stain_matrix)), 0, 255)
            .reshape(image.shape)
            .astype(np.uint8)
        )
    return tuple(output[::-1])


def _get_macenko_stain_matrix(
    image: np.ndarray,
    tissue_mask: np.ndarray | None = None,
    angular_percentile: float = 0.99,
) -> np.ndarray:
    """Estimate stain matrix with the Macenko method."""
    optical_density = _rgb_to_optical_density(image).reshape((-1, 3))
    # Mask background.
    if tissue_mask is not None and tissue_mask.sum() > 0:
        optical_density = optical_density[tissue_mask.flatten() == 1, :]
    # Get eigenvectors of the covariance (symmetric).
    __, eigen_vecs = np.linalg.eigh(np.cov(optical_density, rowvar=False))
    # Select the two principle eigenvectors.
    eigen_vecs = eigen_vecs[:, [2, 1]]
    # Point the vectors to the same direction.
    if eigen_vecs[0, 0] < 0:
        eigen_vecs[:, 0] *= -1
    if eigen_vecs[0, 1] < 0:
        eigen_vecs[:, 1] *= -1
    # Project.
    proj = np.dot(optical_density, eigen_vecs)
    # Get angular coordinates and the min/max angles.
    phi = np.arctan2(proj[:, 1], proj[:, 0])
    min_angle = np.percentile(phi, 100 * (1 - angular_percentile))
    max_angle = np.percentile(phi, 100 * angular_percentile)
    # Select the two principle colors.
    col_1 = np.dot(eigen_vecs, np.array([np.cos(min_angle), np.sin(min_angle)]))
    col_2 = np.dot(eigen_vecs, np.array([np.cos(max_angle), np.sin(max_angle)]))
    # Make sure the order is Haematoxylin & Eosin.
    if col_1[0] > col_2[0]:
        stain_matrix = np.array([col_1, col_2])
    else:
        stain_matrix = np.array([col_2, col_1])
    # Normalize and return.
    return stain_matrix / np.linalg.norm(stain_matrix, axis=1)[:, None]


def _get_vahadane_stain_matrix(
    image: np.ndarray,
    tissue_mask: np.ndarray | None = None,
    alpha: float = 0.1,
    max_iter: int = 3,
) -> np.ndarray:
    """Estimate stain matrix with the Vahadane method."""
    optical_density = _rgb_to_optical_density(image).reshape((-1, 3))
    # Mask background.
    if tissue_mask is not None and tissue_mask.sum() > 0:
        optical_density = optical_density[tissue_mask.flatten() == 1, :]
    # Start learning.
    dict_learn = DictionaryLearning(
        n_components=2,
        alpha=alpha,
        transform_algorithm="lasso_lars",
        positive_dict=True,
        verbose=False,
        max_iter=max_iter,
    )
    dictionary = dict_learn.fit_transform(X=optical_density.T).T
    # Order haematoxylin first.
    if dictionary[0, 0] < dictionary[1, 0]:
        dictionary = dictionary[[1, 0], :]
    return dictionary / np.linalg.norm(dictionary, axis=1)[:, None]


def _normalize_stains(
    image: np.ndarray,
    stain_matrix: np.ndarray,
    *,
    target_stain_matrix: np.ndarray = REFERENCE_STAIN_MATRIX,
    target_max_concentrations: np.ndarray = REFERENCE_MAX_CONCENTRATIONS,
) -> dict[str, np.ndarray]:
    """Normalize image stains to match destination stain matrix."""
    src_stain_concentrations = _get_stain_consentrations(image, stain_matrix)
    src_max_concentrations = np.percentile(
        src_stain_concentrations, 99, axis=0
    ).reshape((1, 2))
    src_stain_concentrations *= target_max_concentrations / src_max_concentrations
    output = 255 * np.exp(-1 * np.dot(src_stain_concentrations, target_stain_matrix))
    return np.clip(output, 0, 255).reshape(image.shape).astype(np.uint8)


def _get_stain_consentrations(
    image: np.ndarray, stain_matrix: np.ndarray
) -> np.ndarray:
    """Collect stain concentrations."""
    optical_density = _rgb_to_optical_density(image).reshape((-1, 3))
    return np.linalg.lstsq(stain_matrix.T, optical_density.T, rcond=-1)[0].T


def _rgb_to_optical_density(image: np.ndarray) -> np.ndarray:
    image[image == 0] = 1  # taking a log.
    return np.maximum(-1 * np.log(image / 255), 1e-6)


def _check_and_copy_image(image: np.ndarray) -> np.ndarray:
    F._check_image(image)
    image = image.copy()
    if image.ndim == 2:  # noqa.
        raise ValueError(ERROR_GRAYSCALE)
    return image
