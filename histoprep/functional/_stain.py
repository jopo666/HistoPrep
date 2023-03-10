from __future__ import annotations

__all__ = [
    "normalize_stains",
    "adjust_stains",
    "separate_stains",
    "get_stain_consentrations",
    "get_macenko_stain_matrix",
    "get_vahadane_stain_matrix",
]


import numpy as np
from sklearn.decomposition import DictionaryLearning

from ._check import check_image

ERROR_GRAYSCALE = "Stain matrix is not defined for grayscale images (`image.ndim==2`)."


def normalize_stains(
    image: np.ndarray,
    src_stain_matrix: np.ndarray,
    dst_stain_matrix: np.ndarray,
    dst_stain_concentrations: np.ndarray,
) -> dict[str, np.ndarray]:
    """Normalize image stains to match destination stain matrix.

    Args:
        image: Image to be normalized.
        src_stain_matrix: Stain matrix for the input image.
        dst_stain_matrix: Stain matrix for the destination image.
        dst_concentrations: Stain concentrations of the destination image.

    Returns:
        Normalized image.
    """
    image = _check_and_copy_image(image)
    src_stain_concentrations = get_stain_consentrations(image, src_stain_matrix)
    dst_max_concentrations = np.percentile(
        dst_stain_concentrations, 99, axis=0
    ).reshape((1, 2))
    src_max_concentrations = np.percentile(
        src_stain_concentrations, 99, axis=0
    ).reshape((1, 2))
    src_stain_concentrations *= dst_max_concentrations / src_max_concentrations
    output = 255 * np.exp(-1 * np.dot(src_stain_concentrations, dst_stain_matrix))
    return np.clip(output, 0, 255).reshape(image.shape).astype(np.uint8)


def adjust_stains(
    image: np.ndarray,
    stain_matrix: np.ndarray,
    haematoxylin_magnitude: float = 1.0,
    eosin_magnitude: float = 1.0,
) -> np.ndarray:
    """Adjust stain magnitudes.

    Args:
        image: Input image.
        stain_matrix: Stain matrix.
        haematoxylin_magnitude: Multiplier for haematoxylin concentrations. Defaults to
            1.0.
        eosin_magnitude: Multiplier for eosin concentrations. Defaults to 1.0.

    Returns:
        Stain adjusted image.
    """
    image = _check_and_copy_image(image)
    stain_concentrations = get_stain_consentrations(image, stain_matrix)
    stain_concentrations[:, 0] *= haematoxylin_magnitude
    stain_concentrations[:, 1] *= eosin_magnitude
    return (
        np.clip(255 * np.exp(-1 * np.dot(stain_concentrations, stain_matrix)), 0, 255)
        .reshape(image.shape)
        .astype(np.uint8)
    )


def separate_stains(
    image: np.ndarray, stain_matrix: np.ndarray
) -> dict[str, np.ndarray]:
    """Separate haematoxylin and eosin stains based on a stain matrix.

    Args:
        image: Input image.
        stain_matrix: Stain matrix.

    Raises:
        ValueError: Non RGB image.

    Returns:
        Haematoxylin or eosin stain images.
    """
    image = _check_and_copy_image(image)
    stain_concentrations = get_stain_consentrations(image, stain_matrix)
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


def get_macenko_stain_matrix(
    image: np.ndarray,
    tissue_mask: np.ndarray | None = None,
    angular_percentile: float = 0.99,
) -> np.ndarray:
    """Estimate stain matrix with the Macenko method.

    Args:
        image: Input image.
        tissue_mask: Tissue mask, which is ignored if empty. Defaults to None.
        angular_percentile: Hyperparameter. Defaults to 0.99.

    Raises:
        ValueError: Non RGB image.

    Returns:
        Stain matrix [2 x 3].
    """
    image = _check_and_copy_image(image)
    # Extraxt optical density and mask background.
    optical_density = _rgb_to_optical_density(image).reshape((-1, 3))
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


def get_vahadane_stain_matrix(
    image: np.ndarray,
    tissue_mask: np.ndarray | None = None,
    alpha: float = 0.1,
    max_iter: int = 3,
) -> np.ndarray:
    """Estimate stain matrix with the Vahadane method.

    Args:
        image: Input image.
        tissue_mask: Tissue mask. Defaults to None.
        alpha: Regulariser for lasso. Defaults to 0.1.
        max_iter: Maximum training iterations. Defaults to 3.

    Raises:
        ValueError: Non RGB image.

    Returns:
        Stain matrix [2 x 3].
    """
    image = _check_and_copy_image(image)
    # Extraxt optical density and mask background.
    optical_density = _rgb_to_optical_density(image).reshape((-1, 3))
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


def get_stain_consentrations(image: np.ndarray, stain_matrix: np.ndarray) -> np.ndarray:
    image = _check_and_copy_image(image)
    optical_density = _rgb_to_optical_density(image).reshape((-1, 3))
    return np.linalg.lstsq(stain_matrix.T, optical_density.T, rcond=-1)[0].T


def _rgb_to_optical_density(image: np.ndarray) -> np.ndarray:
    image[image == 0] = 1  # taking a log.
    return np.maximum(-1 * np.log(image / 255), 1e-6)


def _check_and_copy_image(image: np.ndarray) -> np.ndarray:
    check_image(image)
    image = image.copy()
    if image.ndim == 2:  # noqa.
        raise ValueError(ERROR_GRAYSCALE)
    return image
