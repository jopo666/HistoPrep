__all__ = ["dearray_tma"]

import warnings

import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from ._tissue import clean_tissue_mask


def dearray_tma(
    tissue_mask: np.ndarray, min_area: float = 0.2, max_area: float = 2.0
) -> dict[str, tuple[int, int, int, int]]:
    """Dearray tissue microarray -spots based on a tissue mask of the spots.

    Tissue mask can be obtained with `F.detect_tissue` or `SlideReader.detect_tissue()`
    functions and by increasing the sigma value removes most of the unwanted tissue
    fragements/artifacts. Rest can be handled with `min_area` and `max_area` arguments.

    Args:
        tissue_mask: Tissue mask of TMA-slide.
        min_area: Minimum contour area, defined by `median(contour_areas) * min_area`.
            Defaults to 0.2.
        max_area: Maximum contour area, defined by `median(contour_areas) * max_area`.
            Defaults to 2.0.

    Returns:
        `TMASpots` instance.
    """
    # Clean tissue mask.
    spot_mask = clean_tissue_mask(
        tissue_mask=tissue_mask, min_area=min_area, max_area=max_area
    )
    # Detect contours and get their bboxes and centroids.
    bboxes, centroids = contour_bboxes_and_centroids(spot_mask)
    # Detect possible rotation of the image based on centroids.
    centroids = rotate_coordinates(centroids, detect_rotation(centroids))
    # Detect optimal number of rows and columns and cluster each spot.
    num_cols = optimal_cluster_size(centroids[:, 0].reshape(-1, 1))
    num_rows = optimal_cluster_size(centroids[:, 1].reshape(-1, 1))
    col_labels = hierachial_clustering(
        centroids[:, 0].reshape(-1, 1), n_clusters=num_cols
    )
    row_labels = hierachial_clustering(
        centroids[:, 1].reshape(-1, 1), n_clusters=num_rows
    )
    # Change label numbers to correct order (starting from top-left).
    x_means = [centroids[col_labels == i, 0].mean() for i in range(num_cols)]
    y_means = [centroids[row_labels == i, 1].mean() for i in range(num_rows)]
    for i in range(num_cols):
        new_label = np.arange(num_cols)[np.argsort(x_means) == i]
        col_labels[col_labels == i] = -new_label
    col_labels *= -1
    for i in range(num_rows):
        new_label = np.arange(num_rows)[np.argsort(y_means) == i]
        row_labels[row_labels == i] = -new_label
    row_labels *= -1
    # Collect numbers.
    numbers = np.zeros(len(centroids)).astype("str")
    current_number = 1
    same_spot_number = False
    for r in range(num_rows):
        for c in range(num_cols):
            matches = [x == (c, r) for x in zip(col_labels, row_labels)]
            if sum(matches) == 1:
                numbers[matches] = str(current_number)
            elif sum(matches) > 1:
                same_spot_number = True
                numbers[matches] = [
                    f"{current_number}-{version+1}" for version in range(sum(matches))
                ]
            current_number += 1
    if same_spot_number:
        warnings.warn("Some spots were assigned the same number.")
    # Return bboxes and numbers.
    return {f"spot_{k}": tuple(v) for k, v in zip(numbers, bboxes)}


def contour_bboxes_and_centroids(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract contour bounding boxes and centroids."""
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    bboxes, centroids = [], []
    for cnt, is_parent in zip(contours, hierarchy[0][:, -1] == -1):
        # Skip non-parents and contours without area.
        if not is_parent or cv2.contourArea(cnt) == 0:
            continue
        # Get bounding box.
        bboxes.append(cv2.boundingRect(cnt))
        # Get centroid.
        moments = cv2.moments(cnt)
        centroids.append(
            (
                int(moments["m10"] / moments["m00"]),
                int(moments["m01"] / moments["m00"]),
            )
        )
    return np.array(bboxes), np.array(centroids)


def detect_rotation(centroids: np.ndarray) -> float:
    """Detect rotation from centroid coordinates and return angle in radians."""
    # Calculate angle between each centroid.
    n = len(centroids)
    thetas = []
    for r in range(n):
        for c in range(n):
            x1, y1 = centroids[r]
            x2, y2 = centroids[c]
            thetas.append(np.rad2deg(np.arctan2(y2 - y1, x2 - x1)))
    # We want deviations from 0 so divide corrections.
    corrections = np.arange(0, 361, 45)
    for i, theta in enumerate(thetas):
        sign = np.sign(theta)
        idx = np.abs(np.abs(theta) - corrections).argmin()
        thetas[i] = theta - sign * corrections[idx]
    # Finally return most common angle.
    values, counts = np.unique(np.round(thetas), return_counts=True)
    theta = values[counts.argmax()]
    return np.radians(theta)


def rotate_coordinates(coords: np.ndarray, theta: float) -> np.ndarray:
    """Rotate coordinates with given theta."""
    c, s = np.cos(theta), np.sin(theta)
    r_matrix = np.array(((c, -s), (s, c)))
    return coords @ r_matrix


def optimal_cluster_size(data: np.ndarray) -> int:
    """Find optimal cluster size for dataset X."""
    sil = []
    if data.shape[0] <= 2:
        return 1
    for n in range(2, data.shape[0]):
        labels = hierachial_clustering(data=data, n_clusters=n)
        sil.append(silhouette_score(data, labels))
    return np.argmax(sil) + 2


def hierachial_clustering(data: np.ndarray, n_clusters: int) -> np.ndarray:
    """Perform hierarchian clustering and get labels."""
    if n_clusters == 1:
        return np.zeros(data.shape[0], dtype=np.int32)
    clust = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    clust.fit(data)
    return clust.labels_
