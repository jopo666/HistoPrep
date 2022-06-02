import logging
from typing import List, Tuple

import cv2
import numpy
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

__all__ = ["dearray"]


def dearray(
    tissue_mask: numpy.ndarray,
    kernel_size: int = 5,
    iterations: int = 3,
    min_area: float = 0.1,
    max_area: float = 3.0,
) -> Tuple[numpy.ndarray, Tuple[int, int, int, int], List[int]]:
    """Detects TMA spots from an image and gives each spot a number starting
    from the top-left corner.

    Args:
        tissue_mask: Tissue mask.
        kernel_size: Kernel size for dilate. Defaults to 5.
        iterations: Dilate iterations. Defaults to 3.
        min_area: Minimum area for a spot. Defaults to 0.1 and calculated with:
            `median(areas) * min_area`
        max_area_ Maximum area for a spot. Similar to `min_area`. Defaults to 3.0.
    """
    # Dilate.
    kernel = numpy.ones((kernel_size, kernel_size), numpy.uint8)
    spot_mask = cv2.dilate(tissue_mask, kernel, iterations=iterations)
    # Detect contours and get their areas.
    contours, __ = cv2.findContours(spot_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    areas = numpy.array([cv2.contourArea(cnt) for cnt in contours])
    # Define min and max values.
    min_area = numpy.median(areas) * min_area
    max_area = numpy.median(areas) * max_area
    # Initialize new mask.
    new_mask = numpy.zeros_like(spot_mask)
    bboxes = []
    centroids = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Select only contours that fit into area range.
        if not min_area < area < max_area:
            continue
        # Get bounding box.
        bbox = cv2.boundingRect(cnt)
        # Get centroid.
        moments = cv2.moments(cnt)
        centroid = (
            int(moments["m10"] / moments["m00"]),
            int(moments["m01"] / moments["m00"]),
        )
        # Draw to the new map.
        cv2.drawContours(new_mask, [cnt], -1, 1, -1)
        # Add bbox and centroid.
        bboxes.append(bbox)
        centroids.append(centroid)
    # Detect possible rotation of the image based on centroids.
    theta = get_theta(centroids)
    # Rotate centroids.
    centroids = rotate_coordinates(centroids, theta)
    # Detect optimal number of rows and columns.
    num_cols = get_optimal_cluster_size(centroids[:, 0])
    num_rows = get_optimal_cluster_size(centroids[:, 1])
    # Cluster each coordinate to column and row.
    col_labels = hierachial_clustering(centroids[:, 0], n_clusters=num_cols)
    row_labels = hierachial_clustering(centroids[:, 1], n_clusters=num_rows)
    # Detect cluster means.
    x_means = [centroids[col_labels == i, 0].mean() for i in range(num_cols)]
    y_means = [centroids[row_labels == i, 1].mean() for i in range(num_rows)]
    # Change label numbers to correct order (starting from top-left).
    for i in range(num_cols):
        new_label = numpy.arange(num_cols)[numpy.argsort(x_means) == i]
        col_labels[col_labels == i] = -new_label
    col_labels *= -1
    for i in range(num_rows):
        new_label = numpy.arange(num_rows)[numpy.argsort(y_means) == i]
        row_labels[row_labels == i] = -new_label
    row_labels *= -1
    # Collect numbers.
    numbers = numpy.zeros(len(centroids)).astype("str")
    i = 1
    complain = False
    for r in range(num_rows):
        for c in range(num_cols):
            idx = [x == (c, r) for x in zip(col_labels, row_labels)]
            if sum(idx) == 1:
                numbers[idx] = i
            elif sum(idx) > 1:
                complain = True
                numbers[idx] = [f"{i}_{ii}" for ii in range(sum(idx))]
            i += 1
    if complain:
        logging.warning("Some spots were assigned the same number.")
    # Return new mask, coords and numbers.
    return new_mask, bboxes, numbers


def get_theta(centroids: numpy.ndarray) -> float:
    """Detect rotation from centroid coordinates and return angle in radians."""
    # Calculate angle between each centroid.
    n = len(centroids)
    thetas = []
    for r in range(n):
        for c in range(n):
            x1, y1 = centroids[r]
            x2, y2 = centroids[c]
            thetas.append(numpy.rad2deg(numpy.arctan2(y2 - y1, x2 - x1)))
    # We want deviations from 0 so divide corrections.
    corrections = numpy.arange(0, 361, 45)
    for i, theta in enumerate(thetas):
        sign = numpy.sign(theta)
        idx = numpy.abs(numpy.abs(theta) - corrections).argmin()
        thetas[i] = theta - sign * corrections[idx]
    # Finally return most common angle.
    values, counts = numpy.unique(numpy.round(thetas), return_counts=True)
    theta = values[counts.argmax()]
    return numpy.radians(theta)


def rotate_coordinates(coords: numpy.ndarray, theta: float) -> numpy.ndarray:
    """Rotate coordinates with given theta."""
    c, s = numpy.cos(theta), numpy.sin(theta)
    R = numpy.array(((c, -s), (s, c)))
    return coords @ R


def get_optimal_cluster_size(X: numpy.ndarray) -> int:
    """Find optimal cluster size for dataset X."""
    if len(X.shape) < 2:
        X = X.reshape(-1, 1)
    sil = []
    for n in range(2, X.shape[0]):
        clust = AgglomerativeClustering(n_clusters=n, linkage="ward")
        clust.fit(X)
        sil.append(silhouette_score(X, clust.labels_))
    return numpy.argmax(sil) + 2


def hierachial_clustering(
    X: numpy.ndarray, n_clusters: int, linkage: str = "ward"
) -> numpy.ndarray:
    """Perform hierarchian clustering."""
    if len(X.shape) < 2:
        X = X.reshape(-1, 1)
    clust = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    clust.fit(X)
    return clust.labels_
