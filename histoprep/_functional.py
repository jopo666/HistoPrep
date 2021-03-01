import os
import itertools
import multiprocessing as mp
from functools import partial
from typing import Tuple, List, Callable, Union

import cv2
from openslide import OpenSlide
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from .preprocess.functional import tissue_mask, PIL_to_array
from ._czi_reader import OpenSlideCzi


########################
### Common functions ###
########################


def get_downsamples(slide_path: str) -> dict:
    reader = OpenSlide(slide_path)
    downsamples = [round(x) for x in reader.level_downsamples]
    dims = [x for x in reader.level_dimensions]
    return dict(zip(downsamples, dims))


def get_thumbnail(
    slide_path: str,
    downsample: int = None,
    create_thumbnail: bool = False
) -> Image.Image:
    """Return thumbnail by using openslide or by creating a new one."""
    if slide_path.endswith('czi'):
        if create_thumbnail:
            return generate_thumbnail(slide_path, downsample)
        else:
            return None
    else:
        reader = OpenSlide(slide_path)
        level_downsamples = [round(x) for x in reader.level_downsamples]
        if downsample is None:
            downsample = max(level_downsamples)
        if downsample in level_downsamples:
            level = level_downsamples.index(downsample)
            dims = reader.level_dimensions[level]
            thumbnail = reader.get_thumbnail(dims)
        elif create_thumbnail:
            thumbnail = generate_thumbnail(slide_path, downsample)
        else:
            thumbnail = None
        return thumbnail


def generate_thumbnail(
    slide_path: str,
    downsample: int,
    width: int = 4096
) -> Image.Image:
    """Generate thumbnail for a slide."""
    if slide_path.endswith('czi'):
        reader = OpenSlideCzi(slide_path)
        thumbnail = reader.get_thumbnail(downsample, 2048, fast=True)
        return thumbnail
    else:
        reader = OpenSlide(slide_path)
        dims = reader.dimensions
        blocks = (
            int(dims[0]/width) + 1,
            int(dims[1]/width) + 1
        )
        x = (i*width for i in range(blocks[0]))
        y = (i*width for i in range(blocks[1]))
        coords = list(enumerate(itertools.product(x, y)))
        # Multiprocessing to make things speedier.
        with mp.Pool(processes=os.cpu_count()) as p:
            func = partial(load_tile, slide_path, width, downsample)
            tiles = []
            for result in tqdm(
                p.imap(func, coords),
                total=len(coords),
                desc='Generating thumbnail',
                bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
            ):
                tiles.append(result)
        # Collect each column seperately and mash them together.
        all_columns = []
        col = []
        for i, tile in tiles:
            if i % blocks[1] == 0 and i != 0:
                all_columns.append(np.vstack(col))
                col = []
            col.append(np.array(tile))
        thumbnail = np.hstack(all_columns)
        # Turn into into pillow image.
        thumbnail = Image.fromarray(thumbnail.astype(np.uint8))
        return thumbnail


def load_tile(
        slide_path: str,
        width: int,
        downscale: int,
        coords: Tuple[int, Tuple[int, int]]):
    # Load slide from global.
    reader = OpenSlide(slide_path)
    i, (x, y) = coords
    out_shape = (int(width/downscale), int(width/downscale))
    try:
        tile = reader.read_region((x, y), 0, (width, width)).convert('RGB')
        tile = cv2.resize(np.array(tile), out_shape, cv2.INTER_LANCZOS4)
    except:
        tile = np.zeros(out_shape).astype(np.uint8)
    return i, tile


def resize(image: Union[np.ndarray, Image.Image], max_pixels: int = 1_000_000):
    """Donwsaple image until it has less than max_pixels pixels."""
    if isinstance(image, Image.Image):
        image = PIL_to_array(image)
    elif isinstance(image, np.ndarray):
        image = image.astype(np.uint8)
    else:
        raise TypeError('Excpected {} or {} not {}.'.format(
            np.ndarray, Image.Image, type(image)
        ))
    dimensions = image.shape[:2]
    width, height = dimensions
    factor = 0
    while width*height > max_pixels:
        factor += 1
        width = int(dimensions[0]/2**factor)
        height = int(dimensions[1]/2**factor)
    image = Image.fromarray(image)
    return image.resize((height, width))


def try_thresholds(
    thumbnail: Image.Image,
    thresholds: List[int],
    max_pixels=1_000_000
) -> Image.Image:
    """Returns a summary image of different thresholds."""
    thumbnail = resize(thumbnail)
    gray = cv2.cvtColor(np.array(thumbnail), cv2.COLOR_RGB2GRAY)
    images = [gray]
    for t in thresholds:
        mask = tissue_mask(thumbnail, t)
        # Flip for a nicer image.
        mask = 1 - mask
        mask = mask*255
        images.append(mask.astype(np.uint8))
    images = [images[i:i + 4] for i in range(0, len(images), 4)]
    rows = []
    for row in images:
        while len(row) != 4:
            row.append(np.ones(row[0].shape)*255)
        rows.append(np.hstack(row))
    summary = Image.fromarray(np.vstack(rows).astype('uint8'))
    l = ['original'] + thresholds
    print('Thresholds:\n')
    for row in [l[i:i + 4] for i in range(0, len(l), 4)]:
        [print(str(x).center(8), end='') for x in row]
        print()
    return resize(summary, max_pixels)


#################################
### Functions for Dearrayer() ###
#################################

def detect_spots(
        mask: np.ndarray,
        min_area_multiplier: float,
        max_area_multiplier: float,
        kernel_size: Tuple[int, int],):
    """ Detect TMA spots from a thumbnail image.

    How: Detect tissue mask -> clean up non-TMA stuff -> return mask.

    Arguments:
        min_area_multiplier: 
            median_spot_area * min_area_multiplier
        max_area_multiplier: 
            median_spot_area * max_area_multiplier
        kernel_size: 
            Sometimes the default doesn't work for large/small thumbnails.
    """
    # Opening to remove small shit.
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_size, iterations=3)
    # Dilate.
    mask = cv2.dilate(mask, np.ones(kernel_size, np.uint8), iterations=3)
    # Remove too small/large spots.
    contours, __ = cv2.findContours(
        mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    areas = np.array([cv2.contourArea(x) for x in contours])
    # Define min and max values.
    min_area = np.median(areas)*min_area_multiplier
    if max_area_multiplier is not None:
        max_area = np.mean(areas)*max_area_multiplier
    else:
        max_area = np.max(areas)
    idx = (areas >= min_area) & (areas <= max_area)
    contours = [contours[i] for i in range(len(idx)) if idx[i]]
    # Draw new mask.
    new_mask = np.zeros(mask.shape, dtype="uint8")
    for i, cnt in enumerate(contours):
        cv2.drawContours(new_mask, [contours[i]], -1, 1, -1)
    return new_mask


def get_spots(spot_mask: np.ndarray, downsample: int):
    """Orders the spots on a spot_mask taking into consideration empty spots."""
    # Collect bounding boxes.
    contours, _ = cv2.findContours(
        spot_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None, None
    # The format is (x,y,w,h)
    boxes = [cv2.boundingRect(cnt) for cnt in contours]
    boxes = np.array(boxes) * downsample
    # Get centroid of each bounding box.
    coords = np.zeros(boxes[:, :2].shape)
    for i, cnt in enumerate(contours):
        M = cv2.moments(cnt)
        centroid_x = int(M["m10"] / M["m00"])
        centroid_y = int(M["m01"] / M["m00"])
        coords[i, :] = [centroid_x, centroid_y]
    # Rotate coordinates.
    theta = get_theta(coords)
    coords = rotate_coordinates(coords, theta)
    # Detect optimal number of rows and columns.
    cols = get_optimal_cluster_size(coords[:, 0])
    rows = get_optimal_cluster_size(coords[:, 1])
    # Cluster.
    col_labels = hierachial_clustering(coords[:, 0], n_clusters=cols)
    row_labels = hierachial_clustering(coords[:, 1], n_clusters=rows)
    # Detect cluster means.
    x_means = [coords[col_labels == i, 0].mean() for i in range(cols)]
    y_means = [coords[row_labels == i, 1].mean() for i in range(rows)]
    # Change label numbers to correct order (starting from top-left).
    for i in range(cols):
        new_label = np.arange(cols)[np.argsort(x_means) == i]
        col_labels[col_labels == i] = -new_label
    col_labels *= -1
    for i in range(rows):
        new_label = np.arange(rows)[np.argsort(y_means) == i]
        row_labels[row_labels == i] = -new_label
    row_labels *= -1
    labels = list(zip(col_labels, row_labels))
    # Collect numbers.
    numbers = np.zeros(len(coords)).astype(np.str)
    i = 1
    for r in range(rows):
        for c in range(cols):
            idx = [x == (c, r) for x in labels]
            if sum(idx) == 1:
                numbers[idx] = i
            elif sum(idx) > 1:
                numbers[idx] = [f'{i}_{ii}' for ii in range(sum(idx))]
            i += 1
    return numbers, boxes


def get_theta(coords: np.ndarray) -> float:
    """Detect rotation from centroid coordinates and return angle in radians."""
    n = len(coords)
    thetas = []
    for r in range(n):
        for c in range(n):
            x1, y1 = coords[r, :]
            x2, y2 = coords[c, :]
            thetas.append(np.rad2deg(np.arctan2(y2 - y1, x2 - x1)))
    # We want deviations from 0 so divide corrections.
    corr = np.array([0, 45, 90, 135, 180])
    for i, theta in enumerate(thetas):
        sign = np.sign(theta)
        idx = np.abs(np.abs(theta)-corr).argmin()
        thetas[i] = theta-sign*corr[idx]
    # Finally return most common angle
    values, counts = np.unique(np.round(thetas), return_counts=True)
    theta = values[counts.argmax()]
    return np.radians(theta)


def rotate_coordinates(coords: np.ndarray, theta: float) -> np.ndarray:
    """Rotate coordinates with given theta."""
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    return coords @ R


def get_optimal_cluster_size(X: np.ndarray) -> int:
    """Find optimal cluster size for dataset X."""
    if len(X.shape) < 2:
        X = X.reshape(-1, 1)
    sil = []
    for n in range(2, X.shape[0]):
        clust = AgglomerativeClustering(n_clusters=n, linkage='ward')
        clust.fit(X)
        sil.append(silhouette_score(X, clust.labels_))
    return np.argmax(sil)+2


def hierachial_clustering(
        X: np.ndarray,
        n_clusters: int,
        linkage: str = 'ward'
) -> np.ndarray:
    if len(X.shape) < 2:
        X = X.reshape(-1, 1)
    clust = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    clust.fit(X)
    return clust.labels_
