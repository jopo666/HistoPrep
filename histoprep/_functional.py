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

from .preprocess.functional import tissue_mask


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
    # Save reader as global for multiprocessing
    global __READER__
    __READER__ = OpenSlide(slide_path)
    dims = __READER__.dimensions
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
    coords: Tuple[int, Tuple[int, int]]
):
    # Load slide from global.
    reader = __READER__
    i, (x, y) = coords
    out_shape = (int(width/downscale), int(width/downscale))
    tile = reader.read_region((x, y), 0, (width, width)).convert('RGB')
    tile = cv2.resize(np.array(tile), out_shape, cv2.INTER_LANCZOS4)
    return i, tile


def resize(image: Union[np.ndarray, Image.Image], max_pixels: int = 1_000_000):
    """Donwsaple image until it has less than max_pixels pixels."""
    if isinstance(image, Image.Image):
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = np.array(image, dtype=np.uint8)
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
    thresholds: List[int] = [5, 10, 15,
                             20, 30, 40, 50, 60, 80, 100, 120],
    max_pixels=1_000_000
) -> Image.Image:
    """Returns a summary image of different thresholds."""
    thumbnail = resize(thumbnail)
    gray = cv2.cvtColor(np.array(thumbnail), cv2.COLOR_RGB2GRAY)
    images = [gray]
    for t in thresholds:
        mask = tissue_mask(thumbnail, t)
        # Flip for a nicer image
        mask = 1 - mask
        mask = mask/mask.max()*255
        images.append(mask.astype(np.uint8))
    images = [images[i:i + 4] for i in range(0, len(images), 4)]
    rows = []
    for row in images:
        while len(row) != 4:
            row.append(np.ones(row[0].shape)*255)
        rows.append(np.hstack(row))
    summary = Image.fromarray(np.vstack(rows).astype('uint8'))
    l = ['original'] + thresholds
    print('Saturation thresholds:\n')
    for row in [l[i:i + 4] for i in range(0, len(l), 4)]:
        [print(str(x).center(8), end='') for x in row]
        print()
    return resize(summary, max_pixels)


#################################
### Functions for Dearrayer() ###
#################################

def detect_spots(
        image: Union[np.ndarray, Image.Image],
        mask: np.ndarray,
        min_area: float = 0.1,
        max_area: float = 3,
        kernel_size: Tuple[int] = (5, 5)
):
    """ Detect TMA spots from image.

    How: Detect tissue mask -> clean up non-TMA stuff -> return mask

    Arguments:
        image: thumbnail
        min_area: min_area = median_spot_area * min_area
        max_area: max_area = median_spot_area * max_area
        kernel_size: Sometimes the default doesn't work for large/small
            thumbnails
    """
    if isinstance(image, Image.Image):
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = np.array(image, dtype=np.uint8)
    elif isinstance(image, np.ndarray):
        image = image.astype(np.uint8)
    else:
        raise TypeError('Excpected {} or {} not {}.'.format(
            np.ndarray, Image.Image, type(image)
        ))
    # Structuring element to close gaps.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=10)
    # Erode & dilate trick.
    mask = cv2.erode(mask, np.ones(kernel_size, np.uint8), iterations=5)
    mask = cv2.dilate(mask, np.ones(kernel_size, np.uint8), iterations=10)
    # Remove too small/large spots.
    contours, __ = cv2.findContours(
        mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    areas = np.array([cv2.contourArea(x) for x in contours])
    # Define min and max values
    min_area, max_area = np.median(areas)*0.1, np.median(areas)*5
    idx = (areas > min_area) & (areas < max_area)
    contours = [contours[i] for i in range(len(idx)) if idx[i]]
    # Draw new mask.
    new_mask = np.zeros(mask.shape, dtype="uint8")
    for i, cnt in enumerate(contours):
        cv2.drawContours(new_mask, [contours[i]], -1, 1, -1)
    return new_mask


def get_spots(
    image: Union[np.ndarray, Image.Image],
    spot_mask: np.ndarray,
    downsample: int,
):
    """Orders the spots on a spot_mask taking into consideration empty spots."""
    if isinstance(image, Image.Image):
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = np.array(image, dtype=np.uint8)
    elif isinstance(image, np.ndarray):
        image = image.astype(np.uint8)
    else:
        raise TypeError('Excpected {} or {} not {}.'.format(
            np.ndarray, Image.Image, type(image)
        ))
    # Collect bounding boxes
    contours, _ = cv2.findContours(
        spot_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(cnt) for cnt in contours]
    boxes = np.array(boxes) * downsample
    # Get centroid of each contour.
    coords = np.zeros(boxes[:,:2].shape)
    for i,cnt in enumerate(contours):
        # compute the center of the contour
        M = cv2.moments(cnt)
        centroid_x = int(M["m10"] / M["m00"])
        centroid_y = int(M["m01"] / M["m00"])
        coords[i,:] = [centroid_x,centroid_y]
    # Detect rotation and rotate coordinates.
    theta = get_theta(image)
    coords = rotate_coordinates(coords, theta)
    # Find optimal columns and rows.
    cols = get_optimal_n(coords[:, 0])
    rows = get_optimal_n(coords[:, 1])
    # Cluster.
    col_labels = hierachial_clustering(coords[:, 0], n_clusters=cols)
    row_labels = hierachial_clustering(coords[:, 1], n_clusters=rows)
    # Order so that top-left spot is first.
    idx = np.lexsort((-coords[:, 0], coords[:, 1]))
    coords = coords[idx, :]
    boxes = boxes[idx, :]
    col_labels = col_labels[idx] + 1000
    row_labels = row_labels[idx] + 1000
    # Rename clusters.
    for labs in [col_labels, row_labels]:
        __, idx = np.unique(labs, return_index=True)
        idx = np.sort(idx)
        for cluster, i in enumerate(idx):
            labs[labs == labs[i]] = cluster
    # Collect spot numbers
    labels = list(zip(col_labels, row_labels))
    numbers = []
    i = 1
    for row in range(rows):
        for col in range(cols):
            labs = [x for x in labels if x ==  (col, row)]
            if len(labs) == 1:
                numbers.append(i)
            elif len(labs) > 1:
                ii = 1
                for z in labs:
                    numbers.append(f'{str(i)} _{str(ii)}')
                    ii += 1
            i += 1
    return numbers, boxes


def get_theta(image: Union[np.ndarray, Image.Image]) -> float:
    """Detect if image is rotated and return the angle."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 0, 255, apertureSize=3)
    theta = cv2.HoughLines(edges, 1, np.pi/180, 200)[0][0][1]
    return theta


def rotate_coordinates(coords: np.ndarray, theta: float) -> np.ndarray:
    """Rotate coordinates with given theta."""
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    return coords @ R


def get_optimal_n(X: np.ndarray, min_n: int = 2, max_n: int = 50) -> int:
    """Find optimal cluster size for dataset X."""
    if len(X.shape) < 2:
        X = X.reshape(-1, 1)
    sil = []
    for n in range(2, 50):
        clust = AgglomerativeClustering(n_clusters=n, linkage='ward')
        clust.fit(X)
        sil.append(silhouette_score(X, clust.labels_))
    return np.argmax(sil)+min_n


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
