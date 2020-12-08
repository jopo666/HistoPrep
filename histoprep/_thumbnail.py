import os
import itertools
import multiprocessing as mp
from functools import partial
from typing import Tuple, List

import cv2
from openslide import OpenSlide
import numpy as np
from PIL import Image
from tqdm import tqdm


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
    tile = cv2.resize(np.array(tile),out_shape,cv2.INTER_NEAREST)
    return i, tile
