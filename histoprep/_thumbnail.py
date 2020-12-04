import os
import itertools
import multiprocessing as mp
from functools import partial
from typing import Tuple, List

from openslide import OpenSlide
import numpy as np
from PIL import Image
from tqdm import tqdm


def available_downsamples(slide_path: str, return_dict: bool = False) -> dict:
    reader = OpenSlide(slide_path)
    downsamples = [round(x) for x in reader.level_downsamples]
    dims = [x for x in reader.level_dimensions]
    return dict(zip(downsamples, dims))

def get_thumbnail(
        slide_path: str,
        downsample: int = None,
        generate: bool = False
) -> Image.Image:
    """Return thumbnail by using openslide or by creating a new one."""
    if generate:
        thumbnail = generate_thumbnail(slide_path, downsample)
    reader = OpenSlide(slide_path)
    level_downsamples = [round(x) for x in reader.level_downsamples]
    if downsample is None:
        downsample = max(level_downsamples)
    if downsample in level_downsamples:
        level = level_downsamples.index(downsample)
        dims = reader.level_dimensions[level]
        thumbnail = reader.get_thumbnail(dims)
    else:
        thumbnail = generate_thumbnail(slide_path, downsample)
    return thumbnail


def generate_thumbnail(
        slide_path: str,
        downsample: int,
        width: int = 4096
) -> Image.Image:
    """Generate thumbnail for a slide."""
    with OpenSlide(slide_path) as r:
        dims = r.dimensions
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
    i, (x, y) = coords
    out_shape = (int(width/downscale), int(width/downscale))
    with OpenSlide(slide_path) as r:
        tile = r.read_region((x, y), 0, (width, width))
        tile = tile.resize(out_shape).convert('RGB')
    return i, tile
