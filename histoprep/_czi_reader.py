import os
import itertools
import time
import multiprocessing as mp
from functools import partial
from typing import Tuple, List

from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image, ImageDraw
from aicspylibczi import CziFile
from shapely.geometry import Polygon, MultiPolygon, mapping

__all__ = [
    'OpenSlideCzi'
]


class OpenSlideCzi(object):
    """
    Open a czi-file like one of your french OpenSlide readers.

    Args:
        slide_path (str): Path to the slide.

    Raises:
        NotImplementedError: czi-file is non-mosaic.
    """

    def __init__(self, slide_path: str):
        self.slide_path = slide_path
        reader = CziFile(slide_path)
        if not reader.is_mosaic():
            raise NotImplementedError(
                'This class has only been defined for mosaic czi files. '
                'You should be able to simply convert the non-mosaic czi-file '
                'into a format that can be opened with openslide.'
            )
        self.bboxes = np.array(reader.mosaic_scene_bounding_boxes())
        self.data_mask = self._get_data_mask(self.bboxes)
        self.shape = reader.read_mosaic_size()
        self.dimensions = self.shape[2:]
        self.region_mask = self._get_region_mask(self.shape)

    def _get_data_mask(self, bboxes):
        polys = []
        for x, y, w, h in bboxes:
            polys.append(
                Polygon([[x, y], [x+w, y], [x+w, y+h], [x, y+h], [x, y]]))
        return MultiPolygon(polys).buffer(0)

    def _get_region_mask(self, shape):
        x, y, w, h = shape
        mask = Polygon([[x, y], [x+w, y], [x+w, y+h], [x, y+h], [x, y]])
        return mask

    def _get_all_coordinates(self, bboxes, width, overlap=0.0):
        """Return all coordinates."""
        stop_x = bboxes[:, 0].max()
        stop_y = bboxes[:, 1].max()
        x = [bboxes[:, 0].min()]
        y = [bboxes[:, 1].min()]
        overlap_px = int(width * overlap)
        while x[-1] < stop_x:
            x.append(x[-1] + width - overlap_px)
        x = x[:-1]
        while y[-1] < stop_y:
            y.append(y[-1] + width - overlap_px)
        y = y[:-1]
        blocks = (len(x), len(y))
        coordinates = list(itertools.product(x, y))
        return coordinates, blocks

    def get_tiles_with_data(self, width, overlap):
        coords, __ = self._get_all_coordinates(self.bboxes, width, overlap)
        # Wrap function.
        func = partial(check_tile, **{
            'data_mask': self.data_mask,
            'region_mask': self.region_mask,
        })
        # Load tiles.
        filtered = []
        with mp.Pool(processes=os.cpu_count() - 1) as p:
            for result in tqdm(
                p.imap(func, coords),
                total=len(coords),
                desc='Filtering tiles',
                bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
            ):
                filtered.append(result)
        filtered = list(filter(None, filtered))
        return filtered

    def read_region(self, location, scale, size):
        """Return a PIL.Image containing the contents of the region.

        Args:
            location: (x, y) tuple giving the top left pixel in the level 0
                      reference frame.
            level:    the level number.
            size:     (width, height) tuple giving the region size.
        """
        # Fix coordinates to czi's shitty negative ones...
        x, y = location
        x = self.shape[0] + x
        y = self.shape[1] + y
        location = (0, (x, y))
        width = size[0] * (2 ** scale)
        __, tile = load_tile(
            coords=location,
            slide_path=self.slide_path,
            data_mask=self.data_mask,
            region_mask=self.region_mask,
            width=width,
            downsample=2**scale,
            fast=False
        )
        return Image.fromarray(tile.astype(np.uint8))

    def get_thumbnail(self, downsample: int, width: int = 4096,
                      fast: bool = False) -> Image.Image:
        """
        Generate a thumbnail image with given downsample.

        Args:
            downsample (int): Desired downsample for the thumbnail.
            width (int, optional): Width of each tile during construction.
                Defaults to 4096.
            fast (bool, optional): Use lower pyramid levels in the czi-file to 
                generate each tile. If the slide contains two or more scenes 
                then some parts of the thumbnail can be missing. It is fast as
                fuck though so idk... Defaults to False.

        Returns:
            [Image.Image]: Thumbnail image.
        """
        # Load ALL coordinates.
        coordinates, blocks = self._get_all_coordinates(self.bboxes, width)
        coordinates = list(enumerate(coordinates))
        # Wrap function.
        func = partial(load_tile, **{
            'slide_path': self.slide_path,
            'data_mask': self.data_mask,
            'region_mask': self.region_mask,
            'width': width,
            'downsample': downsample,
            'fast': fast,
        })
        # Load tiles.
        tiles = []
        with mp.Pool(processes=os.cpu_count() - 1) as p:
            for result in tqdm(
                p.imap(func, coordinates),
                total=len(coordinates),
                desc='Generating thumbnail',
                bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
            ):
                tiles.append(result)
        tiles.sort()
        # Check tiles...
        if len(tiles) == 0:
            print('No tiles found!')
            return None
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
        thumbnail = Image.fromarray(thumbnail.astype(np.uint8)).convert('RGB')
        return thumbnail


def polygon_to_mask(poly, x, y, width, downsample):
    """Turn polygon to tile mask."""
    mask = Image.new('L', (int(width/downsample), int(width/downsample)), 0)
    mapp = mapping(poly)
    if mapp['type'] == 'Polygon' or mapp['type'] == 'MultiPolygon':
        coords = mapp['coordinates']
    elif mapp['type'] == 'GeometryCollection':
        coords = []
        for key, val in mapp.items():
            if key == 'geometries':
                for d in val:
                    if d['type'] == 'Polygon':
                        coords.append(d['coordinates'])
    else:
        return mask
    if len(coords) > 1:
        coords = [item for sublist in coords for item in sublist[0]]
    else:
        coords = coords[0]
    if len(coords) > 2 and isinstance(coords[0], tuple):
        coords = [((c[0] - x)/downsample, (c[1] - y)/downsample)
                  for c in coords]
        ImageDraw.Draw(mask).polygon(coords, outline=1, fill=1)
    mask = np.array(mask)
    return mask


def check_tile(xy, data_mask, region_mask):
    tile = Polygon([
        [x, y],
        [x+width, y],
        [x+width, y+width],
        [x, y+width],
        [x, y]
    ])
    d_perc = tile.intersection(data_mask).area/tile.area
    r_perc = tile.intersection(region_mask).area/tile.area
    if d_perc == 1 and r_perc == 1:
        return xy
    else:
        return None


def load_tile(coords, slide_path, data_mask,
              region_mask, width, downsample, fast):
    """Load a tile from the czi-file (parallizeable)."""
    # Load reader.
    reader = CziFile(slide_path)
    # Unpack coords.
    i, (x, y) = coords
    # Define final out_shape.
    out_shape = (int(width/downsample), int(width/downsample))
    # Get data and region percentages.
    tile_mask = Polygon(
        [[x, y], [x+width, y], [x+width, y+width], [x, y+width], [x, y]])
    intersection = tile_mask.intersection(data_mask)
    data_perc = intersection.area/tile_mask.area
    region_perc = tile_mask.intersection(region_mask).area/tile_mask.area
    if data_perc < 0.05 or region_perc < 1:
        # Return empty image.
        tile = np.ones(out_shape + (3,)) * 255
    else:
        # Load tile.
        bbox = (x, y, width, width)
        if fast:
            tile = reader.read_mosaic(bbox, C=0, scale_factor=1/downsample)
        else:
            tile = reader.read_mosaic(bbox, C=0)
        if tile.shape[0] == 1:
            # Something is wrong with the tile...
            tile = np.ones(out_shape + (3,)) * 255
        else:
            tile = np.moveaxis(tile, 0, 2)
            tile = cv2.resize(tile, out_shape, cv2.INTER_LANCZOS4)
            if data_perc < 1:
                mask = polygon_to_mask(intersection, x, y, width, downsample)
                mask = cv2.resize(mask, out_shape, cv2.INTER_LANCZOS4)
                tile[mask == 0] = 255
    tile = tile.astype(np.uint8)
    return i, tile
