import sys
import os
from os.path import dirname, join, basename, exists
from typing import List, Tuple, Callable
import itertools
import multiprocessing as mp
from functools import partial
import warnings

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageDraw
from openslide import OpenSlide

from ._thumbnail import get_thumbnail
from .preprocess.functional import preprocess, tissue_mask
from ._helpers import remove_extension, remove_images


class Cutter(object):
    """Cut tiles from histological images.

    Parameters:
        slide_path:
            Path to the slide image. All formats that are supported by openslide
            can be used.
        width: 
            Width of the square tiles to be cut.
        overlap: 
            Proportion of overlap between neighboring tiles
        sat_thresh: 
            Saturation threshold for tissue detection. Can be left 
            undefined, in which case Otsu's binarization is used. This is not
            recommended! Values can easily be searched with 
            Cutter.try_thresholds() function.
        downsample: 
            Downsample used for the thumbnail. When a lower downsample is used, 
            the thumbnail-based background detection is more accurate but 
            slower. Good results are achieved with downsample=16.
        max_background:
            Maximum amount of background allowed for a tile. Due to the 
            thumbnail-based background detection, tiles with higher background
            percentage may pass through but rarely the other way.
        generate_thumbnail:
            Force thumbnail generation. This is sometimes required if the
            thumbnails are shitty.
    """

    def __init__(
        self,
        slide_path: str,
        width: int,
        overlap: float = 0.0,
        sat_thresh: int = None,
        downsample: int = 16,
        max_background: float = 0.999,
        generate_thumbnail: bool = False
    ):
        super().__init__()
        # Define openslide reader.
        if not exists(slide_path):
            raise IOError(f'{slide_path} not found.')
        self._reader = OpenSlide(slide_path)
        # Assing basic stuff that user can see/check.
        self.slide_path = slide_path
        self.slide_name = remove_extension(basename(slide_path))
        self.dimensions = self._reader.dimensions
        self.downsample = downsample
        self.width = width
        self.overlap = overlap
        self.sat_thresh = sat_thresh
        self.max_background = max_background
        self.all_coordinates = self._get_all_coordinates()
        # Filter coordinates.
        self._thumbnail = get_thumbnail(
            slide_path=self.slide_path,
            downsample=self.downsample,
            generate=generate_thumbnail
        )
        self.sat_thresh, self._tissue_mask = tissue_mask(
            image=self._thumbnail,
            sat_thresh=self.sat_thresh,
            return_threshold=True
        )
        self.filtered_coordinates = self._filter_coordinates()
        # Annotate thumbnail
        self._annotate()

    def __repr__(self):
        return self.__class__.__name__ + '()'

    def summary(self):
        print(self._summary())

    def _summary(self):
        return (
            f"{self.slide_name}"
            f"\n  Tile width: {self.width}"
            f"\n  Tile overlap: {self.overlap}"
            f"\n  Saturation threshold: {self.sat_thresh}"
            f"\n  Max background: {self.max_background}"
            f"\n  Total number of tiles: {len(self.all_coordinates)}"
            f"\n  After background filtering: {len(self.filtered_coordinates)}"
        )

    def plot_tiles(self, max_pixels=1_000_000) -> Image.Image:
        return self._resize(self._annotated_thumbnail, max_pixels)

    def plot_thumbnail(self, max_pixels=1_000_000) -> Image.Image:
        return self._resize(self._thumbnail, max_pixels)

    def plot_tissue_mask(self, max_pixels=1_000_000) -> Image.Image:
        mask = self._tissue_mask
        # Flip for a nicer image
        mask = 1 - mask
        mask = mask/mask.max()*255
        mask = Image.fromarray(mask.astype(np.uint8))
        return self._resize(mask, max_pixels)

    def _prepare_directories(self, parent_dir: str) -> None:
        out_dir = join(parent_dir, self.slide_name)
        # Save paths.
        self._meta_path = join(out_dir, 'metadata.csv')
        self._thumb_path = join(out_dir, 'thumbnail.jpeg')
        self._image_dir = join(out_dir, 'images')
        # Make dirs.
        os.makedirs(dirname(self._meta_path), exist_ok=True)
        os.makedirs(dirname(self._thumb_path), exist_ok=True)
        os.makedirs(self._image_dir, exist_ok=True)
        # Save summary.
        with open(join(out_dir, 'summary.txt'), "w") as f:
            f.write(self._summary())

    def _annotate(self) -> None:
        # Draw tiles to the thumbnail.
        self._annotated_thumbnail = self._thumbnail.copy()
        annotated = ImageDraw.Draw(self._annotated_thumbnail)
        w = h = int(self.width/self.downsample)
        for (x, y), __ in self.filtered_coordinates:
            x_d = round(x/self.downsample)
            y_d = round(y/self.downsample)
            annotated.rectangle([x_d, y_d, x_d+w, y_d+h],
                                outline='red', width=4)

    def try_thresholds(
            self,
            thresholds: List[int] = [5, 10, 15,
                                     20, 30, 40, 50, 60, 80, 100, 120]
    ) -> Image.Image:
        """Returns a summary image of different thresholds."""
        thumbnail = self._resize(self._thumbnail)
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
        return self._resize(summary)

    def cut(
            self,
            parent_dir: str,
            overwrite: bool = False,
            image_format: str = 'jpeg',
            quality: int = 95,
            custom_preprocess: Callable[[Image.Image], dict] = None
    ) -> pd.DataFrame:
        """Cut and save tile images and metadata.

        Arguments:
            parent_dir: save all information here
            overwrite: This will REMOVE all saved images,thumbnail and metadata
                and save images again.
            image_format: jpeg or png.
            quality: For jpeg saving.
            custom_preprocess: This is intended for users that want to define 
                their own preprocessing function. Function must take a Pillow
                image as an input and return a dictionary of desired metrics.
        """
        allowed_formats = ['jpeg', 'png']
        if image_format not in allowed_formats:
            raise ValueError(
                'Image format {} not allowed. Select from {}'.format(
                    image_format, allowed_formats
                ))
        self._prepare_directories(parent_dir)
        # Check if slide has been cut before.
        if exists(self._meta_path) and not overwrite:
            print(f'Slide has already been cut!')
            return pd.read_csv(self._meta_path)
        elif exists(self._meta_path) and overwrite:
            # Remove all previous files.
            os.remove(self._thumb_path)
            os.remove(self._meta_path)
            remove_images(self._image_dir)
        # Warn about Otsu's thresholding.
        if self.sat_thresh is None:
            warnings.warn(
                "Otsu's binarization will be used which might lead to errors "
                "in tissue detection (seriously it takes a few seconds to "
                "check for a good value with Cutter.try_thresholds() "
                "function...)"
            )
        # Save annotated thumbnail
        self._annotated_thumbnail.save(self._thumb_path, quality=95)
        # Wrap the saving function so it can be parallized.
        func = partial(save_tile, **{
            'slide_path': self.slide_path,
            'slide_name': self.slide_name,
            'image_dir': self._image_dir,
            'width': self.width,
            'sat_thresh': self.sat_thresh,
            'image_format': image_format,
            'quality': quality,
            'custom_preprocess': custom_preprocess,
        })
        # Multiprocessing to speed things up!
        metadata = []
        with mp.Pool(processes=os.cpu_count()) as p:
            for result in tqdm(
                p.imap(func, self.filtered_coordinates),
                total=len(self.filtered_coordinates),
                desc=self.slide_name,
                bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
            ):
                metadata.append(result)
        metadata = list(filter(None, metadata))
        if len(metadata) == 0:
            print(f'No tiles saved from slide {self.slide_path}!')
            return
        # Save metadata
        metadata = pd.DataFrame(metadata)
        metadata.to_csv(self._meta_path, index=False)
        return metadata

    def _get_all_coordinates(self):
        """Return tile coordinates over the whole slide."""
        x = [0]
        y = [0]
        overlap_px = int(self.width*self.overlap)
        while x[-1] < self.dimensions[0]:
            x.append(x[-1] + self.width - overlap_px)
        x = x[:-1]
        while y[-1] < self.dimensions[1]:
            y.append(y[-1] + self.width - overlap_px)
        y = y[:-1]
        coordinates = list(itertools.product(x, y))
        return coordinates

    def _filter_coordinates(self):
        """Filter out coordinates with too much background."""
        filtered = []
        width_d = np.ceil(self.width/self.downsample).astype(int)
        for x, y in self.all_coordinates:
            y_d = int(y/self.downsample)
            x_d = int(x/self.downsample)
            mask = self._tissue_mask[y_d:y_d+width_d, x_d:x_d+width_d]
            if mask.size == 0:
                continue
            bg_perc = 1 - mask.sum()/mask.size
            if bg_perc < self.max_background:
                filtered.append(((x, y), bg_perc))
        return filtered

    def _resize(self, image, MAX_PIXELS=5_000_000):
        dimensions = np.array(image).shape[:2]
        width, height = dimensions
        factor = 0
        while width*height > MAX_PIXELS:
            factor += 1
            width = int(dimensions[0]/2**factor)
            height = int(dimensions[1]/2**factor)
        image = Image.fromarray(np.array(image).astype('uint8'))
        return image.resize((height, width))

    def __len__(self):
        return len(self.filtered_coordinates)


def save_tile(
        coords: Tuple[int, int, float],
        slide_path: str,
        slide_name: str,
        image_dir: str,
        width: int,
        sat_thresh: int,
        image_format: str,
        quality: int,
        custom_preprocess: Callable[[Image.Image], dict] = None
) -> dict:
    """Saves tile and returns metadata (parallizable)."""
    # Load slide.
    reader = OpenSlide(slide_path)
    (x, y), bg_estimate = coords
    # Prepare filename.
    filepath = join(image_dir, f'{slide_name}_x-{x}_y-{y}')
    if image_format == 'png':
        filepath = filepath + '.png'
    else:
        filepath = filepath + '.jpeg'
    # Collect basic metadata.
    metadata = {
        'path': filepath,
        'slide_name': slide_name,
        'x': x,
        'y': y,
        'width': width,
        'background_estimate': bg_estimate
    }
    # Load image.
    try:
        image = reader.read_region((x, y), 0, (width, width)).convert('RGB')
    except:
        # Sometimes parts of the slide are corrupt or something...
        return
    # Update metadata with preprocessing metrics.
    if custom_preprocess is None:
        metadata.update(preprocess(image=image, sat_thresh=sat_thresh))
    else:
        metadata.update(custom_preprocess(image))
    # Save image.
    image.save(filepath, quality=quality)
    return metadata
