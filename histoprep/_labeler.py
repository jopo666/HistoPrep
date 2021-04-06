import os
import warnings
from os.path import join, exists
from typing import List, Union

import cv2
import numpy as np
import pandas as pd
import openslide
from PIL import Image, ImageDraw
from shapely.geometry import Polygon, MultiPolygon

from ._functional import resize
from .helpers._utils import load_pickle


class TileLabeler():
    """Add labels for tiles.

        Args:
            data_dir (str): Data directory of the slide (created 
                during cutting). 

        Raises:
            IOError: `data_dir`, `metadata.csv` or the thumbnail image
                doesn't exist.
            ValueError: More than one width found from the metadata.
        """

    def __init__(
        self,
        data_dir: str,
    ):
        super().__init__()
        self.data_dir = data_dir
        self._meta_path = join(data_dir, 'metadata.csv')
        self._thumb_path = None
        # Check paths.
        if not exists(data_dir):
            raise IOError(f'{self._meta_path} does not exist!')
        if not exists(self._meta_path):
            raise IOError(f'{self._meta_path} does not exist!')
        for f in os.scandir(data_dir):
            if 'thumbnail_' in f.name and not 'annot' in f.name:
                self._thumb_path = f.path
        if self._thumb_path is None:
            raise IOError(f'Thumbnail was not found!')
        # Get metadata.
        self.metadata = pd.read_csv(self._meta_path)
        # Get necessary parameters for plotting.
        self.width = self.metadata['width'].unique()
        if self.width.size == 1:
            self.width = int(self.width)
        else:
            raise ValueError(
                'More than one width found from the metadata '
                f'({self.width.tolist()}).'
            )
        self.downsample = int(self._thumb_path.split('_')[-1].split('.')[0])
        # Load thumbnail.
        self._thumbnail = Image.open(self._thumb_path)
        self._annotated_thumbnail = False

    def _drop_labels(self, prefix):
        cols = self.metadata.columns.tolist()
        drop = [x for x in cols if prefix in x]
        return self.metadata.drop(columns=drop)

    def get_thumbnail(self, max_pixels=1_000_000) -> Image.Image:
        return resize(self._thumbnail, max_pixels)

    def get_annotated_thumbnail(self, max_pixels=1_000_000) -> Image.Image:
        if not self._annotated_thumbnail:
            print(
                "You haven't created any labels yet!"
            )
        else:
            return resize(self._annotated_thumbnail, max_pixels)

    def _annotate(self, mask: Union[Polygon, MultiPolygon]):
        """Draw the shapely mask to the thumbnail."""
        # Init mask image.
        overlay = Image.new('RGBA', self._thumbnail.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        if isinstance(mask, Polygon):
            mask = [mask]
        for polygon in mask:
            coords = []
            for x, y in polygon.exterior.coords:
                x = int(x/self.downsample)
                y = int(y/self.downsample)
                coords.append((x, y))
            draw.polygon(coords, fill=(255, 0, 0, 180), outline="blue")
        self._annotated_thumbnail = Image.alpha_composite(
            self._thumbnail.convert('RGBA'),
            overlay.convert('RGBA')
        ).convert('RGB')

    def label_from_shapely(
        self,
        mask: Union[Polygon, MultiPolygon],
        prefix: str,
        threshold: float,
        overwrite: bool = False,
    ) -> None:
        """Add labels and intersection percentages to metadata from a 
        shapely mask.

        Args:
            mask (Union[Polygon, MultiPolygon]): Annotation mask.
            prefix (str): Name for the label.
            threshold (float): How much overlap with mask is required to
                turn the label into 1.
            overwrite (bool, optional): Wheter to overwrite labels if 
                they already exist in the metadata. Defaults to False.

        Raises:
            ValueError: Mask is not in Shapely format.
        """
        if not isinstance(mask, Polygon) and not isinstance(mask, MultiPolygon):
            raise ValueError('Excpected {} or {} not {}.'.format(
                Polygon, MultiPolygon, type(mask)
            ))
        if (
            any(f'{prefix}_label' in x for x in self.metadata.columns)
            and not overwrite
        ):
            print(
                'This dataset has already been labeled with '
                f'prefix={prefix}! To overwrite, set overwrite=True.'
            )
        # Drop previous labels if found.
        self.metadata = self._drop_labels(prefix)
        if len(mask.bounds) == 0:
            # "Draw" a thumbnail.
            self._annotated_thumbnail = self._thumbnail.copy()
            # Empty mask so all are 0.
            rows = []
            for x in range(self.metadata.shape[0]):
                rows.append({
                    f'{prefix}_perc': 0.0,
                    f'{prefix}_label': 0,
                })
        else:
            # Draw thumbnail.
            self._annotate(mask)
            # Collect labels.
            coords = np.vstack((
                self.metadata.x.to_numpy(),
                self.metadata.y.to_numpy(),
                self.metadata.width.to_numpy()
            )).T
            # Shapely uses (minx, miny, maxx, maxy).
            rows = []
            for coord in coords:
                x, y, width = coord
                tile = Polygon([
                    (x, y), (x, y+width),  # lower corners
                    (x+width, y+width), (x+width, y)  # upper corners
                ])
                percentage = tile.intersection(mask).area/tile.area
                rows.append({
                    f'{prefix}_perc': percentage,
                    f'{prefix}_label': int(percentage > threshold)
                })
        # Concatenate to metadata.
        self.metadata = pd.concat([self.metadata, pd.DataFrame(rows)], axis=1)
        # Save thumbnail.
        path = join(self.data_dir, f'{prefix}_mask.jpeg')
        self._annotated_thumbnail.save(path)
        # Save new metadata.
        self.metadata.to_csv(self._meta_path, index=False)

    def __repr__(self):
        return self.__class__.__name__ + '()'
