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


from ._functional import get_downsamples, resize, get_thumbnail
from ._helpers import load_data


class TileLabeler():
    """Class for labeling tiles.

    Arguments:
        data_dir: 
            Directory that contains the output of Cutter.save() function.
        create_thumbnail:
            Create a thumbnail if downsample is not available.
    """

    def __init__(
        self,
        data_dir: str,
        create_thumbnail: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        # Check paths.
        self._meta_path = join(data_dir, 'metadata.csv')
        param_path = join(data_dir, 'parameters.p')
        if not exists(self._meta_path):
            raise IOError(f'{self._meta_path} does not exist!')
        if not exists(param_path):
            raise IOError(f'{param_path} does not exist!')
        # Get metadata and params.
        self.metadata = pd.read_csv(self._meta_path)
        self._params = load_data(param_path)
        self.slide_path = self._params['slide_path']
        self.downsample = self._params['downsample']
        self.width = self._params['width']
        self._thumbnail = get_thumbnail(
            slide_path=self.slide_path,
            downsample=self.downsample,
            create_thumbnail=create_thumbnail
        )
        if self._thumbnail is None:
            # Downsample not available.
            raise ValueError(
                f'Thumbnail not available for downsample {self.downsample}. '
                'Please set create_thumbnail=True or select downsample from\n\n'
                f'{self._downsamples()}'
            )
        self._annotated_thumbnail = False

    def _drop_labels(self, prefix):
        cols = self.metadata.columns.tolist()
        drop = [x for x in cols if prefix in x]
        return self.metadata.drop(columns=drop)

    def _downsamples(self):
        string = 'Downsample  Dimensions'
        d = get_downsamples(self.slide_path)
        for item, val in d.items():
            string += f'\n{str(item).ljust(12)}{val}'
        return string

    def plot_thumbnail(self, max_pixels=1_000_000) -> Image.Image:
        return resize(self._thumbnail, max_pixels)

    def plot_labels(self, max_pixels=1_000_000) -> Image.Image:
        if not self._annotated_thumbnail:
            print(
                "You haven't created any labels yet! Use the "
                "Labeler.label_from_shapely function to create these.")
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
            draw.polygon(coords, fill=(255, 0, 0, 150), outline="blue")
        self._annotated_thumbnail = Image.alpha_composite(
            self._thumbnail.convert('RGBA'),
            overlay.convert('RGBA')
        ).convert('RGB')

    def numpy_to_shapely(
        self,
        mask: np.ndarray,
        downsample: int = None
    ) -> MultiPolygon:
        """Convert a binary numpy mask to a shapely mask.

        Arguments:
            mask:
                Mask of the desired labels in a 2-dimensional numpy.ndarray.
            downsample:
                If the mask is a downsample of the real mask then give this to
                multiply the shapely mask coordinates.
        """
        if len(mask.shape) != 2:
            raise ValueError('Expected a 2-dimensional numpy mask.')
        contours, __ = cv2.findContours(
            mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        coords = []
        for cnt in contours:
            if downsample is not None:
                coords.append(np.squeeze(cnt) * downsample)
            else:
                coords.append(np.squeeze(cnt))
        polygons = [Polygon(x) for x in coords if x.shape[0] > 2]
        return MultiPolygon(polygons).buffer(0)

    def label_from_shapely(
        self,
        mask: Union[Polygon, MultiPolygon],
        prefix: str,
        threshold: int,
        overwrite: bool = False,
    ) -> None:
        """
        Add labels and intersection percentages to metadata from a shapely mask.

        Args:
            mask:
                Shapely.Polygon mask of the annotations.
            prefix: 
                Name for the label.
            threshold: 
                How much overlap with mask is required to turn the 
                label to 1.
            overwrite:
                Wheter to overwrite labels if they are already in the metadata.
        """
        if not isinstance(mask, Polygon) and not isinstance(mask, MultiPolygon):
            raise ValueError('Excpected {} or {} not {}.'.format(
                Polygon, MultiPolygon, type(mask)
            ))
        if (
            any(f'{prefix}_label' in x for x in self.metadata.columns)
            and not overwrite
        ):
            raise ValueError(
                f'This dataset has already been labeled with {prefix}! '
                'To overwrite set overwrite=True.'
            )
        # Drop previous labels if found.
        self.metadata = self._drop_labels(prefix)
        if len(mask.bounds) == 0:
            # "Draw thumbnail".
            self._annotated_thumbnail = self._thumbnail
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
