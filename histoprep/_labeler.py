import os
from typing import List

import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from openslide import OpenSlide


class Labeler():
    """Class for labeling tiles."""

    def __init__(self, metadata_path: str):
        super().__init__()
        self.metadata_path = metadata_path
        self.metadata = pd.read_csv(metadata_path)

    def _drop_labels(self, prefix):
        cols = self.metadata.columns.tolist()
        drop = [x for x in cols if prefix in x]
        return self.metadata.drop(columns=drop)

    def label_from_mask(
            self,
            mask_path: str,
            values: List[int],
            prefix: str,
            threshold: int,
            save_to_csv: bool = False
    ) -> pd.DataFrame:
        """
        Add labels and intersection percentages to metadata.

        Args:
            mask_path: Path to openslide-image mask.
            values: Values that equal label in mask.
            prefix: Name for the label.
            threshold: How much overlap with mask is required to turn the
                       label to 1.
        """
        r = OpenSlide(mask_path)
        # Drop previous labels if found.
        self.metadata = self._drop_labels(prefix)
        # Collect labels.
        rows = []
        coords = np.vstack((
            self.metadata.x.to_numpy(),
            self.metadata.y.to_numpy(),
            self.metadata.width.to_numpy()
        )).T
        for coord in coords:
            x, y, width = coord
            label_pixels = 0
            mask = np.array((r.read_region((x, y), 0, (width, width))))
            for val in values:
                label_pixels += mask[mask == val].size
            percentage = label_pixels/mask.size
            rows.append({
                f'{prefix}_perc': percentage,
                f'{prefix}_label': int(percentage > threshold)
            })
        # Concatenate to metadata.
        metadata = pd.concat([self.metadata, pd.DataFrame(rows)], axis=1)
        if save_to_csv:
            # Save to csv
            metadata.to_csv(self.metadata_path, index=False)
        return metadata

    def label_from_shapely(
            self,
            shapely_mask: Polygon,
            prefix: str,
            threshold: int,
            save_to_csv: bool = False
    ):
        """
        Add labels and intersection percentages to metadata.

        Args:
            shapely_mask: Mask of the annotations in shapely format.
            prefix: Name for the label.
            threshold: How much overlap with mask is required to turn the
                       label to 1.
        """
        # Drop previous labels if found.
        self.metadata = self._drop_labels(prefix)
        # Collect labels.
        coords = np.vstack((
            self.metadata.x.to_numpy(),
            self.metadata.y.to_numpy(),
            self.metadata.width.to_numpy()
        )).T
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
        metadata = pd.concat([self.metadata, pd.DataFrame(rows)], axis=1)
        if save_to_csv:
            # Save to csv
            metadata.to_csv(self.metadata_path, index=False)
        return metadata
        return self.metadata

    def __repr__(self):
        return self.__class__.__name__ + '()'
