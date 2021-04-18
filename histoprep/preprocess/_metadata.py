import os
from os.path import join, basename, dirname, exists
from typing import Union, List, Tuple
import warnings

from PIL import Image, ImageDraw
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


__all__ = [
    'combine_metadata',
    'update_paths',
]


def combine_metadata(
        parent_dir: str,
        csv_path: str = None,
        overwrite: bool = False,
        tma_spots=False) -> pd.DataFrame:
    """
    Combines all metadata in ``parent_dir`` into a single csv-file.

    Args:
        parent_dir (str): Directory with all the processed tiles.
        csv_path (str, optional): Path for the combined metadata.csv.
            Doesn't have to be defined if you just want to return the 
            pandas dataframe. Defaults to ``None``.
        overwrite (bool, optional): Whether to overwrite if csv_path 
            exists. Defaults to ``False``.
        tma_spots (bool, optional): Wheter to combine spot metadata.
            Defaults to ``False``.

    Raises:
        IOError: ``parent_dir`` does not exist.
        IOError: ``csv_path`` exist and ``overwrite=False``.

    Returns:
        pd.DataFrame: Combined metadata.
    """
    if not os.path.exists(parent_dir):
        raise IOError(f'{parent_dir} does not exist.')
    if csv_path is not None and os.path.exists(csv_path) and not overwrite:
        raise IOError(f'{csv_path} exists and overwrite=False.')
    dataframes = []
    directories = [x.path for x in os.scandir(parent_dir) if x.is_dir()]
    if tma_spots:
        meta_name = 'spot_metadata'
    else:
        meta_name = 'metadata'
    for directory in tqdm(
            directories,
            total=len(directories),
            desc='Combining metadata',
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
    ):
        metadata_path = os.path.join(directory, meta_name + '.csv')
        # There might be slides that haven't been finished.
        if not os.path.exists(metadata_path):
            warnings.warn(
                f'{metadata_path} path not found! This warning might arise '
                'if you are still cutting slides, some of the slides were '
                'broken or no tissue was found on the slide.'
            )
        # There might be empty files.
        elif os.path.getsize(metadata_path) > 5:
            dataframes.append(pd.read_csv(metadata_path))
    if len(dataframes) == 0:
        print('No metadata.csv files found!')
        return
    metadata = pd.concat(dataframes)
    if csv_path is not None:
        metadata.to_csv(csv_path, index=False)
    return metadata


def update_tile_paths(paths: list, data_dir: str, TMA: bool) -> list:
    new_paths = []
    for old in paths:
        filename = basename(old)
        if TMA:
            subdir = join(dirname(dirname(old)), dirname(old))
        else:
            subdir = 'tiles'
        new = join(data_dir, subdir, filename)
        new_paths.append(new)
    return new_paths


def update_spot_paths(paths: list, data_dir: str) -> list:
    new_paths = []
    for old in paths:
        filename = basename(old)
        new = join(data_dir, 'spots', filename)
        new_paths.append(new)
    return new_paths


def update_paths(parent_dir: str):
    """Rename all paths in metadata.

    This function can be used if you move around/rename the folder with 
    metadata etc. In this case all paths saved in metadata.csv will be
    wrong and have to be updated.

    Args:
        `parent_dir` (str): Directory to loop through

    Raises:
        IOError: ``parent_dir`` does not exists
    """
    if not os.path.exists(parent_dir):
        raise IOError(f'Path {parent_dir} does not exists!')
    # Collect data directories to update.
    update = []
    for f in os.scandir(parent_dir):
        if f.is_dir():
            update.append(f)
    if len(update) == 0:
        print(f'No directories found at {parent_dir}!')
        return
    for f in tqdm(update, desc='Updating paths'):
        # Collect metadata paths.
        meta_paths = []
        for x in os.scandir(f.path):
            if x.name.endswith('metadata.csv'):
                meta_paths.append(x.path)
        for path in meta_paths:
            meta = pd.read_csv(path)
            if 'spot_metadata' in path:
                # Spot metadata
                idx = ~meta.path.isna()
                spot_paths = meta.path[idx]
                new_paths = update_spot_paths(spot_paths, f.path)
                meta.loc[idx, 'path'] = new_paths
            elif 'spot_path' in meta.columns:
                # Tile metadata from a TMA array.
                # Update tile paths.
                idx = ~meta.path.isna()
                tile_paths = meta.path[idx]
                new_paths = update_tile_paths(tile_paths, f.path, TMA=True)
                meta.loc[idx, 'path'] = new_paths
                # Update spot paths.
                idx = ~meta.spot_path.isna()
                spot_paths = meta.spot_path[idx]
                new_paths = update_spot_paths(spot_paths, f.path)
                meta.loc[idx, 'spot_path'] = new_paths
            else:
                # Metadata from a normal slide.
                idx = ~meta.path.isna()
                tile_paths = meta.path[idx]
                new_paths = update_tile_paths(tile_paths, f.path, TMA=False)
                meta.loc[idx, 'path'] = new_paths
            # Save updated metadata.
            meta.to_csv(path, index=False)