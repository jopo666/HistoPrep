import os
from os.path import join, basename, dirname, exists
from typing import List
import logging
import multiprocessing as mp

import pandas as pd
from tqdm import tqdm


__all__ = [
    'combine_metadata',
    'update_paths',
]

# Define logger.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
    # Collect metadata paths.
    directories = [x.path for x in os.scandir(parent_dir) if x.is_dir()]
    if tma_spots:
        meta_name = 'spot_metadata'
    else:
        meta_name = 'metadata'
    meta_paths = [join(d, f'{meta_name}.csv') for d in directories]
    # Collect dataframes
    dataframes = multiprocessing_loop(
        func=get_metadata,
        loop_this=meta_paths,
        desc='Combining metadata',
        processes=1 if len(meta_paths) < 20 else None,
    )
    dataframes = list(filter(lambda x: x is not None, dataframes))
    if len(dataframes) == 0:
        logger.info(f'No metadata found at {parent_dir}!')
        return
    metadata = pd.concat(dataframes)
    if csv_path is not None:
        logger.info(f'Saving combined metadata to {csv_path}.')
        metadata.to_csv(csv_path, index=False)
    return metadata


def get_metadata(metadata_path):
    """Safely load metadata."""
    if not exists(metadata_path):
        # There might be slides that haven't been finished.
        logger.warn(
            f'{metadata_path} path not found! This warning might arise '
            'if you are still cutting slides, some of the slides were '
            'broken or no tissue was found on the slide.'
        )
        return None
    elif os.path.getsize(metadata_path) < 6:
        # There might be empty files.
        logger.info(f'{metadata_path} is an empty file.')
        return None
    else:
        return pd.read_csv(metadata_path)


def update_tile_paths(paths: list, data_dir: str, TMA: bool) -> list:
    new_paths = []
    for old in paths:
        filename = basename(old)
        if TMA:
            spot_dir = basename(dirname(old))
            subdir = join('tiles', spot_dir)
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
        if f.is_dir() and exists(join(f.path, 'metadata.csv')):
            update.append(f)
    if len(update) == 0:
        logger.info(f'No directories found at {parent_dir}!')
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


def multiprocessing_loop(func, loop_this, desc=None, processes=None):
    results = []
    if processes is None:
        processes = os.cpu_count() - 1
    with mp.Pool(processes=processes) as p:
        for result in tqdm(
            p.imap(func, loop_this),
            total=len(loop_this),
            desc=desc,
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
        ):
            results.append(result)
    return results


def check_tiles(metadata: pd.DataFrame):
    pass


def multiprocessing_loop(func, loop_this, desc=None, processes=None):
    results = []
    if processes is None:
        processes = os.cpu_count() - 1
    with mp.Pool(processes=processes) as p:
        for result in tqdm(
            p.imap(func, loop_this),
            total=len(loop_this),
            desc=desc,
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
        ):
            results.append(result)
    return results


def check_image(path):
    img = load_image(path)
    if img is None:
        logger.warn(f'Not a complete image: {path}')
        return path, (0, 0, 0, 0)
    shape = img.shape
    vals = img.sum()
    return path, shape + (vals,)
