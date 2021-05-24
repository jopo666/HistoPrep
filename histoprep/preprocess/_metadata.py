import os
from os.path import join, basename, dirname, exists
from typing import List
import logging
import multiprocessing as mp

import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm


__all__ = [
    'combine_metadata',
    'update_paths',
    'check_tiles'
]

# Define logger.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def combine_metadata(
        parent_dir: str,
        csv_path: str = None,
        overwrite: bool = False,
        tma_spots: bool = False
) -> pd.DataFrame:
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
    update_paths = []
    for f in os.scandir(parent_dir):
        if f.is_dir() and exists(join(f.path, 'metadata.csv')):
            update_paths.append(f.path)
    if len(update_paths) == 0:
        logger.info(f'No directories found at {parent_dir}!')
        return
    multiprocessing_loop(
        func=update_meta,
        loop_this=update_paths,
        desc='Updating paths',
        processes=1 if len(update_paths) < 20 else None
    )


def update_meta(data_dir: str):
    """Update metadata paths."""
    all_paths = []
    for x in os.scandir(data_dir):
        if x.name.endswith('metadata.csv'):
            all_paths.append(x.path)
    for path in all_paths:
        # Load metadata.
        meta = pd.read_csv(path)

        if 'spot_metadata' in path:
            # Update spot metadata.
            spot_paths = meta.path[~meta.path.isna()]
            meta.loc[~meta.path.isna(), 'path'] = update_spot_paths(
                spot_paths, data_dir)
        elif 'spot_path' in meta.columns:
            # Update spot and tile metadata from a TMA array.
            spot_paths = meta.spot_path[~meta.spot_path.isna()]
            meta.loc[~meta.spot_path.isna(), 'spot_path'] = update_spot_paths(
                spot_paths, data_dir)
            tile_paths = meta.path[~meta.path.isna()]
            meta.loc[~meta.path.isna(), 'path'] = update_tile_paths(
                tile_paths, data_dir, TMA=True)
        else:
            # Update only tile paths.
            tile_paths = meta.path[~meta.path.isna()]
            meta.loc[~meta.path.isna(), 'path'] = update_tile_paths(
                tile_paths, data_dir, TMA=False)
        # Save updated metadata.
        meta.to_csv(path, index=False)


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


def check_tiles(parent_dir: str, overwrite: bool = False) -> pd.DataFrame:
    """Checks each saved tile in the parent_dir.

    Loads each tile in the metadata.path column and saves the image 
    dimensions, number of NaN pixels and any file corruptions to 
    metadata.csv.

    Args:
        parent_dir (str): Directory of processed slides.

    Raises:
        IOError: parent_dir does not exists.

    Returns:
        pd.DataFrame: Combined dataframe with additional info for each image.
    """
    # Define logger.
    if not os.path.exists(parent_dir):
        raise IOError(f'Path {parent_dir} does not exists!')
    # Collect metadata to check.
    meta_paths = []
    for f in os.scandir(parent_dir):
        path = join(f.path, 'metadata.csv')
        if f.is_dir() and exists(path):
            meta_paths.append(path)
    if len(meta_paths) == 0:
        logger.info(f'No metadata dataframes found inside {parent_dir}!')
        return
    total = str(len(meta_paths))
    combined = []
    for i, meta_path in enumerate(meta_paths):
        # Define desc and load metadata.
        current = str(i+1).rjust(len(total))
        desc = f'[{current}/{total}] {basename(dirname(meta_path))}'
        metadata = pd.read_csv(meta_path)
        if not overwrite and 'corrupted' in metadata.columns:
            multiprocessing_loop(lambda x: x, [],
                                 desc=f'[{current}/{total}] Already checked')
            combined.append(metadata)
            continue
        paths = metadata.path.tolist()
        # Check images.
        results = multiprocessing_loop(
            func=check_image,
            loop_this=paths,
            desc=desc,
        )
        # Unpack results.
        shapes = []
        nan_counts = []
        corruptions = []
        file_exists = []
        for d in results:
            shapes.append(d['shape'])
            nan_counts.append(d['nan_count'])
            corruptions.append(d['corrupted'])
            file_exists.append(d['exists'])
        # Add to metadata and save!
        metadata['shape'] = shapes
        metadata['nan_count'] = nan_counts
        metadata['corrupted'] = corruptions
        metadata['exists'] = file_exists
        metadata.to_csv(meta_path, index=False)
        # Add to combined list.
        combined.append(metadata)
    # Combine each metadata and return to user.
    if len(combined) > 0:
        combined = pd.concat(combined)
        unique_shapes = len(
            combined['shape'][~combined['shape'].isna()].unique()
        )
        # Log results.
        print(f'Found {sum(~combined.exists)} tiles that do not exist.')
        print(f'Found {combined.corrupted.sum()} corrupted images.')
        print(f'Found {unique_shapes} unique shapes for images.')
    else:
        combined = None
    return combined


def check_image(path):
    """Collect info on the image."""
    if not exists(path):
        return {
            'shape': None,
            'nan_count': None,
            'corrupted': None,
            'exists': False
        }
    with open(path, 'rb') as f:
        check_chars = f.read()[-2:]
    # Check corruption
    if check_chars != b'\xff\xd9':
        return {
            'shape': None,
            'nan_count': None,
            'corrupted': True,
            'exists': True,
        }
    else:
        # Load image.
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return {
            'shape': image.shape,
            'nan_count': np.isnan(image).sum(),
            'corrupted': False,
            'exists': True,
        }


def multiprocessing_loop(func, loop_this, desc=None, processes=None):
    """Use for easy multiprocessing."""
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
