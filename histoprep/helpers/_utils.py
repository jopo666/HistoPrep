import os
import time
from typing import List
import pickle
import multiprocessing as mp

import numpy as np
import pandas as pd
from tqdm import tqdm

__all__ = [
    'load_pickle',
    'save_pickle'
]


def load_pickle(path: str):
    """
    Load pickle from path.

    Args:
        path (str): Path to the pickle file.

    Raises:
        IOError: Path does not exist.

    Returns:
        [any]: File saved in a pickle.
    """
    if not os.path.exists(path):
        raise IOError(f'{path} does not exist!')
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def save_pickle(data, path: str):
    """
    Save data to pickle.

    Args:
        data (any): Data to be saved.
        path (str): Filepath.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def remove_extension(path: str) -> str:
    """Return filename with the extension removed."""
    if '.' in path:
        return '.'.join(path.split('.')[:-1])
    else:
        return path


def remove_images(image_dir: str) -> None:
    """
    Remove all images in the image folder

    Args:
        image_dir (str): Directory to be removed
    """
    paths = [x.path for x in os.scandir(image_dir)]
    if len(paths) > 0:
        with mp.Pool(processes=os.cpu_count()) as p:
            for __ in tqdm(
                p.imap(remove, paths),
                total=len(paths),
                desc='Removing images',
                bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
            ):
                continue


def remove(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)


def flatten(l):
    """Flatten list of lists."""
    if all(isinstance(x, list) for x in l):
        return [item for sublist in l for item in sublist]
    else:
        return l


def format_seconds(n: int) -> str:
    """Format seconds into pretty string format."""
    days = int(n // (24 * 3600))
    n = n % (24 * 3600)
    hours = int(n // 3600)
    n %= 3600
    minutes = int(n // 60)
    n %= 60
    seconds = int(n)
    strtime = f'{(hours)}h:{minutes}m:{seconds}s'
    if days > 1:
        strtime = f'{days}d ' + strtime
    return strtime
