import os
from typing import Callable, List
import pickle
import multiprocessing as mp
from functools import partial

from .._logger import progress_bar

__all__ = [
    'load_pickle',
    'save_pickle',
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
        multiprocess_map(func=remove, lst=paths, total=len(paths),
                         desc='Removing images')


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
    if days > 0:
        strtime = f'{days}d{(hours)}h:{minutes}m:{seconds}s'
    elif hours > 0:
        strtime = f'{(hours)}h:{minutes}m:{seconds}s'
    else:
        strtime = f'{minutes}m:{seconds}s'
    return strtime


def multiprocess_map(
    func: Callable,
    lst: list,
    processes: int = None,
    func_args: dict = {},
    **kwargs
):
    """Map function to a iterable and process with multiple processes."""
    results = []
    if processes is None:
        processes = os.cpu_count() - 1
    if kwargs.get('total') is None:
        try:
            kwargs['total'] = len(lst)
        except:
            pass
    func = partial(func, **func_args)
    with mp.Pool(processes=processes) as p:
        if kwargs.get('desc') is None or kwargs.get('desc') == "":
            loop = p.imap(func, lst)
        else:
            loop = progress_bar(p.imap(func, lst), **kwargs)
        for result in loop:
            results.append(result)
    return results
