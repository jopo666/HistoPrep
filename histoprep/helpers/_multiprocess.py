import functools
import logging
import multiprocessing
import os
from typing import Any, Callable, Iterable, Iterator, Tuple

__all__ = ["multiprocess_loop"]


def multiprocess_loop(
    func: Callable,
    iterable: Iterable,
    num_workers: int = None,
    initializer: Callable = None,
    initializer_args: Tuple[Any] = (),
    use_imap: bool = True,
    **kwargs,
) -> Iterator[Iterable]:
    """Maps function and iteration with multiple workers and yields the outputs.

    Args:
        func: Function to be called on each list item.
        iterable: Iterable passed to the function.
        num_workers: Number of worker processes. If None, set to the number of
            CPU cores. Defaults to None.
        initializer: Initializer function for each worker process. Defaults to
            None.
        initializer_args:: Arguments for the `initializer` function. Defaults to
            ().
        use_imap: Uses imap instead of map. Imap returns results in the same
            order but is slightly slower to start up. Defaults to True.
        **kwargs: Passed to the `func`.

    Return:
        Iterable of function outputs.

    Example:
        ```python
        from histoprep.helpers import multiprocess_loop, read_image, progress_bar

        tiles = []
        for tile in multiprocess_loop(
            func=read_image,
            iterable=paths,
            num_workers=20,
            return_arr=True # <- This is a keyword argument for read_image!
        ):
            tiles.append(tile)
        ```
    """
    if is_lambda(func):
        raise AttributeError("Lambda functions cannot be pickled for multiprocessing!")
    if is_local(func):
        raise AttributeError(
            "Locally defined functions cannot be pickled for multiprocessing!"
        )
    if not isinstance(iterable, Iterable):
        raise TypeError(
            "Passed `iterable` ({}) is not iterable.".format(type(iterable))
        )
    # Define processes.
    if num_workers is None:
        num_workers = os.cpu_count()
        logging.debug("Setting num_workers to {}.".format(num_workers))
    elif num_workers < 0:
        raise ValueError(
            "Number of workers should be positive (not {}).".format(num_workers)
        )
    # Wrap function.
    func = functools.partial(func, **kwargs)
    # Start pool.
    if num_workers > 1:
        with multiprocessing.Pool(
            processes=num_workers,
            initializer=initializer,
            initargs=initializer_args,
        ) as pool:
            if use_imap:
                for result in pool.imap(func, iterable):
                    yield result
            else:
                for result in pool.map(func, iterable):
                    yield result
    else:
        # Yield with the main process.
        for x in iterable:
            yield func(x)


def is_lambda(func):
    """Test if a function is a lambda function."""
    return hasattr(func, "__qualname__") and "<lambda>" in func.__qualname__


def is_local(func):
    """Test if a function is a locally defined function."""
    return hasattr(func, "__qualname__") and "<locals>" in func.__qualname__
