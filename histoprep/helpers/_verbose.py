import time
from typing import Iterable

__all__ = ["progress_bar"]


def progress_bar(
    iterable: Iterable,
    total: int = None,
    desc: str = None,
    log_interval: int = 1,
    log_values: bool = False,
    suppress: bool = False,
):
    """Simple print-based progress bar.

    Args:
        iterable: Iterable to wrap.
        total: Total steps. Defaults to None.
        desc: Description for progress bar. Defaults to None.
        log_interval: Log every n steps. Defaults to 1.
        log_values: Returns (dict, next(iterable)), where values added to the dict are
            logged to the progress bar. Defaults to True.
        suppress: Suppress all output. Defaults to False.

    Yields:
        Iterable output
    """
    logs = {}
    if suppress:
        for output in iterable:
            # Return logs still.
            if log_values:
                yield logs, output
            else:
                yield output
        return
    try:
        total = len(iterable) if total is None else total
        if total == 0:
            total = None
    except TypeError:
        total = None
    tic = time.perf_counter()
    tic_out = time.perf_counter()
    logs = {}
    bar = __get_bar(0, tic, total, desc=desc, logs=logs)
    last_length = len(bar)
    print(bar, end="\r")
    step = 0
    for step, output in enumerate(iterable):
        if step == 0 or (
            (step + 1) % log_interval == 0 and time.perf_counter() - tic_out > 0.05
        ):
            bar = __get_bar(step + 1, tic, total, desc=desc, logs=logs)
            print(" " * last_length, end="\r")
            print(bar, end="\r")
            last_length = len(bar)
            tic_out = time.perf_counter()
        if log_values:
            yield logs, output
        else:
            yield output
    print(" " * last_length, end="\r")
    print(__get_bar(step + 1, tic, total, desc=desc, final=True, logs=logs))


def __get_bar(
    step: int,
    tic: float,
    total: int = None,
    bar_length: int = 10,
    final: bool = False,
    desc: str = None,
    logs: dict = {},
):
    elapsed = __format_seconds(time.perf_counter() - tic)
    log_str = ""
    if len(logs) > 0:
        log_str = ", " + ", ".join(["{}={}".format(k, v) for k, v in logs.items()])
    if total is None:
        prefix = "" if desc is None else "{}: ".format(desc)
        bar = "{}{}it [{}{}]".format(prefix, step, elapsed, log_str)
    else:
        n = min(bar_length, int(bar_length * step / total))
        if final:
            bar = "#" * bar_length
        else:
            bar = "#" * n + " " * (bar_length - n)
        if step == 0:
            etc = "???"
        else:
            s_per_iter = (time.perf_counter() - tic) / step
            etc = __format_seconds((total - step) * s_per_iter)
        prefix = "" if desc is None else desc + " "
        bar = "{}|{}| {}/{} [{}<{}{}]".format(
            prefix, bar, step, total, elapsed, etc, log_str
        )
    return bar


def __format_seconds(n: int):
    """Format seconds into pretty string format."""
    days = int(n // (24 * 3600))
    n = n % (24 * 3600)
    hours = int(n // 3600)
    n %= 3600
    minutes = int(n // 60)
    n %= 60
    seconds = int(n)
    if days > 0:
        strtime = f"{days}d {hours}h:{minutes:02}m:{seconds:02}s"
    elif hours > 0:
        strtime = f"{hours}:{minutes:02}:{seconds:02}"
    else:
        strtime = f"{minutes:02}:{seconds:02}"
    return strtime


def verbose_fn(msg: str, desc: str, verbose: bool = True, color: bool = True):
    """Verbose output with description."""
    if not verbose:
        return
    if color:
        print("\x1b[1m\x1b[34m[{}]\x1b[0m {}".format(desc, msg))
    else:
        print("[{}] {}".format(desc, msg))
