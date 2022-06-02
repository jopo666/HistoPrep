import time
from contextlib import contextmanager


@contextmanager
def timeit() -> float:
    """Use for easy speed tests."""
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start


def format_seconds(n: float) -> str:
    """Format seconds into pretty string format."""
    if n < 0:
        raise ValueError("n should be positive.")
    years = int(n // (365.25 * 24 * 3600))
    n %= 365.25 * 24 * 3600
    days = int(n // (24 * 3600))
    n %= 24 * 3600
    hours = int(n // 3600)
    n %= 3600
    minutes = int(n // 60)
    n %= 60
    seconds = n
    if years > 0:
        strtime = f"{years}y {days}d {(hours)}h:{minutes}m:{int(seconds)}s"
    elif days > 0:
        strtime = f"{days}d {(hours)}h:{minutes}m:{int(seconds)}s"
    elif hours > 0:
        strtime = f"{(hours)}h:{minutes}m:{int(seconds)}s"
    elif minutes > 0:
        strtime = f"{minutes}m:{int(seconds)}s"
    else:
        strtime = f"{seconds:.3g}s"
    return strtime


def strtime():
    """Returns date and timestamp as a string."""
    return time.strftime("%h_%d_%H:%M")


def get_ETC(times: list, num_left: int):
    """Calculates ETC based on mean times and number of iters left."""
    if len(times) == 0:
        return ""
    else:
        seconds = sum(times) / len(times)
        return f"ETC: {format_seconds(seconds * num_left)}"
