__all__ = ["worker_init"]

from pathlib import Path


def worker_init(worker_state, reader_class, path: Path) -> None:  # noqa
    """Worker initialization function for concurrent functions with reader."""
    worker_state["reader"] = reader_class(path)
