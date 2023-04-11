from __future__ import annotations

__all__ = ["read_tile", "read_slide"]

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from histoprep.backend import (
    OPENSLIDE_READABLE_FORMATS,
    CziBackend,
    OpenSlideBackend,
    PillowBackend,
)

ERROR_AUTOMATIC = (
    "Could not automatically assing reader for path: '{}'. Please choose from {}."
)
ERROR_BACKEND = "Backend '{}' does not exist, choose from: {}."
AVAILABLE_BACKENDS = ["PILLOW", "OPENSLIDE", "CZI"]


def read_slide(  # noqa
    path: str | Path, backend: str | None = None
) -> CziBackend | OpenSlideBackend | PillowBackend:
    """Get backend based on file-extension or backend argument."""
    if not isinstance(path, Path):
        path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path.resolve()))
    if backend is None:
        # Based on file-extension.
        if path.name.endswith(OPENSLIDE_READABLE_FORMATS):
            return OpenSlideBackend(path)
        if path.name.endswith(("jpeg", "jpg")):
            return PillowBackend(path)
        if path.name.endswith("czi"):
            return CziBackend(path)
        raise ValueError(ERROR_AUTOMATIC.format(path, AVAILABLE_BACKENDS))
    if isinstance(backend, str):
        # Based on backend argument.
        if "PIL" in backend.upper():
            return PillowBackend(path)
        if "OPEN" in backend.upper():
            return OpenSlideBackend(path)
        if "CZI" in backend.upper() or "ZEISS" in backend.upper():
            return CziBackend(path)
    if isinstance(
        backend, (type(CziBackend), type(OpenSlideBackend), type(PillowBackend))
    ):
        return backend(path=path)
    raise ValueError(ERROR_BACKEND.format(backend, AVAILABLE_BACKENDS))


def read_tile(
    worker_state: dict,
    xywh: tuple[int, int, int, int],
    *,
    level: int,
    transform: Callable[[np.ndarray], Any] | None,
    return_exception: bool,
) -> np.ndarray | Exception | Any:
    """Parallisable tile reading function."""
    reader = worker_state["reader"]
    try:
        tile = reader.read_region(xywh=xywh, level=level)
    except KeyboardInterrupt:
        raise KeyboardInterrupt from None
    except Exception as catched_exception:  # noqa
        if not return_exception:
            raise catched_exception  # noqa
        return catched_exception
    if transform is not None:
        return transform(tile)
    return tile
