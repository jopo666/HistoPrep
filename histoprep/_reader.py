__all__ = ["SlideReader"]

from pathlib import Path
from typing import Optional, Union

from .backend import CziReader, OpenSlideReader, PillowReader
from .backend._openslide import OPENSLIDE_READABLE

ERROR_AUTOMATIC_FAILED = "Could not automatically assing reader for path: {}"


def SlideReader(  # noqa: N802
    path: Union[str, Path],
    backend: Optional[Union[CziReader, OpenSlideReader, PillowReader]] = None,
) -> Union[PillowReader, OpenSlideReader, CziReader]:
    """Reader for histological slides.

    Args:
        path: Path to slide image.
        backend: Backend to use for reading image data. If None, attempts to assign
            reader based on the path suffix. Defaults to None.
    """
    if not isinstance(path, Path):
        path = Path(path)
    if backend is None:
        if path.name.endswith(OPENSLIDE_READABLE):
            return OpenSlideReader(path)
        if path.name.endswith(("jpeg", "jpg")):
            return PillowReader(path)
        if path.name.endswith("czi"):
            return CziReader(path)
        raise ValueError(ERROR_AUTOMATIC_FAILED.format(path))
    return backend.value(path)
