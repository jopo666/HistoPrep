from pathlib import Path
from typing import Optional, Union

from ._czi import CziBackend
from ._openslide import OPENSLIDE_READABLE, OpenSlideBackend
from ._pillow import PillowBackend

ERROR_AUTOMATIC = (
    "Could not automatically assing reader for path: '{}'. Please choose from {}."
)
ERROR_BACKEND = "Backend '{}' does not exist, choose from: {}."
AVAILABLE_BACKENDS = ["PILLOW", "OPENSLIDE", "CZI"]


def read_slide(
    path: Union[str, Path], backend: Optional[str] = None
) -> Union[CziBackend, OpenSlideBackend, PillowBackend]:
    """Get backend based on file-extension or backend argument."""
    if not isinstance(path, Path):
        path = Path(path)
    if backend is None:
        # Based on file-extension.
        if path.name.endswith(OPENSLIDE_READABLE):
            return OpenSlideBackend(path)
        if path.name.endswith(("jpeg", "jpg")):
            return PillowBackend(path)
        if path.name.endswith("czi"):
            return CziBackend(path)
        raise ValueError(ERROR_AUTOMATIC.format(path, AVAILABLE_BACKENDS))
    # Based on backend argument.
    if "PIL" in backend.upper():
        return PillowBackend(path)
    if "OPENS" in backend.upper():
        return OpenSlideBackend(path)
    if "CZI" in backend.upper() or "ZEISS" in backend.upper():
        return CziBackend(path)
    raise ValueError(ERROR_BACKEND.format(backend, AVAILABLE_BACKENDS))
