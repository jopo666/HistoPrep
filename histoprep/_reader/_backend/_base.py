from typing import Dict, Tuple

import numpy


class Backend:
    def __init__(self):
        pass

    @property
    def dimensions(self) -> Tuple[int, int]:
        return self.get_dimensions()

    @property
    def level_downsamples(self) -> Dict[int, Tuple[float, float]]:
        return self.get_level_downsamples()

    @property
    def level_dimensions(self) -> Dict[int, Tuple[int, int]]:
        return self.get_level_dimensions()

    def get_dimensions(self) -> Tuple[int, int]:
        raise NotImplementedError()

    def get_level_downsamples(self) -> Dict[int, Tuple[int, int]]:
        raise NotImplementedError()

    def get_level_dimensions(self) -> Dict[int, Tuple[int, int]]:
        raise NotImplementedError()

    def get_thumbnail(self, level: int) -> numpy.ndarray:
        raise NotImplementedError()

    def read_region(self, XYWH: Tuple[int, int, int, int], level: int) -> numpy.ndarray:
        raise NotImplementedError()
