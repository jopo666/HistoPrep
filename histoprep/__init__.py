"""HistoPrep: Preprocessing large medical images for machine learning made easy!"""  # noqa

__all__ = ["SlideReader", "functional", "utils"]

from . import functional, utils
from ._reader import SlideReader
