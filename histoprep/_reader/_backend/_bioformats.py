## DEPRACATED (for now at least)

# import logging
# import warnings
# from typing import Dict, Tuple

# import bioformats
# import numpy

# from ._base import Backend
# from ...helpers._javabridge import start_javabridge

# __all__ = ["BioformatsBackend"]

# BIOFORMATS_READABLE = bioformats.READABLE_FORMATS


# class BioformatsBackend(Backend):
#     def __init__(self, path: str):
#         # Start javabridge.
#         start_javabridge()
#         self.reader = bioformats.get_image_reader("reader", path=path)
#         # Figure out levels and dimensions.
#         self.__get_level_info()

#     def __get_level_info(self):
#         # Find out dimensions for each level.
#         level_dimensions = {}
#         for d in range(self.reader.rdr.getSeriesCount()):
#             # Read something from dimension
#             __ = self.reader.read(series=d, XYWH=(0, 0, 1, 1))
#             # Get level dimensions.
#             level_dimensions[d] = (
#                 self.reader.rdr.getSizeY(),
#                 self.reader.rdr.getSizeX(),
#             )
#         # Find out level downsamples.
#         level_downsamples = {}
#         remove = []
#         for level, dims in level_dimensions.items():
#             downsample_found = False
#             # Test different downsamples.
#             for d in range(20):
#                 # Calculate difference in dimensions form optimal downsample.
#                 y_diff = abs(dims[0] - level_dimensions[0][0] / 2**d)
#                 x_diff = abs(dims[1] - level_dimensions[0][1] / 2**d)
#                 if x_diff <= 1 and y_diff <= 1:
#                     # Differences less than one are caused by rounding.
#                     level_downsamples[level] = (
#                         level_dimensions[0][0] / dims[0],
#                         level_dimensions[0][1] / dims[1],
#                     )
#                     downsample_found = True
#                     break
#             # If we couldn't find a downsample the level does not contain
#             # an image of the tissue --> add to remove list.
#             if not downsample_found:
#                 remove.append(level)
#         # Remove levels with some other images.
#         for level in remove:
#             logging.debug(
#                 "Level {} is not a downsample of the slide image.".format(level)
#             )
#             level_dimensions.pop(level)
#         # Cahce results.
#         self.__level_dimensions = level_dimensions
#         self.__level_downsamples = level_downsamples

#     def get_dimensions(self) -> Dict[int, Tuple[int, int]]:
#         return self.__level_dimensions[0]

#     def get_level_dimensions(self) -> Dict[int, Tuple[int, int]]:
#         return self.__level_dimensions

#     def get_level_downsamples(self) -> Dict[int, Tuple[int, int]]:
#         return self.__level_downsamples

#     def get_thumbnail(self, level: int) -> numpy.ndarray:
#         # Read thumbnail.
#         thumbnail = self.reader.read(series=level)
#         # To uint8.
#         return (thumbnail * 255).astype(numpy.uint8)

#     def read_region(
#         self, XYWH: Tuple[int, int, int, int], level: int
#     ) -> numpy.ndarray:
#         # Read region
#         tile = self.reader.read(series=level, XYWH=XYWH)
#         # To uint8.
#         return (tile * 255).astype(numpy.uint8)

#     def __repr__(self):
#         return "BIOFORMATS"
