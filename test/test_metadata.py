import os
import shutil

import pandas
from histoprep.helpers import combine_metadata, strip_metric_colums

from paths import DATA_PATH, TMP_DIR, clean_tmp_dir


def test_strip_metrics():
    meta = pandas.read_csv(os.path.join(DATA_PATH, "metadata.csv"))
    assert meta.columns.tolist() == [
        "slide_name",
        "x",
        "y",
        "w",
        "h",
        "path",
        "background",
        "gray_mean",
        "red_mean",
        "green_mean",
        "blue_mean",
        "hue_mean",
        "saturation_mean",
        "brightness_mean",
        "gray_std",
        "red_std",
        "green_std",
        "blue_std",
        "hue_std",
        "saturation_std",
        "brightness_std",
        "black_pixels",
        "white_pixels",
        "sharpness_max",
        "red_q=0.05",
        "red_q=0.1",
        "red_q=0.5",
        "red_q=0.9",
        "red_q=0.95",
        "green_q=0.05",
        "green_q=0.1",
        "green_q=0.5",
        "green_q=0.9",
        "green_q=0.95",
        "blue_q=0.05",
        "blue_q=0.1",
        "blue_q=0.5",
        "blue_q=0.9",
        "blue_q=0.95",
        "hue_q=0.05",
        "hue_q=0.1",
        "hue_q=0.5",
        "hue_q=0.9",
        "hue_q=0.95",
        "saturation_q=0.05",
        "saturation_q=0.1",
        "saturation_q=0.5",
        "saturation_q=0.9",
        "saturation_q=0.95",
        "brightness_q=0.05",
        "brightness_q=0.1",
        "brightness_q=0.5",
        "brightness_q=0.9",
        "brightness_q=0.95",
    ]
    assert strip_metric_colums(meta).columns.tolist() == [
        "slide_name",
        "x",
        "y",
        "w",
        "h",
        "path",
        "background",
    ]


def test_combine_metadata():
    metadata_path = os.path.join(DATA_PATH, "metadata.csv")
    clean_tmp_dir()
    for i in range(5):
        slide_dir = os.path.join(TMP_DIR, "slide_{}".format(i))
        os.makedirs(slide_dir)
        slide_metadata_path = os.path.join(slide_dir, "metadata.csv")
        shutil.copy(metadata_path, slide_metadata_path)
    original = pandas.read_csv(metadata_path)
    combined = combine_metadata(TMP_DIR, "metadata.csv")
    assert len(original) * 5 == len(combined)
    clean_tmp_dir()
