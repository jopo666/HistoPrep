import os
from typing import List
import pickle
import multiprocessing as mp
import xml.etree.ElementTree as ET


import numpy as np
import pandas as pd
from tqdm import tqdm
from shapely.geometry import Polygon, MultiPolygon


def load_data(path):
    """Load pickle from path."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def save_data(data, path):
    """Save data to pickle."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def remove_extension(path: str) -> str:
    """Return filename with the extension removed."""
    if '.' in path:
        return '.'.join(path.split('.')[:-1])
    else:
        return path


def remove_images(image_dir: str) -> None:
    """Remove all images in the image folder"""
    paths = [x.path for x in os.scandir(image_dir)]
    if len(paths) > 0:
        with mp.Pool(processes=os.cpu_count()) as p:
            for __ in tqdm(
                p.imap(remove, paths),
                total=len(paths),
                desc='Removing images',
                bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
            ):
                continue


def remove(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)


def mask_from_xml(xml_path: str) -> Polygon:
    """
    Create a polygon mask based on ASAP annotations (xml).

    Args:
        xml_path (str): path to the annotation file

    Return:
        mask (shapely.Polygon): Polygon object mask
    """
    annotations = ET.parse(xml_path).getroot()
    polygons = []
    for annotation in annotations[0]:
        polygon = []
        for point in annotation[0]:
            polygon.append((
                round(float(point.get('X'))),
                round(float(point.get('Y'))))
            )
        if len(polygon) > 1:
            polygons.append(Polygon(np.array(polygon)))
    mask = MultiPolygon(polygons).buffer(0)
    return mask
