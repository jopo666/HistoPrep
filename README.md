<div align="center">

# HistoPrep
Preprocessing large medical images for machine learning made easy!

<p align="center">
    <a href="#version" alt="Version">
        <img src="https://img.shields.io/pypi/v/histoprep"/></a>
    <a href="#licence" alt="Licence">
        <img src="https://img.shields.io/github/license/jopo666/HistoPrep"/></a>
    <a href="#issues" alt="Issues">
        <img src="https://img.shields.io/github/issues/jopo666/HistoPrep"/></a>
    <a href="#activity" alt="Activity">
        <img src="https://img.shields.io/github/last-commit/jopo666/HistoPrep"/></a>
</p>

<p align="center">
  <a href="#description">Description</a> •
  <a href="#installation">Installation</a> •
  <a href="https://jopo666.github.io/HistoPrep/">Documentation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#examples">Examples</a> •
  <a href="#whats-coming">What's coming?</a> •
  <a href="#citation">Citation</a>
</p>

</div>


## Description

This module allows you to easily **cut** and **preprocess** large histological slides.

- Cut tiles from large slide images.
- Dearray TMA spots (and cut tiles from individual spots).
- Preprocess extracted tiles **automatically**.

## Installation 

```bash 
pip install histoprep
```

## Cutting slide into tiles

``HistoPrep`` can be used easily to prepare histological slide images for machine learning tasks.

You can either use `HistoPrep` as a python module...

```python
import histoprep

# Cutting tiles is super easy!
reader = histoprep.SlideReader('/path/to/slide')
metadata = reader.save_tiles(
    '/path/to/output_folder',
    coordinates=reader.get_tile_coordinates(
        width=512, 
        overlap=0.1, 
        max_background=0.96
    ),
)
```
or as an excecutable from your command line!

```bash
jopo666@MacBookM1$ HistoPrep input_dir output_dir width {optional arguments}
```

### Preprocessing

After the tiles have been saved, preprocessing is just a simple outlier detection from the preprocessing metrics saved in `tile_metadata.csv`!

```python
from histoprep import OutlierDetector
from histoprep.helpers import combine metadata

# Let's combine all metadata from the cut slides
metadata = collect_metadata("/path/to/output_folder", "tile_metadata.csv")
metadata["outlier"] = False 
# Then mark any outlying values!
metadata.loc[metadata['sharpness_max'] < 5, "outlier"] = True     # blurry
metadata.loc[metadata['black_pixels'] > 0.05, "outlier"] = True   # data loss
metadata.loc[metadata['saturation_mean'] > 230, "outlier"] = True # weird blue shit

# This can also be done automatically!
detector = OutlierDetector(metadata, num_clusters=10)
# Plot clusters from most likely outlier to least likely outlier
detector.plot_clusters()
# After visual inspection we can discard some clusters as outliers.
metadata.loc[detector.clusters < 2, "outlier"] = True 
```

## Examples

Examples can be found in the [docs](https://github.io/jopo666/HistoPrep/).

## What's coming?

`HistoPrep` is under constant development. If there are some features you would like to be added, just submit an [issue](https://github.com/jopo666/HistoPrep/issues) and we'll start working on the feature!

## Citation

If you use `HistoPrep` in a publication, please cite the github repository.

```
@misc{histoprep2021,
  author = {Pohjonen J. and Ariotta. V},
  title = {HistoPrep: Preprocessing large medical images for machine learning made easy!},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/jopo666/HistoPrep}},
}
```
