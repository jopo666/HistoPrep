<div align="center">

# HistoPrep
Preprocessing large histological slides for machine learning made easy!
</div>

## Description

This module allows you to cut and preprocess large histological slides. Some of the features include:

- Cut large whole slide image (WSI) into tiles of desired size.
- Dearray individual tissue microarray (TMA) spots from a large slide image.
- **Easily** detect and discard blurry images or images with artifacts after cutting.
- Save a lot of tears while preprocessing images.

## Installation

First install `OpenCV` and `OpenSlide` on your system (instructions [here](https://docs.opencv.org/master/d0/d3d/tutorial_general_install.html) and [here](https://openslide.org/download/)).

```bash
# install as a module   
pip install histoprep

# install as an executable
git clone https://github.com/jopo666/HistoPrep
cd HistoPrep
pip install -r requirements.txt
```

## Usage

HistoPrep can be used either as a module...

```python
import histoprep as hp
cutter = hp.Cutter('/path/to/slide', width=512, overlap=0.25)
metadata = cutter.save('/path/to/output_folder')
```

or as an excecutable!

```bash
python3 HistoPrep cut ./input_dir ./output_dir --width 512 --overlap 0.25 --img_type jpeg
```

## Documentation

Documentation can be found [here](https://histoprep.readthedocs.io/en/latest/)!


## Examples

Detailed examples with best practices:

- Coming soon!