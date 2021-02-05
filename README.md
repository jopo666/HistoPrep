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

## Requirements

`python >= 3.8` and `openslide`

```bash
sudo apt-get install openslide-tools
```

## Installation
```bash
# install as a module   
pip install histoprep

# install as an executable
git clone https://github.com/jopo666/HistoPrep
```
## Usage

HistoPrep can be used either as a module...

```python
import histoprep as hp
cutter = hp.Cutter('/path/to/slide', width=512, overlap=0.25)
cutter.save('/path/to/output_folder')
```

or as an excecutable!

```bash
python3 HistoPrep cut ./input_dir ./output_dir --width 512 --overlap 0.25 --img_type jpeg
```

## Examples

Detailed examples with best practices:

- [Cutting and preprocessing a whole slide image.](https://github.com/jopo666/HistoPrep/examples/cut.ipynb)
- [Cutting individual TMA spots from a slide.](https://github.com/jopo666/HistoPrep/examples/dearray.ipynb)

## Documentation

Work in progress! Each function does have a detailed `__doc__` explaining the use of each argument.
