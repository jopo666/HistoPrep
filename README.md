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

```bash
# install as a module   
pip install histoprep

# install as an executable
git clone https://github.com/jopo666/HistoPrep
cd HistoPrep
pip install -r requirements.txt
```

You should also have `openslide-tools` installed on your machine.

```bash
sudo apt-get install openslide-tools
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

- [Cutting and preprocessing a whole slide image.](https://github.com/jopo666/HistoPrep/blob/master/examples/cut.ipynb)
- [Cutting individual TMA spots from a slide.](https://github.com/jopo666/HistoPrep/blob/master/examples/dearray.ipynb) [coming in the future!]
