How to use ``HistoPrep``?
=========================

There are two main ways of using ``HistoPrep`` to cut/dearray large histological slides.

1. As a python module.

.. code-block:: python

  import histoprep as hp

  cutter = hp.Cutter('/path/to/slide', width=512, overlap=0.25,max_background=0.7)
  metadata = cutter.save('/path/to/output_folder')

2. As an excecutable.

.. code-block:: bash

  python3 HistoPrep cut ./input_dir ./output_dir --width 512 --overlap 0.25 --max_background 0.7


.. note::
    Recommended practice is to use the ``HistoPrep`` module inside a jupyter-notebook to find out optimal values that can then *optionally* be passed onto the excecutable to process a large number of files easily.


After the tiles have been extracted from the slides, preprocessing can be done with the ``histoprep`` module.

Cutting tiles from a large histological slide.
********************************************************************************

Cutting a large image into smaller tiles for machine learning can be done with only a few lines.

.. code-block:: python

  import histoprep as hp

  cutter = hp.Cutter('/data/slide_1', width=512, overlap=0.25, max_background=0.7)
  metadata = cutter.save('/data/output_dir')

In the above example we cut out ``512x512`` pixel tiles from the image. Each tile overlaps with it's neighbours by 25% and tiles with more than 70% are discarded.

With the save function, we save the tile images with metadata for each tile.

::

  /data/output_dir/
  └── slide_1
      ├── images [20456 entries exceeds filelimit, not opening dir]
      ├── metadata.csv
      ├── parameters.p
      ├── summary.txt
      ├── thumbnail.jpeg
      └── thumbnail_annotated.jpeg

During saving ``HistoPrep`` also calculates useful preprocessing metrics for each tile. These can be used to remove unwanted tiles (tiles with pen markings etc.) with *outlier analysis*. We'll take a closer look at these metrics in the preprocessing section.

To see all the available options for please take look at the documentation for 
`Cutter() <https://histoprep.readthedocs.io/en/latest/#cutter>`_.

.. note::
    A detailed example jupyter-notebook can be found `here <https://github.com/jopo666/HistoPrep/blob/master/examples/cut.ipynb>`_.


Dearraying a TMA slide
********************************************************************************

Tissue microarray slides often have tens or hundreds of samples from different patients. In practice, we could just cut the TMA slide as well wit the ``Cutter`` class, but then all the tiles from different patient samples would be mixed. Therefore, if we want to separate the tiles/spots by patient, we must first dearray the TMA slide.

Lucky for you, with ``HistoPrep`` dearraying can also be done with only a few lines of code!

.. code-block:: python

  import histoprep as hp

  dearrayer = hp.Dearrayer('/data/TMA_1')
  metadata = dearrayer.save('/data/output_dir')
  tile_metadata = dearrayer.cut_spots(width=512, overlap=0.25, max_background=0.7)


Preprocessing
********************************************************************************

**Under construction!**