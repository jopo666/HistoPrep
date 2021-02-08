HistoPrep
=========

Welcome to the documentation of ``HistoPrep``!

What is ``HistoPrep``?
**********************

``HistoPrep`` can be used to process and preprocess large histological slides for machine learning.

- Cut large slide images into tiles of desired size.
- Dearray individual tissue microarray spots from a large slide image.
- **Easily** detect and discard artifacts or blurry images after cutting.

Installation
************

.. code-block:: bash

   # install as a module   
   pip install histoprep

   # install as an executable
   git clone https://github.com/jopo666/HistoPrep
   cd HistoPrep
   pip install -r requirements.txt

You should also have ``openslide-tools`` installed on your machine.

.. code-block:: bash

   sudo apt-get install openslide-tools

API documentation
=================

``Cutter``
**********
.. autoclass:: histoprep.Cutter
    :members:
    :undoc-members:
    :show-inheritance:


``Dearrayer``
*************
.. autoclass:: histoprep.Dearrayer
    :members:
    :undoc-members:
    :show-inheritance:

``preprocess``
**************
.. automodule:: histoprep.preprocess
    :members: combine_metadata, plot_on_thumbnail, plot_tiles, plot_histograms, 
      plot_ranges
    :undoc-members:
    :show-inheritance:

``preprocess.functional``
+++++++++++++++++++++++++
.. automodule:: histoprep.preprocess.functional
    :members: tissue_mask, HSV_quantiles, RGB_quantiles, data_loss, sharpness,
      preprocess, sliding_window, PIL_to_array, array_to_PIL, mask_to_PIL,
    :undoc-members:
    :show-inheritance: