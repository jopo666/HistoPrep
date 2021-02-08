``HistoPrep``
=============

Welcome to the documentation of ``HistoPrep``!

What is ``HistoPrep``
*********************

``HistoPrep`` can be used to process and preprocess large histological slides for machine learning.

- Cut large slide images into tiles of desired size.
- Dearray individual tissue microarray spots from a large slide image.
- **Easily** detect and discard artifacts or blurry images after cutting.

Installation
************

``
# install as a module   
pip install histoprep

# install as an executable
git clone https://github.com/jopo666/HistoPrep
cd HistoPrep
pip install -r requirements.txt
``

You should also have `openslide-tools` installed on your machine.

``
sudo apt-get install openslide-tools
``

Cut
*******

.. automodule:: histoprep
    :members: Cutter

Dearray
*******

.. automodule:: histoprep
    :members: Dearray

Preprocess
**********

.. automodule:: histoprep.preprocess
    :members:

Indices and tables
******************

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
