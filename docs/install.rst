Installation
================================================================================

Dependecies
********************************************************************************

``HistoPrep`` depends heavily on two exceptional C libraries, `OpenSlide <https://openslide.org>`_ and `OpenCV <https://opencv.org>`_). Both of these have python interfaces (``openslide-python`` and ``opencv-python``) which require that the original C libraries are installed and compiled on your system.

The official instructions for installing ``OpenSlide`` on your system can be found `here <https://openslide.org/download/>`_.

The official instructions for installing ``OpenCV`` on your system can be found below.

* `General install instructions <https://docs.opencv.org/master/d0/d3d/tutorial_general_install.html>`_
* `MacOS <https://docs.opencv.org/master/d0/db2/tutorial_macos_install.html>`_
* `Linux <https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html>`_
* `Windows <https://docs.opencv.org/master/d3/d52/tutorial_windows_install.html>`_


.. note::
    There is an annoying `bug <https://github.com/openslide/openslide/issues/278>`_ which sometimes arises when installing ``openslide``. This can be easily fixed by upgrading ``pixman`` package to the its newest version. If you are using Anaconda, you can simply run ``conda install pixman`` in your active conda environment. On MacOS you can also use ``brew upgrade pixman``.


``HistoPrep``
********************************************************************************

Be sure you have installed ``OpenSlide`` and ``OpenCV`` succesfully on your system before installing ``HistoPrep``.

.. code-block:: bash

   # upgrade pip
   pip install --upgrade pip

   # install as a module   
   pip install histoprep

   # or install as an executable
   git clone https://github.com/jopo666/HistoPrep
   pip install -r HistoPrep/requirements.txt
