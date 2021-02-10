Installation
================================================================================

Dependecies
********************************************************************************

``HistoPrep`` depends heavily on two exceptional python packages ``openslide-python`` and ``opencv-python``. Both of these are C libraries and require additional steps to install on your system.

The official instructions for installing ``openslide`` on your system can be found `here <https://openslide.org/download/>`_.

The official instructions for installing ``opencv`` on your system can be found below.
   - `General install instructions <https://docs.opencv.org/master/d0/d3d/tutorial_general_install.html>`_
   - `MacOS <https://docs.opencv.org/master/d0/db2/tutorial_macos_install.html>`_
   - `Linux <https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html>`_
   - `Windows <https://docs.opencv.org/master/d3/d52/tutorial_windows_install.html>`_


.. note::
    There is an annoying `bug <https://github.com/openslide/openslide/issues/278>`_ when installing ``openslide`` on MacOS with ``brew``. This can be easily fixed by running ``brew upgrade pixman`` after the installation is complete.


``HistoPrep``
********************************************************************************

Be sure you have installed ``openslide`` and ``opencv`` succesfully on your system before installing ``HistoPrep``

.. code-block:: bash

   # update pip
   pip install --upgrade pip

   # install as a module   
   pip install histoprep

   # or install as an executable
   git clone https://github.com/jopo666/HistoPrep
   cd HistoPrep
   pip install -r requirements.txt


