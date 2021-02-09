Installation
================================================================================

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
   # You might also need this if errors keep showing up.
   # sudo apt-get install libjpeg-dev zlib1g-de

There is also an annoying `bug <https://github.com/openslide/openslide/issues/278>`_. in ``openslide`` dependecies that might have to be fixed. If you are using ``conda`` the quick-fix is:

.. code-block:: bash

   conda install pixman