SkyPlot Documentation
====================

``skyplot`` is a visualization library for CMB and other astrophysical sky maps 
and aims to make figures optimized for publications and presentations easier.

Features
--------

- Inputs can be 1D HEALPix arrays, 2D arrays plus ``wcs=``, or pixell 
  ndmap-like objects with a ``.wcs`` attribute.
- Available projections are ``mollweide``, ``orthographic``, ``platecarree``,
  ``equidistantconic``, and ``gnomonic``.
- Preset sampling densities are ``low=(480, 960)``, ``medium=(720, 1440)``,
  and ``high=(1440, 2880)``; ``medium`` is the default.
- ``cmap`` accepts Matplotlib colormap names and names from the ``colormaps``
  package. See `colormaps documentation <https://pratiman-91.github.io/colormaps/>`__
  for supported colormaps; ``planck`` and ``planck_log`` are also available.
- Binary mask and transparent vector overlays are available through
  ``plot_mode="overlay_mask"`` and ``plot_mode="vector_field"``. Vector
  overlays are drawn on axes from a prior scalar magnitude-map call.
- Comes optimized for publication and presentation quality figures.

Installation
------------

Install from conda-forge (recommended). This installs SkyPlot and its dependencies
 in a compatible environment.

.. code-block:: console

   conda install -c conda-forge skyplot

For development, clone the repository and install it in an isolated Python
environment. The editable install keeps your environment in sync with local
source changes.

.. code-block:: console

   git clone https://github.com/1cosmologist/skyplot.git
   cd skyplot
   python -m venv .venv       # if needed
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   python -m pip install --upgrade pip
   python -m pip install .

For the test and documentation dependencies, install the optional extras:

.. code-block:: console

   python -m pip install -e '.[dev,docs]'

Longitude convention
--------------------

Important: Longitude increases from :math:`-180^\circ` to :math:`180^\circ` from *left to right* 
to keep the astro convention used in CMB map plots. :math:`({\rm lon}=0^\circ, {\rm lat}=0^\circ)` 
is the center of any map by default. With astro convention, the logitude increases to the left till 
:math:`180^\circ` and then wraps around increasing from :math:`180^\circ` to :math:`360^\circ` 
(which wraps to :math:`0^\circ`). In ``skyplot`` we transform the :math:`180^\circ` to :math:`360^\circ` 
range as :math:`{\rm longitude} - 360^\circ`. This is more suitable for passing arguments to the
``extent`` kwargs.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   usage
   api

