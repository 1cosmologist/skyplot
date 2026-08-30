skyplot documentation
====================

``skyplot`` provides HEALPix map sampling and Cartopy-projected pcolormesh rendering
for CMB and other astrophysical sky maps.

Features
--------

- Rendering uses Matplotlib ``pcolormesh`` with Cartopy projections.
- Inputs can be 1D HEALPix arrays, 2D arrays plus ``wcs=``, or ndmap-like
  objects with a ``.wcs`` attribute.
- Available projections are ``mollweide``, ``orthographic``, ``platecarree``,
  ``equidistantconic``, and ``gnomonic``.
- Preset sampling densities are ``low=(480, 960)``, ``medium=(720, 1440)``,
  and ``high=(1440, 2880)``; ``medium`` is the default.
- ``cmap`` accepts Matplotlib colormap names and names from the ``colormaps``
  package. See `colormaps documentation <https://pratiman-91.github.io/colormaps/>`__
  for supported colormaps; ``planck`` and ``planck_log`` are also available.
- Projection controls, including ``central_longitude`` and ``central_latitude``,
  are passed through ``projection_kwargs``.
- Set ``figsize`` and ``dpi`` to control figure pixel dimensions.
- Binary mask and transparent vector overlays are available through
  ``plot_mode="overlay_mask"`` and ``plot_mode="vector_field"``. Vector
  overlays are drawn on axes from a prior scalar magnitude-map call.
- Gnomonic views default to no graticules and display the center, patch size,
  and pixel size as projection metadata; enable local graticules with
  ``show_gridlines=True``.
- Fonts in ``skyplot/fonts/`` are bundled and set as the default sans-serif
  family on import.

Installation
------------

Clone the repository and install the package in an isolated Python environment.
The editable install keeps your environment in sync with local source changes.

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

.. toctree::
   :maxdepth: 2
   :caption: Contents

   usage
   api
