Usage
=====

Basic example
-------------

.. code-block:: python

   import healpy as hp
   import numpy as np
    from skyplot import mollweide, save_figure

   nside = 64
   hp_map = np.random.default_rng(123).normal(size=hp.nside2npix(nside))

   fig = mollweide(
       hp_map,
       projection_kwargs={"central_longitude": 120.0},
       cmap="batlow",
       figsize=(12, 6),
       dpi=300,
       title="Simulated sky map",
   )

   fig.show()
   save_figure(fig, "simulated_sky.html")
   save_figure(fig, "simulated_sky.png", figsize=(12, 6), dpi=300)
   save_figure(fig, "simulated_sky_data", output_format="json")

Notes
-----

- Rendering uses Matplotlib ``pcolormesh`` with Cartopy projections.
- Inputs can be 1D HEALPix arrays, 2D arrays plus ``wcs=``, or ndmap-like objects with a ``.wcs`` attribute.
- Available projections include ``mollweide``, ``orthographic``, ``platecarree``,
    and ``equidistantconic``.
- Preset sampling densities are ``low=(480, 960)``, ``medium=(720, 1440)``, and
    ``high=(1440, 2880)`` with ``medium`` as default behavior.
- ``cmap`` accepts Matplotlib colormap names and names from the ``colormaps`` package.
- See ``colormaps`` documentation `<https://pratiman-91.github.io/colormaps/>`__ for all supported colormaps.
- Additionally ``planck`` and ``planck_log`` colormaps are also available.
- Projection controls (including ``central_longitude`` and ``central_latitude``)
    can be passed through ``projection_kwargs``.
- Set ``figsize`` and ``dpi`` to control figure pixel dimensions.
- HTML export embeds a PNG preview of the Matplotlib figure.
- Fonts placed in ``skyplot/fonts/`` are bundled and set as default sans-serif on import.
