Usage
=====

This page follows the examples in ``notebooks/skyplot_package_test.ipynb``.
They render a HEALPix dust map, optionally overlay a Galactic-plane mask, and
then show the same data with each available sky projection.

Set up the map
----------------

Import the plotting functions and inspect the available projections and
sampling presets. ``resolution="high"`` is used by the notebook examples to
produce publication-sized renderings; use ``"low"`` or ``"medium"`` for
quicker previews.

.. code-block:: python

   from pathlib import Path

   import healpy as hp
   import matplotlib.colors as mpc
   import numpy as np

   from skyplot import (
       AVAILABLE_PROJECTIONS,
       FONT_SIZE_PRESETS,
       RESOLUTION_PRESETS,
       equidistantconic,
       gnomonic,
       mollweide,
       orthographic,
       platecarree,
       save_figure,
   )

   print("Available projections:", AVAILABLE_PROJECTIONS)
   print("Resolution presets:", RESOLUTION_PRESETS)
   print("Base font sizes (points):", FONT_SIZE_PRESETS)

Load a 1D HEALPix FITS map by setting ``map_path`` to a local file. The map
must contain the field selected by ``field``; the notebook uses the first
field, ``field=0``.

.. code-block:: python

   # Replace this with the path to your HEALPix FITS map.
   map_path = Path("path/to/your_healpix_map.fits")

   if not map_path.exists():
       raise FileNotFoundError(f"Set map_path to an existing file: {map_path}")

   hp_map = hp.read_map(map_path, field=0)
   print("Map loaded:", map_path)
   print("npix:", hp_map.size, "nside:", hp.get_nside(hp_map))
   print("dtype:", hp_map.dtype)

Mollweide full-sky view
-----------------------

This full-sky view uses a symmetric-logarithmic scale to retain both faint and
bright map structure. The second call layers the optional mask onto the axes
created by the first call.

Mask overlay (optional)
~~~~~~~~~~~~~~~~~~~~~~~

Load a Galactic-plane mask only if it is needed for this view. Zero-valued
pixels are painted as a translucent overlay. Select this layer solely with
``plot_mode="overlay_mask"``; scalar color options are ignored in this mode.

.. code-block:: python

   mask_path = Path("path/to/your_mask.fits")
   mask = hp.read_map(mask_path, field=1)

.. code-block:: python

   fig = mollweide(
       hp_map,
       resolution="high",
       cmap="sunburst",
       norm="symlog",
       vmin=0.0,
       vmax=100.0,
       badcolor=None,
       title="HEALPix map - Mollweide render",
   )

   mollweide(
       mask,
       resolution="high",
       ax=fig.axes[0],
       badcolor=None,
       plot_mode="overlay_mask",
       alpha=0.4,
       overlay_color="k",
       show_gridlines=False,
   )

.. figure:: figures/mollweide.png
   :alt: Full-sky Mollweide rendering of the HEALPix dust map with a subtle mask overlay.
   :width: 100%

   The notebook's Mollweide full-sky rendering.

Equidistant Conic regional view
-------------------------------

Use ``extent`` to select a region and set the conic's center and standard
parallels through ``projection_kwargs``. Here a ``SymLogNorm`` supplies more
control than the shorthand ``norm="symlog"``.

.. code-block:: python

   equidistantconic(
       hp_map,
       projection_kwargs={
           "central_longitude": -60.0,
           "central_latitude": -30.0,
           "standard_parallels": (-70.0, -15.0),
       },
       extent=(-150.0, 30.0, -80.0, 10.0),
       resolution="high",
       cmap="lipari",
       norm=mpc.SymLogNorm(
           linthresh=2.5, linscale=0.35, vmin=0.0, vmax=100.0
       ),
       title="HEALPix map - EquidistantConic regional render",
   )

.. figure:: figures/equidistantconic.png
   :alt: Southern regional Equidistant Conic rendering of the HEALPix dust map.
   :width: 100%

   The notebook's regional Equidistant Conic rendering.

Customization - Dark Mode
-------------------------

SkyPlot returns an ordinary Matplotlib figure, so use Matplotlib's
``rc_context`` to apply a style locally.  For Cartopy projections, also set
the ``"geo"`` spine explicitly: it is the map boundary and is separate from
the normal Matplotlib axes spines.  The colorbar is an additional axes, so its
spines, ticks, and label need the same foreground color. Here the map is Galactic 
coordinates while the mask is in Celestial coordinates.

.. code-block:: python

   import matplotlib.pyplot as plt

   with plt.rc_context({
       "figure.facecolor": "#101820",
       "axes.facecolor": "#101820",
       "axes.edgecolor": "white",
       "text.color": "white",
       "axes.labelcolor": "white",
       "xtick.color": "white",
       "ytick.color": "white",
       "font.size": 10,
       "font.family": "serif",
       "font.serif": ["STIX Two Text"],
       "mathtext.fontset": "stix",
   }):
       fig = equidistantconic(
           npipe_143[0],
           projection_kwargs={
               "central_longitude": 15.0,
               "central_latitude": -45.0,
               "standard_parallels": (-70.0, -15.0),
           },
           extent=(-90.0, 120.0, -65.0, -5.0),
           coordinate_transform=('galactic', 'icrs'),
           resolution="high",
           cmap="planck",
           vmin=-300.0,
           vmax=300.0,
           title="NPIPE CMB map at 143 GHz",
           gridline_kwargs={"linewidth": 0.4, "color": "white", "linestyle": "-"},
           colorbar_title=r"$\mu$K${}_{\rm CMB}$",
       )

       map_ax = fig.axes[0]
       map_ax.spines["geo"].set_edgecolor("white")
       map_ax.spines["geo"].set_linewidth(0.8)

       cbar_ax = fig.axes[-1]
       for spine in cbar_ax.spines.values():
           spine.set_edgecolor("white")
           spine.set_linewidth(0.8)
       cbar_ax.tick_params(colors="white")
       cbar_ax.xaxis.label.set_color("white")

       equidistantconic(
           mask,
           resolution="high",
           plot_mode="overlay_mask",
           overlay_color="black",
           alpha=0.4,
           ax=map_ax,
       )

.. figure:: figures/dark_mode.png
   :alt: Dark-themed Equidistant Conic CMB map with a translucent mask overlay.
   :width: 100%

   Equidistant Conic rendering styled with a local Matplotlib dark theme.

Orthographic view
-----------------

An Orthographic projection presents the sky as viewed from outside a globe.
Choose the displayed hemisphere with ``central_longitude`` and
``central_latitude``.

.. code-block:: python

   orthographic(
       hp_map,
       resolution="high",
       projection_kwargs={
           "central_longitude": 0.0,
           "central_latitude": 60.0,
           "azimuth": 0.0,
       },
       cmap="guppy_r",
       norm=mpc.SymLogNorm(
           linthresh=2.5, linscale=1.0, vmin=0.0, vmax=100.0
       ),
       badcolor=None,
       title="HEALPix map - Orthographic render",
   )

.. figure:: figures/orthographic.png
   :alt: Orthographic northern-sky rendering of the HEALPix dust map.
   :width: 75%

   The notebook's Orthographic rendering centered at longitude 0 and latitude 60 degrees.

Plate Carree coordinate transform
---------------------------------

This example displays a Galactic map on an ICRS/equatorial longitude-latitude
grid. ``coordinate_transform`` is ordered ``(source_frame, display_frame)``;
use Astropy frame strings such as ``"galactic"``, ``"icrs"``, and
``"geocentrictrueecliptic"``.

.. code-block:: python

   platecarree(
       hp_map,
       coordinate_transform=("galactic", "icrs"),
       extent=(-180.0, 180.0, -70.0, 20.0),
       resolution="high",
       cmap="wildfire",
       norm=mpc.SymLogNorm(
           linthresh=2.5, linscale=1.0, vmin=0.0, vmax=100.0
       ),
       badcolor=None,
       gridline_kwargs={"linewidth": 0.2, "color": "w", "linestyle": ":"},
       title="HEALPix map - PlateCarree regional render",
   )

.. figure:: figures/platecarree.png
   :alt: Plate Carree regional rendering of the HEALPix dust map transformed from Galactic to ICRS coordinates.
   :width: 100%

   The notebook's Plate Carree rendering after transforming Galactic coordinates to ICRS.

WCS-backed maps
---------------

SkyPlot also accepts a two-dimensional array with an Astropy WCS. The
notebook example starts from a three-axis CO cube whose WCS has the following
world-axis order:

.. code-block:: text

   0: VOPT       (velocity)
   1: GLON-CAR   (Galactic longitude)
   2: GLAT-CAR   (Galactic latitude)

The corresponding FITS pixel lengths are ``(146, 1441, 677)``. FITS data are
presented to NumPy in reverse axis order, so the cube has shape
``(677, 1441, 146)``. Integrating its last (velocity) axis produces the
two-dimensional map shape ``(677, 1441) = (GLAT, GLON)``.

.. code-block:: python

   from astropy.io import fits
   from astropy.wcs import WCS

   with fits.open("path/to/co_cube.fits") as hdulist:
       hdu = hdulist[0]
       wcs = WCS(hdu.header)
       data = hdu.data

   data = np.nan_to_num(data, nan=0.0)
   vel_inted_data = np.trapezoid(data, dx=0.65019, axis=2)
   assert vel_inted_data.shape == (677, 1441)  # (GLAT, GLON)

Pass the zero-based WCS *world-axis* indices, rather than NumPy array-axis
indices, through ``world_axis_mapping``. Here ``(1, 2)`` selects
``(GLON-CAR, GLAT-CAR)``; the retained velocity world axis is evaluated at its
reference value. If the non-spatial and sky axes are coupled, slice both the
WCS and cube to the desired plane before plotting.

.. code-block:: python

   platecarree(
       vel_inted_data,
       extent=(-180.0, 180.0, -80.0, 80.0),
       resolution="high",
       cmap="amethyst",
       wcs=wcs,
       world_axis_mapping=(1, 2),
       badcolor=None,
       norm=mpc.SymLogNorm(linthresh=1.0, linscale=1.0, vmin=0.0, vmax=50.0),
       gridline_kwargs={"linewidth": 0.2, "color": "w", "linestyle": ":"},
       title="WCS map - PlateCarree regional render",
       colorbar_title="Galactic CO (velocity integrated) in K km/s",
   )

The source WCS longitude pixels span ``-60°`` to ``300°`` and therefore have
a native seam at ``-60°/300°``. SkyPlot samples those native coordinates onto
the uniform displayed ``[-180°, 180°]`` longitude grid, including equivalent
360-degree longitudes as needed. The output is consequently centered at
``(0°, 0°)`` because of its ``extent``; ``central_longitude`` is not inferred
from the WCS. This differs from the source-array midpoint: the middle GLON
column is at ``120°``, while GLON ``0°`` is column 240. The array center and
the displayed map center therefore should not be expected to coincide.

.. figure:: figures/wcs.png
   :alt: Velocity-integrated Galactic CO WCS map rendered in Plate Carree coordinates.
   :width: 100%

   WCS-backed Galactic CO map sampled onto the standard Plate Carree display grid.

Gnomonic detail view
--------------------

Gnomonic is intended for a small tangent-plane region. ``center`` gives the
longitude and latitude in degrees, while ``xsize``, ``ysize``, and
``pixel_size_arcmin`` set the output sampling. The x-axis records the plot
center; the y-axis records patch dimensions and pixel size in arcminutes.
Optional curved, unlabeled longitude/latitude graticules can be enabled with
``show_gridlines=True`` and styled with ``gridline_kwargs``.

.. code-block:: python

   gnomonic(
       hp_map,
       center=(0.0, 90.0),
       xsize=1000,
       ysize=1000,
       pixel_size_arcmin=5.0,
       cmap="balance",
       vmin=0.0,
       vmax=5.0,
       title="HEALPix map - Gnomonic regional render",
       show_gridlines=True,
       gridline_kwargs={"color": "white", "linestyle": ":", "alpha": 0.5},
       figsize=(5.5, 6.5),
   )

.. figure:: figures/gnomonic.png
   :alt: Gnomonic tangent-plane rendering centered on the north pole of the HEALPix dust map.
   :width: 75%

   The notebook's Gnomonic detail view centered at longitude 0 and latitude 90 degrees.

Vector field visualization
--------------------------

``plot_mode="vector_field"`` adds a transparent streamplot or quiver artist
to an existing map axes. Render the scalar magnitude in a separate ordinary
map call first, then pass ``ax=fig.axes[0]`` for the overlay. Scalar map
arguments (``cmap``, ``vmin``, ``vmax``, and ``norm``) are ignored by the
overlay; configure the artist through ``vector_kwargs``.

For CMB polarization, the first and second input maps are the **east** and
**north** components of the director field, not the input Stokes ``U`` map.
The following notebook-derived snippet follows the **HEALPix/COSMO**
polarization convention:

.. code-block:: python

   hp_qu_map = hp.read_map("path/to/stokes_qu_map.fits", field=None)
   Q_stokes, U_stokes = hp_qu_map[:2]
   P = np.hypot(Q_stokes, U_stokes)
   psi = 0.5 * np.arctan2(U_stokes, Q_stokes)
   east_component = np.sin(psi)       # HEALPix e_phi
   north_component = -np.cos(psi)     # negative HEALPix e_theta

   vector_fig = mollweide(
       P,
       cmap="lipari",
       norm=mpc.SymLogNorm(linthresh=0.01, linscale=1.0, vmin=0.0, vmax=1.0),
       colorbar_title="Polarized intensity P",
       resolution="medium",
       title="HEALPix polarization orientation",
   )

.. warning::

   Polarization conventions require care. The component conversion above is
   for HEALPix/COSMO maps. IAU-convention products use the opposite Stokes
   ``U`` sign, so convert their ``U`` map before calculating ``psi``. Always
   confirm the convention recorded by the data product.

Streamplot
~~~~~~~~~~

Use streamplot for continuous paths through the direction field. ``density``
controls the number of paths; ``linewidth``, ``arrowsize``, and ``arrowstyle``
control their appearance.

.. code-block:: python

   mollweide(
       (east_component, north_component),
       ax=vector_fig.axes[0],
       plot_mode="vector_field",
       vector_kwargs={
           "method": "streamplot",
           "color": "white",
           "linewidth": 0.3,
           "arrowstyle": "-",
           "arrowsize": 0.5,
           "density": 3.,
       },
       resolution="medium",
   )

.. figure:: figures/streamplot.png
   :alt: Polarized intensity map with a white streamline overlay.
   :width: 100%

   Streamplot representation of the polarization direction field.

Quiver
~~~~~~

Use quiver for discrete vectors when controlled sampling is preferable to
continuous paths. Create a fresh magnitude figure when comparing it directly
with a streamplot result.

.. code-block:: python

   vector_fig = mollweide(
      P,
      cmap="lipari",
      norm=mpc.SymLogNorm(linthresh=0.01, linscale=1., vmin=0., vmax=1.),
      colorbar_title="Polarized intensity P",
      resolution="medium",
      title="HEALPix polarization orientation",
   )
   mollweide(
      (east_component, north_component),
      ax=vector_fig.axes[0],
      plot_mode="vector_field",
      vector_kwargs={
         "method": "quiver",
         "color": "white",
         "width": 0.005,
         "headwidth": 0.,
         "minshaft": 3.0,
         "alpha": 0.6,
      },
      resolution="medium",
   )

.. figure:: figures/quiver.png
   :alt: Polarized intensity map with a white quiver overlay.
   :width: 100%

   Quiver representation of the polarization direction field.

Polarization orientation has no intrinsic arrow direction. For a headless
streamline representation, use
``vector_kwargs={"method": "streamplot", "arrowstyle": "-"}``.

Saving a figure
---------------

Each plotting function returns a Matplotlib figure. Pass that figure to
``save_figure`` to write an image.

.. code-block:: python

   save_figure(fig, "simulated_sky.png", figsize=(12, 6), dpi=300)
