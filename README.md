# SkyPlot

SkyPlot is a Python visualization library for CMB and astrophysical sky maps.
It provides one Matplotlib-based plotting interface for full-sky HEALPix maps
and two-dimensional WCS-backed arrays, with Cartopy projections, map overlays,
and publication-oriented output.

Documentation: <https://skyplot.readthedocs.io> · Source and issues:
<https://github.com/1cosmologist/skyplot>

## Highlights

- Plot 1D HEALPix arrays, 2D arrays with an Astropy WCS, and ndmap-like
  objects carrying a `.wcs` attribute.
- Render Mollweide, Orthographic, Plate Carrée, Equidistant Conic, and local
  Gnomonic views.
- Reproject a map between Astropy coordinate frames while plotting.
- Overlay binary masks, streamlines, or quiver vectors on an existing map.
- Use Matplotlib colormaps as well as palettes from
  [`colormaps`](https://pratiman-91.github.io/colormaps/).
- Export figures as PNG, JPG, SVG, PDF, or EPS.

## Installation

Install from conda-forge (recommended):

```console
conda install -c conda-forge skyplot
```

For a source or development installation:

```console
git clone https://github.com/1cosmologist/skyplot.git
cd skyplot
python -m pip install -e '.[dev,docs]'
```

SkyPlot requires Python 3.10 or later.

## Quick start

```python
import healpy as hp
import numpy as np

from skyplot import mollweide, save_figure

nside = 64
sky_map = np.random.default_rng(1234).normal(size=hp.nside2npix(nside))

fig = mollweide(
    sky_map,
    resolution="medium",
    cmap="batlow",
    title="Example CMB-like map",
    colorbar_title=r"$\mu$K",
)

save_figure(fig, "cmb_map.png", figsize=(12, 6), dpi=300)
```

In a notebook, display a returned figure explicitly:

```python
from IPython.display import display

display(fig)
```

## Layers: masks and vector fields

Each plotting function accepts `plot_mode`. Use the ordinary `"map"` mode for
a scalar map, `"overlay_mask"` for a binary mask, and `"vector_field"` for a
transparent streamplot or quiver layer. Render the scalar magnitude first;
then pass its Cartopy axes to the overlay call.

```python
P = np.hypot(q_map, u_map)
fig = mollweide(P, cmap="lipari", colorbar_title="Polarized intensity P")

# Components in the displayed local east/north basis.
psi = 0.5 * np.arctan2(u_map, q_map)
east = np.sin(psi)
north = -np.cos(psi)

mollweide(
    (east, north),
    ax=fig.axes[0],
    plot_mode="vector_field",
    vector_kwargs={
        "method": "streamplot",  # or "quiver"
        "color": "white",
        "linewidth": 0.5,
        "arrowstyle": "-",
        "density": 1.2,
    },
)
```

For HEALPix/COSMO polarization conventions, the components above give
`Q > 0, U = 0` along the North–South axis and `Q = 0, U > 0` along the
North-West to South-East axis. IAU products use the opposite Stokes `U` sign;
confirm and convert the data convention before constructing `psi`.

To overlay the invalid region of a binary allowed-pixel mask:

```python
mollweide(
    allowed_mask,
    ax=fig.axes[0],
    plot_mode="overlay_mask",
    overlay_color="black",
    alpha=0.35,
)
```

## WCS-backed arrays

Pass a 2D array and its WCS with `wcs=`. SkyPlot normally identifies the
longitude and latitude world axes from WCS metadata. If the WCS has ambiguous
or additional world axes, set `world_axis_mapping=(longitude_axis,
latitude_axis)` using **zero-based WCS world-axis indices**, not NumPy array
axes. Slice a coupled data cube and its WCS to the intended two-dimensional
plane before plotting.

```python
fig = mollweide(
    wcs_map,
    wcs=wcs,
    world_axis_mapping=(1, 0),  # e.g. native world order (Dec, RA)
)
```

## Styling

SkyPlot returns standard Matplotlib figures, so use Matplotlib configuration
and normal artist customization. For example, a local dark style can be
applied with `plt.rc_context(...)`; Cartopy map boundaries use the separate
`"geo"` spine and can be styled through `fig.axes[0].spines["geo"]`.
See the [customization guide](https://skyplot.readthedocs.io/en/latest/usage.html)
for a complete example, including colorbar styling.

## Development

Run the test suite from a source checkout:

```console
python -m pytest
```

The documentation source is in `docs/`; build it after installing the `docs`
extra with:

```console
python -m sphinx -b html docs docs/_build/html
```

SkyPlot is licensed under GPL-3.0-only.
