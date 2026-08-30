# SkyPlot

`SkyPlot` is a Python package for visualizing CMB and other astrophysical sky maps stored as HEALPix arrays or `pixell.ndmap`.
It samples maps with `healpy` at configurable angular coordinates and renders them with Matplotlib + Cartopy using `pcolormesh`.

## Features

- HEALPix sampling at arbitrary `theta`/`phi` values
- Input flexibility: 1D HEALPix arrays, 2D WCS arrays (`wcs=`), or ndmap-like objects with `.wcs`
- Full-sky grid sampling with explicit `low`/`medium`/`high` resolution presets
- Cartopy-powered `pcolormesh` rendering (true filled-cell map)
- Transparent streamplot or quiver overlays for vector fields, layered onto a
  separately rendered magnitude map
- Supported projections include `mollweide`, `orthographic`, `platecarree`, `equidistantconic`
- Uses geographic transforms (`PlateCarree`) so full-sky data is projected consistently
- `cmap` accepts Matplotlib colormap names and names from the `colormaps` package
- Multiple export formats (`png`, `jpg`, `svg`, `pdf`, `eps`)
- Bundled fontset support: packaged fonts can become the default sans-serif stack automatically

## Quick start

```python
import healpy as hp
import numpy as np

from skyplot import mollweide, save_figure

nside = 64
npix = hp.nside2npix(nside)
hp_map = np.random.default_rng(1234).normal(size=npix)

fig = mollweide(
    hp_map,
    projection_kwargs={"central_longitude": 120.0},
    cmap="batlow",  # from `colormaps`, or use Matplotlib names like "viridis"
    figsize=(12, 6),
    dpi=300,
    title="Example CMB-like Map",
)
fig.show()

# Display a Galactic map on an ICRS longitude/latitude grid. The transform
# states (source_frame, display_frame), while coordinate_frame is optional
# figure metadata. These keywords are available on non-gnomonic projections.
icrs_fig = mollweide(
    hp_map,
    coordinate_frame="galactic",
    coordinate_transform=("galactic", "icrs"),
)

# WCS maps infer the longitude/latitude world-axis order from their WCS
# metadata, including RA/Dec and Galactic/ecliptic longitude/latitude axes.
# For a WCS with ambiguous metadata, provide the *zero-based WCS world-axis*
# indices as (longitude_axis, latitude_axis). For example, native (Dec, RA)
# world coordinates use (1, 0); these are not NumPy array-axis indices.
wcs_fig = mollweide(
    wcs_map,
    wcs=wcs,
    world_axis_mapping=(1, 0),
)

# healpy.UNSEEN and NaN samples are rendered as grey by default. Customize
# the input sentinel and missing-data color when needed.
masked_fig = mollweide(
    hp_map,
    badvalue=hp.UNSEEN,
    badcolor="lightgrey",
)

save_figure(fig, "cmb_map.png", figsize=(12, 6), dpi=300)

# Preset map densities (n_theta, n_phi)
# low: (480, 960), medium: (720, 1440), high: (1440, 2880)
```

## Development

```bash
pip install -e .[dev,docs]
pytest
```

<!-- ## Custom fontset packaging

To ship a custom sans-serif fontset with this package:

1. Add your distributable font files (`.ttf`, `.otf`, `.ttc`) to `skyplot/fonts/`.
2. Reinstall the package (`pip install -e .` for editable installs).
3. Import `skyplot`; bundled fonts are auto-registered and prepended to Matplotlib `font.sans-serif`.

You can also call these helpers explicitly:

```python
from skyplot import register_package_fonts, configure_default_sans_serif

register_package_fonts()
configure_default_sans_serif()
``` -->

## Documentation

Sphinx documentation is in the `docs/` directory and configured for Read the Docs using `.readthedocs.yaml`.
