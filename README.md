# SkyPlot

`SkyPlot` is a Python package for visualizing CMB and other astrophysical sky maps stored as HEALPix arrays.
It samples maps with `healpy` at configurable angular coordinates and renders them with Matplotlib + Cartopy using `pcolormesh`.

## Features

- HEALPix sampling at arbitrary `theta`/`phi` values
- Input flexibility: 1D HEALPix arrays, 2D WCS arrays (`wcs=`), or ndmap-like objects with `.wcs`
- Full-sky grid sampling with explicit `low`/`medium`/`high` resolution presets
- Cartopy-powered `pcolormesh` rendering (true filled-cell map)
- Supported projections include `mollweide`, `orthographic`, `platecarree`, `equidistantconic`
- Uses geographic transforms (`PlateCarree`) so full-sky data is projected consistently
- `cmap` accepts Matplotlib colormap names and names from the `colormaps` package
- HTML/JSON export plus static export (`png`, `jpg`, `webp`, `svg`, `pdf`, `eps`)
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

save_figure(fig, "cmb_map.html")
save_figure(fig, "cmb_map.png", figsize=(12, 6), dpi=300)
save_figure(fig, "cmb_map_data", output_format="json")

# Preset map densities (n_theta, n_phi)
# low: (480, 960), medium: (720, 1440), high: (1440, 2880)
```

## Development

```bash
pip install -e .[dev,docs]
pytest
```

## Custom fontset packaging

To ship a custom sans-serif fontset with this package:

1. Add your distributable font files (`.ttf`, `.otf`, `.ttc`) to `skyplot/fonts/`.
2. Reinstall the package (`pip install -e .` for editable installs).
3. Import `skyplot`; bundled fonts are auto-registered and prepended to Matplotlib `font.sans-serif`.

You can also call these helpers explicitly:

```python
from skyplot import register_package_fonts, configure_default_sans_serif

register_package_fonts()
configure_default_sans_serif()
```

## Documentation

Sphinx documentation is in the `docs/` directory and configured for Read the Docs using `.readthedocs.yaml`.
