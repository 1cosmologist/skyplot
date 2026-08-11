"""Skyplot package for HEALPix-based astrophysical sky-map visualization."""

from .fonts import configure_default_sans_serif, register_package_fonts, setup_package_fonts
from .plotting import (
    AVAILABLE_PROJECTIONS,
    RESOLUTION_PRESETS,
    add_gridlines,
    equidistantconic,
    gnomonic,
    mollweide,
    orthographic,
    platecarree,
)
from .sampling import make_theta_phi_grid, sample_at_angles, sample_full_sky
from .saving import save_figure

# Apply bundled fonts as default sans-serif family when skyplot is imported.
setup_package_fonts()

__all__ = [
    "AVAILABLE_PROJECTIONS",
    "RESOLUTION_PRESETS",
    "add_gridlines",
    "configure_default_sans_serif",
    "equidistantconic",
    "gnomonic",
    "make_theta_phi_grid",
    "mollweide",
    "orthographic",
    "platecarree",
    "register_package_fonts",
    "sample_at_angles",
    "sample_full_sky",
    "save_figure",
    "setup_package_fonts",
]
