#######################################################################
# This file is a part of SkyPlot
#
# SkyPlot
# Copyright (C) 2026  Shamik Ghosh
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# For more information about SkyPlot please visit 
# <https://github.com/1cosmologist/skyplot> or contact Shamik Ghosh 
# at shamik@lbl.gov
#
#########################################################################

"""Skyplot package for HEALPix-based astrophysical sky-map visualization."""

from .fonts import configure_default_sans_serif, register_package_fonts, setup_package_fonts
from .normalization import PlanckLogNorm
from .plotting import (
    AVAILABLE_PROJECTIONS,
    DPI_PRESETS,
    FONT_SIZE_PRESETS,
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
    "DPI_PRESETS",
    "FONT_SIZE_PRESETS",
    "PlanckLogNorm",
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
