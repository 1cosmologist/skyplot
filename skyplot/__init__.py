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
from .result import PlotResult
from .sampling import make_theta_phi_grid, sample_at_angles, sample_full_sky
from .themes import DARK_COLORS, LIGHT_COLORS, THEMES, register_package_fonts, set_theme

# Apply the modern theme to SkyPlot and subsequent Matplotlib plots.
set_theme("modern")

__all__ = [
    "AVAILABLE_PROJECTIONS",
    "DARK_COLORS",
    "DPI_PRESETS",
    "FONT_SIZE_PRESETS",
    "LIGHT_COLORS",
    "PlanckLogNorm",
    "PlotResult",
    "RESOLUTION_PRESETS",
    "THEMES",
    "add_gridlines",
    "equidistantconic",
    "gnomonic",
    "make_theta_phi_grid",
    "mollweide",
    "orthographic",
    "platecarree",
    "register_package_fonts",
    "sample_at_angles",
    "sample_full_sky",
    "set_theme",
]
