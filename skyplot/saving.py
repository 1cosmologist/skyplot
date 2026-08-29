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

"""Figure export utilities for skyplot (Matplotlib backend)."""

from __future__ import annotations

from pathlib import Path

from matplotlib.figure import Figure

_FORMATS = {"png", "jpg", "jpeg", "svg", "pdf", "eps"}


def save_figure(
    fig: Figure | None = None,
    output_path: str | Path | None = None,
    *,
    width: int | None = None,
    height: int | None = None,
    figsize: tuple[float, float] = (8.0, 5.0),
    dpi: int = 300,
    scale: float = 1.0,
) -> Path:
    """Save a Matplotlib figure to a static image format selected by suffix.

    If ``fig`` is omitted, the most recently created skyplot figure is used.
    A path without a suffix is saved as PNG with ``.png`` appended.
    """
    # Allow save_figure("path.png") by shifting a path passed as `fig`.
    if isinstance(fig, (str, Path)):
        if output_path is not None:
            raise TypeError("output_path was passed both as `fig` and `output_path`.")
        fig, output_path = None, fig

    if output_path is None:
        raise ValueError("output_path is required.")

    if fig is None:
        from .plotlib import _get_last_figure

        fig = _get_last_figure()
        if fig is None:
            raise ValueError("No figure provided and no skyplot figure has been created yet.")

    out = Path(output_path)

    if dpi <= 0:
        raise ValueError("dpi must be a positive integer.")
    if scale <= 0.0:
        raise ValueError("scale must be a positive value.")
    if len(figsize) != 2 or figsize[0] <= 0.0 or figsize[1] <= 0.0:
        raise ValueError("figsize must be a two-element tuple of positive values.")

    fmt = out.suffix.lower().lstrip(".")
    if not fmt:
        fmt = "png"
        out = out.with_suffix(".png")

    out.parent.mkdir(parents=True, exist_ok=True)

    if fmt in _FORMATS:
        export_dpi = int(round(dpi * scale))

        if width is not None or height is not None:
            if width is None or height is None:
                raise ValueError("Both width and height must be provided together.")
            fig.set_size_inches(width / export_dpi, height / export_dpi, forward=True)
        else:
            fig.set_size_inches(figsize[0], figsize[1], forward=True)

        fig.savefig(out, format=fmt, dpi=export_dpi, bbox_inches="tight")
        return out

    supported = sorted(_FORMATS)
    raise ValueError(
        f"Unsupported output format '{fmt}'. Supported formats: {', '.join(supported)}"
    )
