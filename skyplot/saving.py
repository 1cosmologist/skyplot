"""Figure export utilities for skyplot (Matplotlib backend)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from matplotlib.figure import Figure

_FORMATS = {"png", "jpg", "jpeg", "svg", "pdf", "eps"}


def save_figure(
    fig: Figure | None = None,
    output_path: str | Path | None = None,
    *,
    output_format: Literal["png", "jpg", "jpeg", "svg", "pdf", "eps"] | None = None,
    width: int | None = None,
    height: int | None = None,
    figsize: tuple[float, float] = (12.0, 6.0),
    dpi: int = 300,
    scale: float = 1.0,
) -> Path:
    """Save a Matplotlib figure to a static image format.

    If ``fig`` is omitted, the most recently created skyplot figure is used.
    """
    # Allow save_figure("path.png") by shifting a path passed as `fig`.
    if isinstance(fig, (str, Path)):
        if output_path is not None:
            raise TypeError("output_path was passed both as `fig` and `output_path`.")
        fig, output_path = None, fig

    if output_path is None:
        raise ValueError("output_path is required.")

    if fig is None:
        from .plotting import _get_last_figure

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

    fmt = output_format.lower() if output_format is not None else None
    if fmt is None:
        suffix = out.suffix.lower().lstrip(".")
        if not suffix:
            raise ValueError("No output format provided. Use a file suffix or output_format.")
        fmt = suffix
    elif out.suffix == "":
        out = out.with_suffix(f".{fmt}")

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
