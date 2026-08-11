"""Figure export utilities for skyplot (Matplotlib backend)."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Literal

from matplotlib.figure import Figure

_INTERACTIVE_FORMATS = {"html", "json"}
_STATIC_FORMATS = {"png", "jpg", "jpeg", "webp", "svg", "pdf", "eps"}


def save_figure(
    fig: Figure,
    output_path: str | Path,
    *,
    output_format: Literal[
        "html", "json", "png", "jpg", "jpeg", "webp", "svg", "pdf", "eps"
    ]
    | None = None,
    width: int | None = None,
    height: int | None = None,
    figsize: tuple[float, float] = (12.0, 6.0),
    dpi: int = 300,
    scale: float = 1.0,
    auto_open: bool = False,
) -> Path:
    """Save a Matplotlib figure to HTML, JSON, or static image formats.
    """
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

    if fmt == "json":
        payload = getattr(fig, "_skyplot_payload", None)
        if payload is None:
            payload = {
                "backend": "matplotlib",
                "axes": len(fig.axes),
                "size_inches": list(fig.get_size_inches()),
                "dpi": float(fig.dpi),
            }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out

    if fmt == "html":
        # Emit a simple standalone HTML page with embedded PNG.
        img_buf = io.BytesIO()
        export_dpi = int(round(dpi * scale))
        fig.set_size_inches(figsize[0], figsize[1], forward=True)
        fig.savefig(img_buf, format="png", dpi=export_dpi, bbox_inches="tight")
        img_b64 = base64.b64encode(img_buf.getvalue()).decode("ascii")
        html = (
            "<!doctype html>\n"
            "<html><head><meta charset='utf-8'><title>skyplot export</title></head>\n"
            "<body style='margin:0;background:#ffffff;'>\n"
            f"<img alt='skyplot figure' style='display:block;max-width:100%;height:auto;margin:0 auto;' src='data:image/png;base64,{img_b64}'/>\n"
            "</body></html>\n"
        )
        out.write_text(html, encoding="utf-8")
        if auto_open:
            import webbrowser

            webbrowser.open(out.resolve().as_uri())
        return out

    if fmt in _STATIC_FORMATS:
        export_dpi = int(round(dpi * scale))

        if width is not None or height is not None:
            if width is None or height is None:
                raise ValueError("Both width and height must be provided together.")
            fig.set_size_inches(width / export_dpi, height / export_dpi, forward=True)
        else:
            fig.set_size_inches(figsize[0], figsize[1], forward=True)

        fig.savefig(out, format=fmt, dpi=export_dpi, bbox_inches="tight")
        return out

    supported = sorted(_INTERACTIVE_FORMATS | _STATIC_FORMATS)
    raise ValueError(
        f"Unsupported output format '{fmt}'. Supported formats: {', '.join(supported)}"
    )
