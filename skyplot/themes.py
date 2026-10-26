"""Bundled fonts and Matplotlib themes for SkyPlot and other plots."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import matplotlib as mpl
from cycler import cycler
from matplotlib import font_manager


LIGHT_COLORS = (
    "xkcd:blue", "xkcd:golden brown", "xkcd:forest green", "xkcd:crimson",
    "xkcd:purple", "xkcd:teal", "xkcd:rust", "xkcd:magenta",
    "xkcd:dark cyan", "xkcd:olive green", "xkcd:brick red",
    "xkcd:royal blue", "xkcd:burnt orange", "xkcd:dark pink",
    "xkcd:greyish brown",
)

DARK_COLORS = (
    "xkcd:sky blue", "xkcd:golden yellow", "xkcd:bright green",
    "xkcd:coral", "xkcd:lavender", "xkcd:turquoise", "xkcd:salmon",
    "xkcd:light purple", "xkcd:lime green", "xkcd:peach",
    "xkcd:cyan", "xkcd:pink", "xkcd:periwinkle",
    "xkcd:light orange", "xkcd:mint green",
)

_FONT_PAIRS = {
    # Each math font is Regular only; styled math letters use real text faces.
    "modern": (
        "sans-serif", "texgyreheros-regular.otf", "NotoSansMath-Regular.otf",
        "italic", "bold", "bold:italic", "stixsans",
    ),
    "classic": (
        "serif", "EBGaramond-Regular.otf", "Garamond-Math.otf",
        "italic", "weight=semibold", "weight=semibold:style=italic", "cm",
    ),
    "academic": (
        "serif", "STIXTwoText-Regular.otf", "STIXTwoMath-Regular.otf",
        "italic", "weight=semibold", "weight=semibold:style=italic", "stix",
    ),
}

_DEFAULT_CMAPS = {
    "academic": "planck",
    "classic": "w_ymiddle1",
    "modern": "balance",
}

_FONT_DIR = Path(__file__).resolve().parent / "fonts"


@lru_cache(maxsize=None)
def _font_name(filename: str) -> str:
    """Read the family name from a font installed with SkyPlot."""
    return font_manager.FontProperties(fname=_FONT_DIR / filename).get_name()


@lru_cache(maxsize=1)
def register_package_fonts() -> list[str]:
    """Register and prioritize bundled fonts without an OS font installation."""
    names: list[str] = []
    manager = font_manager.fontManager
    for path in sorted(_FONT_DIR.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".otf", ".ttf", ".ttc"}:
            continue
        first_new = len(manager.ttflist)
        manager.addfont(path)
        # Matplotlib may add alternate family names after the main entry.
        # Put every entry from this file before any OS font with the same name.
        entries = manager.ttflist[first_new:]
        del manager.ttflist[first_new:]
        manager.ttflist[:0] = entries
        family = _font_name(path.name)
        if family not in names:
            names.append(family)
    return names


def _theme(
    family: str, text_file: str, math_file: str,
    italic_style: str, bold_style: str, bold_italic_style: str,
    math_fallback: str, *, dark: bool, cmap: str,
) -> dict[str, object]:
    text_font = _font_name(text_file)
    math_font = _font_name(math_file)
    foreground, background = ("white", "black") if dark else ("black", "white")
    family_key = f"font.{family}"
    # Matplotlib parses mathtext font names as fontconfig patterns.
    math_pattern = math_font.replace("-", r"\-")
    return {
        "font.family": [family],
        family_key: [text_font],
        "mathtext.fontset": "custom",
        "mathtext.fallback": math_fallback,
        "mathtext.rm": math_pattern,
        "mathtext.it": f"{text_font}:{italic_style}",
        "mathtext.bf": f"{text_font}:{bold_style}",
        "mathtext.bfit": f"{text_font}:{bold_italic_style}",
        "mathtext.sf": _font_name("texgyreheros-regular.otf"),
        "mathtext.tt": _font_name("texgyreheros-regular.otf"),
        "figure.facecolor": background,
        "figure.edgecolor": foreground,
        "axes.facecolor": background,
        "axes.edgecolor": foreground,
        "axes.labelcolor": foreground,
        "axes.titlecolor": foreground,
        "text.color": foreground,
        "xtick.color": foreground,
        "ytick.color": foreground,
        "xtick.labelcolor": foreground,
        "ytick.labelcolor": foreground,
        "grid.color": foreground,
        "image.cmap": cmap,
        "savefig.facecolor": background,
        "savefig.edgecolor": foreground,
        "legend.facecolor": background,
        "legend.edgecolor": foreground,
        "legend.labelcolor": foreground,
        "axes.prop_cycle": cycler(color=DARK_COLORS if dark else LIGHT_COLORS),
    }


THEMES: dict[str, dict[str, object]] = {
    name + suffix: _theme(*fonts, dark=bool(suffix), cmap=_DEFAULT_CMAPS[name])
    for name, fonts in _FONT_PAIRS.items()
    for suffix in ("", "_dark")
}


def _get_theme(name: str) -> dict[str, object]:
    try:
        return THEMES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown theme {name!r}. Choose one of: {', '.join(THEMES)}") from exc


def set_theme(name: str = "modern") -> None:
    """Apply a theme globally to Matplotlib's rcParams."""
    settings = _get_theme(name)
    register_package_fonts()
    cmap_name = str(settings["image.cmap"])
    if cmap_name not in mpl.colormaps:
        from .plotlib import _resolve_cmap

        resolved = _resolve_cmap(cmap_name)
        if cmap_name not in mpl.colormaps:
            cmap = mpl.colormaps.get_cmap(resolved) if isinstance(resolved, str) else resolved
            mpl.colormaps.register(cmap, name=cmap_name)
    mpl.rcParams.update(settings)
