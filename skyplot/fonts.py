"""Font registration helpers for bundled skyplot font sets."""

from __future__ import annotations

from importlib.resources import as_file, files
from typing import Iterable

from matplotlib import font_manager, rcParams

_FONT_EXTENSIONS = {".ttf", ".otf", ".ttc"}
_PREFERRED_SANS_FONTS = (
    "TeX Gyre Heros",
    "TeX Gyre Heros Cn",
)


def _iter_packaged_font_files() -> Iterable[object]:
    """Yield font resources found in the packaged ``skyplot/fonts`` directory."""
    try:
        font_dir = files("skyplot").joinpath("fonts")
    except Exception:
        return []

    if not font_dir.is_dir():
        return []

    font_files = []
    for entry in font_dir.iterdir():
        if entry.is_file() and entry.suffix.lower() in _FONT_EXTENSIONS:
            font_files.append(entry)
    return font_files


def register_package_fonts() -> list[str]:
    """Register all packaged skyplot fonts with Matplotlib.

    Returns
    -------
    list[str]
        Registered font family names discovered from bundled font files.
    """
    registered_names: list[str] = []

    for resource in _iter_packaged_font_files():
        try:
            with as_file(resource) as font_path:
                font_manager.fontManager.addfont(str(font_path))
                family_name = font_manager.FontProperties(fname=str(font_path)).get_name()
        except Exception:
            continue

        if family_name and family_name not in registered_names:
            registered_names.append(family_name)

    return registered_names


def configure_default_sans_serif() -> list[str]:
    """Register packaged fonts and make them the default Matplotlib sans-serif family.

    Returns
    -------
    list[str]
        Bundled font family names that were registered.
    """
    registered_names = register_package_fonts()
    if not registered_names:
        return []

    existing = list(rcParams.get("font.sans-serif", []))

    preferred = [name for name in _PREFERRED_SANS_FONTS if name in registered_names]
    ordered_names = [*preferred, *registered_names]

    merged: list[str] = []
    for name in [*ordered_names, *existing]:
        if name and name not in merged:
            merged.append(name)

    if preferred:
        # Make TeX Gyre Heros the global default family while keeping sans fallback.
        rcParams["font.family"] = [preferred[0], "sans-serif"]
    else:
        rcParams["font.family"] = ["sans-serif"]
    rcParams["font.sans-serif"] = merged
    return registered_names


def setup_package_fonts() -> list[str]:
    """Public one-call setup for bundled font registration and defaulting."""
    return configure_default_sans_serif()
