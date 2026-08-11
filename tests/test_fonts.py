"""Tests for skyplot bundled font setup."""

import matplotlib

matplotlib.use("Agg")

from skyplot.fonts import configure_default_sans_serif, register_package_fonts


def test_register_package_fonts_returns_list() -> None:
    names = register_package_fonts()
    assert isinstance(names, list)


def test_configure_default_sans_serif_keeps_sans_family() -> None:
    names = configure_default_sans_serif()
    assert isinstance(names, list)
    family = matplotlib.rcParams.get("font.family", [])
    assert isinstance(family, list)
    assert "sans-serif" in family


def test_texgyreheros_is_default_when_available() -> None:
    names = configure_default_sans_serif()
    if "TeX Gyre Heros" not in names:
        return

    sans = matplotlib.rcParams.get("font.sans-serif", [])
    family = matplotlib.rcParams.get("font.family", [])

    assert isinstance(sans, list)
    assert isinstance(family, list)
    assert sans[0] == "TeX Gyre Heros"
    assert family[0] == "TeX Gyre Heros"
