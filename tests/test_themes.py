"""Checks for bundled theme fonts, colors, and rendered Matplotlib artists."""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import healpy as hp
import numpy as np
import pytest
from matplotlib import font_manager
from matplotlib.colors import to_rgb, to_rgba

import skyplot
from skyplot import DARK_COLORS, LIGHT_COLORS, THEMES, gnomonic, mollweide, set_theme
from skyplot.themes import register_package_fonts

_BUNDLED_FONTS = Path(skyplot.__file__).resolve().parent / "fonts"


def _contrast_ratio(color: str, background: str) -> float:
    def luminance(value: str) -> float:
        rgb = np.asarray(to_rgb(value))
        linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        return float(np.dot(linear, [0.2126, 0.7152, 0.0722]))

    first, second = sorted((luminance(color), luminance(background)))
    return (second + 0.05) / (first + 0.05)


@pytest.mark.parametrize(
    ("name", "family", "text_font", "math_font", "bold_family", "math_fallback"),
    [
        ("modern", "sans-serif", "TeX Gyre Heros", "Noto Sans Math", "TeX Gyre Heros", "stixsans"),
        ("classic", "serif", "EB Garamond", "Garamond-Math", "EB Garamond", "cm"),
        ("academic", "serif", "STIX Two Text", "STIX Two Math", "STIX Two Text", "stix"),
    ],
)
def test_theme_font_pairs(name, family, text_font, math_font, bold_family, math_fallback) -> None:
    names = register_package_fonts()
    assert text_font in names
    assert math_font in names

    for suffix in ("", "_dark"):
        with matplotlib.rc_context():
            set_theme(name + suffix)
            rc = matplotlib.rcParams
            assert rc["font.family"] == [family]
            assert rc[f"font.{family}"] == [text_font]
            assert rc["mathtext.fontset"] == "custom"
            assert rc["mathtext.fallback"] == math_fallback
            assert rc["mathtext.rm"].replace(r"\-", "-") == math_font
            assert Path(font_manager.findfont(font_manager.FontProperties(
                family=rc["font.family"],
            ))).parent == _BUNDLED_FONTS
            for slot, expected_family, expected_style, minimum_weight in (
                ("rm", math_font, "normal", 400),
                ("it", text_font, "italic", 400),
                ("bf", bold_family, "normal", 600),
                ("bfit", bold_family, "italic", 600),
                ("sf", "TeX Gyre Heros", "normal", 400),
                ("tt", "TeX Gyre Heros", "normal", 400),
            ):
                selected = font_manager.findfont(font_manager.FontProperties(rc[f"mathtext.{slot}"]))
                assert Path(selected).parent == _BUNDLED_FONTS
                assert any(
                    entry.fname == str(selected)
                    and entry.name == expected_family
                    and entry.style == expected_style
                    and entry.weight >= minimum_weight
                    for entry in font_manager.fontManager.ttflist
                ), (slot, selected, [
                    (entry.name, entry.style, entry.weight)
                    for entry in font_manager.fontManager.ttflist if entry.fname == str(selected)
                ])


@pytest.mark.parametrize("name", THEMES)
def test_theme_colors_and_math_render(name) -> None:
    dark = name.endswith("_dark")
    background = "black" if dark else "white"
    foreground = "white" if dark else "black"
    palette = DARK_COLORS if dark else LIGHT_COLORS
    assert len(palette) == len({to_rgba(color) for color in palette}) == 15
    assert [(color, _contrast_ratio(color, background)) for color in palette
            if _contrast_ratio(color, background) < 3] == []

    with matplotlib.rc_context():
        set_theme(name)
        rc = matplotlib.rcParams
        for key in ("figure.facecolor", "axes.facecolor", "savefig.facecolor"):
            assert to_rgba(rc[key]) == to_rgba(background)
        for key in ("text.color", "axes.edgecolor", "axes.labelcolor", "xtick.color", "ytick.color"):
            assert to_rgba(rc[key]) == to_rgba(foreground)
        assert to_rgba(rc["grid.color"]) == to_rgba(foreground)
        assert rc["axes.prop_cycle"].by_key()["color"] == list(palette)

        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        ax.set_title(r"$E=mc^2$ and $\alpha+\beta+\mathit{x}+\mathbf{y}+\mathbfit{z}$")
        ax.set_xlabel("Position")
        colorbar = fig.colorbar(ax.imshow(np.arange(4).reshape(2, 2)), ax=ax)
        colorbar.set_label("Scale")
        fig.canvas.draw()
        assert to_rgba(fig.get_facecolor()) == to_rgba(background)
        assert to_rgba(ax.spines["bottom"].get_edgecolor()) == to_rgba(foreground)
        assert to_rgba(colorbar.outline.get_edgecolor()) == to_rgba(foreground)
        assert to_rgba(colorbar.ax.xaxis.label.get_color()) == to_rgba(foreground)
        plt.close(fig)


def test_set_theme_and_matplotlib_rc_context_restore() -> None:
    with matplotlib.rc_context():
        set_theme("classic_dark")
        assert matplotlib.rcParams["font.serif"][0] == "EB Garamond"
        assert to_rgba(matplotlib.rcParams["figure.facecolor"]) == to_rgba("black")
        with matplotlib.rc_context():
            set_theme("modern")
            assert matplotlib.rcParams["font.sans-serif"][0] == "TeX Gyre Heros"
        assert matplotlib.rcParams["font.serif"][0] == "EB Garamond"
        assert to_rgba(matplotlib.rcParams["figure.facecolor"]) == to_rgba("black")
    with pytest.raises(ValueError, match="Unknown theme"):
        set_theme("unknown")


def test_modern_is_import_default() -> None:
    assert matplotlib.rcParams["font.family"] == ["sans-serif"]
    assert matplotlib.rcParams["font.sans-serif"][0] == "TeX Gyre Heros"
    assert matplotlib.rcParams["mathtext.rm"] == "Noto Sans Math"
    assert matplotlib.rcParams["image.cmap"] == "balance"


@pytest.mark.parametrize(
    ("name", "expected_cmap"),
    [
        ("academic", "planck"),
        ("academic_dark", "planck"),
        ("classic", "w_ymiddle1"),
        ("classic_dark", "w_ymiddle1"),
        ("modern", "balance"),
        ("modern_dark", "balance"),
    ],
)
def test_theme_default_colormap(name, expected_cmap) -> None:
    with matplotlib.rc_context():
        set_theme(name)
        assert matplotlib.rcParams["image.cmap"] == expected_cmap

        fig, ax = plt.subplots()
        assert ax.imshow(np.arange(4).reshape(2, 2)).cmap.name == expected_cmap
        plt.close(fig)

        result = mollweide(
            np.ones(hp.nside2npix(2)), n_theta=8, n_phi=16,
            show_gridlines=False, add_colorbar=False,
        )
        assert result.figure._skyplot_payload["cmap"] == expected_cmap
        assert result.ax.images[0].cmap.name == expected_cmap
        plt.close(result.figure)


def test_explicit_colormap_overrides_theme_default() -> None:
    with matplotlib.rc_context():
        set_theme("academic_dark")
        default = gnomonic(
            np.ones(hp.nside2npix(2)), xsize=16, ysize=16, add_colorbar=False,
        )
        assert default.figure._skyplot_payload["cmap"] == "planck"
        assert default.ax.images[0].cmap.name == "planck"
        plt.close(default.figure)

        result = gnomonic(
            np.ones(hp.nside2npix(2)), xsize=16, ysize=16,
            cmap="viridis", add_colorbar=False,
        )
        assert result.figure._skyplot_payload["cmap"] == "viridis"
        assert result.ax.images[0].cmap.name == "viridis"
        plt.close(result.figure)


def test_dark_skyplot_edges_and_gridlines() -> None:
    with matplotlib.rc_context():
        set_theme("modern_dark")
        result = mollweide(
            np.ones(hp.nside2npix(2)), n_theta=12, n_phi=24, title="Dark map",
        )
        result.figure.canvas.draw()
        colorbar = result.figure.axes[-1]
        assert to_rgba(result.ax.spines["geo"].get_edgecolor()) == to_rgba("white")
        assert to_rgba(matplotlib.rcParams["grid.color"]) == to_rgba("white")
        assert to_rgba(colorbar.spines["outline"].get_edgecolor()) == to_rgba("white")
        assert to_rgba(colorbar.xaxis.label.get_color()) == to_rgba("white")
        assert result.figure._skyplot_payload["gridline_color"] == "white"
        plt.close(result.figure)

        overridden = mollweide(
            np.ones(hp.nside2npix(2)), n_theta=12, n_phi=24,
            gridline_kwargs={"color": "cyan"},
        )
        assert overridden.figure._skyplot_payload["gridline_color"] == "cyan"
        plt.close(overridden.figure)


def test_classic_mixed_text_math_label_has_no_font_fallback(caplog) -> None:
    with matplotlib.rc_context(), caplog.at_level(logging.WARNING, logger="matplotlib.font_manager"):
        set_theme("classic")
        fig, ax = plt.subplots()
        ax.set_xlabel(r"Galactic CO (velocity integrated) in K km s${}^{-1}$")
        fig.canvas.draw()
        plt.close(fig)
    assert "Falling back to DejaVu Sans" not in caplog.text
