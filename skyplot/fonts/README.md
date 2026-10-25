# Skyplot Font Bundle

Place your distributable font files in this directory to bundle them with the package.

Supported file types loaded at runtime:
- `.ttf`
- `.otf`
- `.ttc`

When `skyplot` is imported, the bundled fonts are registered with Matplotlib and
the `modern` theme selects TeX Gyre Heros for text and Noto Sans Math for
mathtext. The `classic` and `academic` themes use EB Garamond/Garamond-Math
and STIX Two Text/STIX Two Math, respectively. Each also has a `_dark` variant.
Mathtext uses Matplotlib's `stixsans`, `cm`, and `stix` fallbacks for the modern,
classic, and academic themes, respectively.
Theme font names are read from the installed font files, and those files take
priority over system fonts with the same names. No OS font installation is needed.

Use `skyplot.set_theme(name)` to change global Matplotlib settings. Themes also
set a 15-color xkcd line cycle. Map gridlines match the theme's foreground
color by default; use `gridline_kwargs` to choose another color for a map.
The default map colormaps are `balance` (modern), `w_ymiddle1` (classic), and
`planck` (academic), also for their dark variants. Pass `cmap` to override them.

Important:
- Only include fonts you are licensed to redistribute.
- Prefer open-licensed fonts (e.g., OFL) for package distribution.
