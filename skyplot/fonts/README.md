# Skyplot Font Bundle

Place your distributable font files in this directory to bundle them with the package.

Supported file types loaded at runtime:
- `.ttf`
- `.otf`
- `.ttc`

When `skyplot` is imported, bundled fonts are registered with Matplotlib and prepended to `font.sans-serif`, making this bundle the default sans-serif fontset.

Important:
- Only include fonts you are licensed to redistribute.
- Prefer open-licensed fonts (e.g., OFL) for package distribution.
