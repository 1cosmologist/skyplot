import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

font_dirs = '/global/u2/s/shamikg/.local/share/fonts/'
font_files = fm.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    fm.fontManager.addfont(font_file)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"]  = "Myriad Pro"
plt.rcParams['font.weight'] = 'regular'


def _nice_ticks(vmin, vmax, n=6):
    """Return ~n evenly-spaced, rounded tick values spanning [vmin, vmax]."""
    span = vmax - vmin
    if span == 0:
        return np.array([vmin])
    raw_step = span / n
    mag = 10 ** np.floor(np.log10(abs(raw_step)))
    step = mag
    for mult in [1, 2, 2.5, 5, 10]:
        step = mag * mult
        if span / step <= n + 1:
            break
    first = np.ceil(vmin  / step) * step
    last  = np.floor(vmax / step) * step
    return np.arange(first, last + step * 1e-6, step)


def _nice_angular_step(n_ticks, span=180.0):
    """Round span/n_ticks to the nearest astronomically sensible degree step."""
    raw = span / max(n_ticks, 1)
    candidates = [1, 2, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180]
    return min(candidates, key=lambda s: abs(s - raw))


def imshow_wcs(data, wcs, ax=None, figsize=(15, 5), dpi=200,
               n_ra_ticks=12, n_dec_ticks=5,
               ra_fmt='{:.0f}°', dec_fmt='{:.1f}°',
               ra_range=None, dec_range=None,
               grid=True, grid_kwargs=None,
               **imshow_kwargs):
    """
    Display a 2D image with RA/Dec axis labels computed directly from WCS,
    without using astropy WCSAxes.

    For full-sky CAR maps where RA=0 is at the image center, the x-axis shows
    0→180 on the left half and 180→360 on the right (360° omitted = same as 0°).

    Parameters
    ----------
    data         : 2D ndarray
    wcs          : astropy WCS (2-axis celestial, as in a pixell enmap)
    ax           : existing Axes, or None to create a new figure
    figsize, dpi : passed to plt.subplots when ax is None
    n_ra_ticks   : approx number of RA ticks per half for full-sky, or total for zoom
    n_dec_ticks  : approx number of Dec ticks
    ra_fmt       : format string or 1-arg callable for RA tick labels
    dec_fmt      : format string or 1-arg callable for Dec tick labels
    ra_range     : (ra_min, ra_max) in degrees, [-180, 180] convention
                   (negative values map to the 180–360 display range)
    dec_range    : (dec_min, dec_max) in degrees
    grid         : bool, whether to draw coordinate gridlines (default True)
    grid_kwargs  : dict of kwargs forwarded to ax.grid(), e.g.
                   {'color': 'k', 'ls': '--', 'lw': 0.5, 'alpha': 0.4}
    **imshow_kwargs : forwarded to ax.imshow (e.g. vmin, vmax, cmap)

    Returns
    -------
    fig, ax, im
    """
    nrows, ncols = data.shape

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    imshow_kwargs.setdefault('origin', 'lower')
    im = ax.imshow(data, **imshow_kwargs)

    def _label(val, fmt):
        return fmt(val) if callable(fmt) else fmt.format(val)

    def pix2world(col, row):
        return wcs.all_pix2world([[col, row]], 0)[0]   # → (ra, dec)

    def world2pix(ra, dec):
        return wcs.all_world2pix([[ra, dec]], 0)[0]    # → (col, row)

    # ------------------------------------------------------------------
    # RA ticks (x-axis)
    # ------------------------------------------------------------------
    mid_row = nrows // 2
    ra_left,  _       = pix2world(0,         mid_row)
    ra_right, _       = pix2world(ncols - 1, mid_row)
    _,        mid_dec = pix2world(ncols // 2, mid_row)

    is_wrapped = abs(ra_left - 180) < 10 and abs(ra_right - 180) < 10

    if ra_range is not None:
        ra_lo, ra_hi = min(ra_range), max(ra_range)
        ra_tick_vals = _nice_ticks(ra_lo, ra_hi, n=n_ra_ticks)
        ra_tick_pix  = [world2pix(rv, mid_dec)[0] for rv in ra_tick_vals]
        ra_tick_disp = [rv % 360 for rv in ra_tick_vals]

        col_a = world2pix(ra_lo, mid_dec)[0]
        col_b = world2pix(ra_hi, mid_dec)[0]
        ax.set_xlim(min(col_a, col_b) - 0.5, max(col_a, col_b) + 0.5)
        ax.set_xticks(ra_tick_pix)
        ax.set_xticklabels([_label(r, ra_fmt) for r in ra_tick_disp])

    elif is_wrapped:
        step = _nice_angular_step(n_ra_ticks // 2, span=180.0)
        left_ra  = np.arange(0, 181, step)
        left_pix = np.array([world2pix(ra, mid_dec)[0] for ra in left_ra])
        col_ra0  = left_pix[0]

        right_ra_display = 360 - left_ra[1:]
        right_pix = np.clip(2 * col_ra0 - left_pix[1:], 0, ncols - 1)

        all_pix     = np.concatenate([left_pix,  right_pix])
        all_display = np.concatenate([left_ra,   right_ra_display])
        order       = np.argsort(all_pix)
        ax.set_xticks(all_pix[order].tolist())
        ax.set_xticklabels([_label(r, ra_fmt) for r in all_display[order]])

    else:
        ra_lo, ra_hi = min(ra_left, ra_right), max(ra_left, ra_right)
        ra_vals = _nice_ticks(ra_lo, ra_hi, n=n_ra_ticks)
        ra_pix  = [world2pix(rv, mid_dec)[0] for rv in ra_vals]
        ax.set_xticks(ra_pix)
        ax.set_xticklabels([_label(r, ra_fmt) for r in ra_vals])

    ax.set_xlabel('Right Ascension')

    # ------------------------------------------------------------------
    # Dec ticks (y-axis)
    # ------------------------------------------------------------------
    mid_col = ncols // 2
    _,      dec_bot = pix2world(mid_col, 0)
    _,      dec_top = pix2world(mid_col, nrows - 1)
    mid_ra, _       = pix2world(mid_col, nrows // 2)

    if dec_range is not None:
        dec_lo, dec_hi = min(dec_range), max(dec_range)
        row_a = world2pix(mid_ra, dec_lo)[1]
        row_b = world2pix(mid_ra, dec_hi)[1]
        ax.set_ylim(min(row_a, row_b) - 0.5, max(row_a, row_b) + 0.5)
        dec_vals = _nice_ticks(dec_lo, dec_hi, n=n_dec_ticks)
    else:
        dec_lo, dec_hi = min(dec_bot, dec_top), max(dec_bot, dec_top)
        dec_vals = _nice_ticks(dec_lo, dec_hi, n=n_dec_ticks)

    dec_pix = [world2pix(mid_ra, dv)[1] for dv in dec_vals]
    ax.set_yticks(dec_pix)
    ax.set_yticklabels([_label(d, dec_fmt) for d in dec_vals])
    ax.set_ylabel('Declination')

    if ra_range is None:
        ax.set_xlim(-0.5, ncols - 0.5)
    if dec_range is None:
        ax.set_ylim(-0.5, nrows - 0.5)

    # ------------------------------------------------------------------
    # Gridlines at tick positions
    # ------------------------------------------------------------------
    if grid:
        _gkw = dict(color='k', ls='--', lw=0.5, alpha=0.4)
        if grid_kwargs:
            _gkw.update(grid_kwargs)
        ax.grid(True, **_gkw)

    return fig, ax, im
