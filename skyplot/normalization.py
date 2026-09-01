"""Matplotlib normalizations provided by SkyPlot."""

from __future__ import annotations

import matplotlib.colors as mcolors
import numpy as np

_LN10 = np.log(10.0)


def PlanckLogNorm(
    vmin: float | None = None,
    vmax: float | None = None,
    linthresh: float = 10.0,
) -> mcolors.FuncNorm:
    """Return the Planck-style symmetric logarithmic normalization.

    The forward transform is ``arcsinh(0.5 * value / linthresh) / ln(10)``
    and its analytic inverse. It is linear close to zero and becomes
    logarithmic at larger absolute values. When either limit is omitted,
    Matplotlib derives it from the first plotted map, as for its built-in
    normalizations.

    Parameters
    ----------
    vmin, vmax : float or None, defaults=None, None
        Optional data limits passed to :class:`matplotlib.colors.FuncNorm`.
        An omitted limit is derived from the plotted map.
    linthresh : float, default=10.0
        Scale of the linear region around zero. It must be finite and positive.

    Returns
    -------
    matplotlib.colors.FuncNorm
        A Matplotlib normalization suitable for a plotting function's
        ``norm=`` argument.
    """

    linthresh = float(linthresh)
    if not np.isfinite(linthresh) or linthresh <= 0.0:
        raise ValueError("linthresh must be a finite positive number.")

    def symlog_forward(values: object) -> np.ndarray:
        return np.arcsinh(
            0.5 * np.asarray(values, dtype=float) / linthresh
        ) / _LN10

    def symlog_backward(values: object) -> np.ndarray:
        return 2.0 * linthresh * np.sinh(
            np.asarray(values, dtype=float) * _LN10
        )

    return mcolors.FuncNorm(
        (symlog_forward, symlog_backward), vmin=vmin, vmax=vmax
    )


__all__ = ["PlanckLogNorm"]
