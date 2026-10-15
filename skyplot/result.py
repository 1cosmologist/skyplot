"""Result returned by SkyPlot's public plotting functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure


@dataclass(frozen=True)
class PlotResult:
    """A plot's figure and map axes without a second notebook image display.

    The figure stays registered with pyplot. In a notebook, the inline backend
    displays it at the end of the cell; the result itself has no image display
    hook. Use ``figure`` for Matplotlib methods and ``ax`` for overlays.
    """

    figure: Figure
    ax: Axes

    @property
    def axes(self) -> list[Axes]:
        """The figure's axes, including any colorbar axes."""
        return self.figure.axes

    def __getattr__(self, name: str) -> Any:
        # Keep common Figure-based code working, but never inherit Figure's
        # IPython display methods: the inline backend already shows the figure.
        if name.startswith("_repr_") or name == "_ipython_display_":
            raise AttributeError(name)
        return getattr(self.figure, name)
