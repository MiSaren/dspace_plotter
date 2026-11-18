"""
Matplotlib styles for screen viewing and LaTeX publications.

Usage
-----
    from plot_style import set_latex_style, set_screen_style, save_and_show
    from plot_style import downsample   # fast drawing of long measurements

    set_latex_style(palette="tol_bright")  # colour cycle by name, see PALETTES
    set_palette(["#000000", "#E69F00"])     # or change it / give own colours
    show_palettes()                         # swatches of all palettes

Modified from motulator.common.utils._plotting: the LaTeX style uses the
Okabe-Ito colour palette by default instead of the matplotlib colours.
"""
from pathlib import Path

import numpy as np
from cycler import cycler
from matplotlib import pyplot as plt

OKABE_ITO = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]

# Colour cycles by name. Paul Tol's palettes: https://personal.sron.nl/~pault/
PALETTES = {
    # Colour-blind safe
    "okabe_ito": OKABE_ITO,
    "tol_bright": [
        "#4477AA",  # blue
        "#EE6677",  # red
        "#228833",  # green
        "#CCBB44",  # yellow
        "#66CCEE",  # cyan
        "#AA3377",  # purple
        "#BBBBBB",  # grey
    ],
    "tol_vibrant": [
        "#0077BB",  # blue
        "#EE7733",  # orange
        "#009988",  # teal
        "#CC3311",  # red
        "#33BBEE",  # cyan
        "#EE3377",  # magenta
        "#BBBBBB",  # grey
    ],
    "tol_muted": [
        "#332288",  # indigo
        "#CC6677",  # rose
        "#117733",  # green
        "#88CCEE",  # cyan
        "#882255",  # wine
        "#44AA99",  # teal
        "#999933",  # olive
        "#AA4499",  # purple
        "#DDCC77",  # sand
    ],
    "tol_high_contrast": [  # also readable in greyscale print
        "#004488",  # blue
        "#BB5566",  # red
        "#DDAA33",  # yellow
        "#000000",  # black
    ],
    "ibm": [
        "#648FFF",  # blue
        "#DC267F",  # magenta
        "#FE6100",  # orange
        "#785EF0",  # purple
        "#FFB000",  # yellow
    ],
    # Not colour-blind safe
    "tab10": [  # matplotlib default
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ],
    "screen": ["b", "r", "m", "c"],
}


def _colors(palette):
    """Colour list of a palette name in PALETTES, or a list of colours as is."""
    if isinstance(palette, str):
        try:
            return PALETTES[palette]
        except KeyError:
            raise KeyError(f"Unknown palette '{palette}', options: "
                           f"{', '.join(PALETTES)}") from None
    return list(palette)


def set_palette(palette) -> None:
    """
    Change the colour cycle of the current style, e.g. after set_latex_style().

    Parameters
    ----------
    palette : str | list
        Name in PALETTES or a list of colours.

    """
    plt.rcParams["axes.prop_cycle"] = cycler("color", _colors(palette))


def show_palettes(names=None):
    """Draw a swatch of each palette (all in PALETTES by default)."""
    names = list(PALETTES) if names is None else names
    fig, ax = plt.subplots(figsize=(6, 0.4*len(names) + 0.4))
    for row, name in enumerate(names):
        for col, c in enumerate(_colors(name)):
            ax.add_patch(plt.Rectangle((col, -row - 0.4), 0.9, 0.8, color=c))
    ax.set_yticks(-np.arange(len(names)), names)
    ax.set_xlim(-0.1, max(len(_colors(n)) for n in names))
    ax.set_ylim(-len(names) + 0.5, 0.5)
    ax.set_xticks([])
    ax.grid(False)
    for side in ax.spines.values():
        side.set_visible(False)
    return fig


def set_screen_style(palette="screen") -> None:
    """Configure matplotlib for screen viewing, palette: see set_palette()."""
    plt.style.use("default")
    # plt.style.use("dark_background")
    plt.rcParams.update(
        {
            "lines.linewidth": 1,
            "axes.grid": True,
            "text.usetex": False,
            "axes3d.grid": True,
            "figure.constrained_layout.use": True,
        }
    )
    set_palette(palette)


def set_latex_style(palette="okabe_ito") -> None:
    """Configure matplotlib for LaTeX documents, palette: see set_palette()."""
    plt.style.use("default")
    plt.rcParams.update(
        {
            "lines.linewidth": 1,
            "axes.grid": True,
            # "axes.autolimit_mode": "round_numbers",
            "text.usetex": True,
            "text.latex.preamble": (r"\usepackage{newtxtext}\usepackage{newtxmath}"),
            "font.family": "serif",
            "font.serif": ["Liberation Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes3d.grid": True,
            "figure.figsize": [3.5, 2.5],
            # "figure.constrained_layout.use": True,
            # "savefig.bbox": "tight",
            # "savefig.pad_inches": 0.01,
            # Disable automatic adjustment
            "figure.constrained_layout.use": False,
            "figure.subplot.left": 0.18,  # Fixed margins
            "figure.subplot.right": 0.95,
            "figure.subplot.bottom": 0.18,
            "figure.subplot.top": 0.95,
        }
    )
    set_palette(palette)


def _minmax(x, y, lo, hi, n):
    """Points of (x, y) in [lo, hi], reduced to the min and max of n bins."""
    i0 = max(np.searchsorted(x, lo) - 1, 0)
    i1 = min(np.searchsorted(x, hi) + 1, x.size)
    x, y = x[i0:i1], y[i0:i1]
    if x.size <= 2*n:
        return x, y
    k = x.size // n  # samples per bin
    m = n*k
    start = np.arange(n)*k
    bins = y[:m].reshape(n, k)
    # NaN (missing signal) is neither min nor max, all-NaN bins give a gap
    i_min = np.argmin(np.where(np.isnan(bins), np.inf, bins), axis=1) + start
    i_max = np.argmax(np.where(np.isnan(bins), -np.inf, bins), axis=1) + start
    idx = np.unique(np.concatenate([i_min, i_max, np.arange(m, x.size)]))
    return x[idx], y[idx]


def downsample(fig=None, n=2000) -> None:
    """
    Draw only the visible part of long lines, reduced to the min and max of
    n bins. Looks the same at screen and print resolution, but pan and zoom
    are fast. Recomputed from the full data when the x-limits change.

    Parameters
    ----------
    fig : Figure, optional
        Figure to downsample. Default is the current figure.
    n : int, optional
        Number of bins over the visible x-range.

    """
    fig = plt.gcf() if fig is None else fig
    for ax in fig.axes:
        lines = []
        for line in ax.get_lines():
            x, y = (np.asarray(v, dtype=float) for v in line.get_data())
            if x.size > 2*n and np.all(np.diff(x) >= 0):
                lines.append((line, x, y))
        if not lines:
            continue

        def update(ax, lines=lines):
            lo, hi = ax.get_xlim()
            for line, x, y in lines:
                line.set_data(*_minmax(x, y, lo, hi, n))

        update(ax)
        ax.callbacks.connect("xlim_changed", update)


def save_and_show(save_path: str | Path | None = None, **savefig_kwargs) -> None:
    """
    Helper function to optionally save and show plots.

    Parameters
    ----------
    save_path : str | Path | None, optional
        Path to save the figure. If None, the figure is not saved.
    savefig_kwargs : dict, optional
        Additional keyword arguments for `plt.savefig`.

    """
    if save_path is not None:
        default_kwargs = {"bbox_inches": "tight"}
        default_kwargs.update(savefig_kwargs)
        plt.savefig(save_path, **default_kwargs)
    plt.show()
