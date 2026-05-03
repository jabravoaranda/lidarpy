"""
function needed to create plot with GFAT formatting
"""

import os
from pathlib import Path
from cycler import cycler

import matplotlib as mpl

# mpl.use('Agg')
import numpy as np
import datetime as dt
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.axes import Axes
from matplotlib.offsetbox import AnchoredOffsetbox, OffsetImage


BASE_DIR = os.path.dirname(__file__)

COEFF = 2.0


COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

plt.rcParams["axes.prop_cycle"] = cycler("color", COLORS)

plt.rcParams["xtick.major.pad"] = 1.5 * COEFF
plt.rcParams["xtick.minor.pad"] = 1.5 * COEFF
plt.rcParams["ytick.major.pad"] = 1.5 * COEFF
plt.rcParams["ytick.minor.pad"] = 1.5 * COEFF

plt.rcParams["xtick.major.size"] = 1.0 * COEFF
plt.rcParams["xtick.minor.size"] = 1.0 * COEFF
plt.rcParams["ytick.major.size"] = 1.0 * COEFF
plt.rcParams["ytick.minor.size"] = 1.0 * COEFF

plt.rcParams["xtick.labelsize"] = 5 * COEFF
plt.rcParams["ytick.labelsize"] = 5 * COEFF

plt.rcParams["axes.linewidth"] = 0.5 * COEFF
plt.rcParams["axes.labelsize"] = 5 * COEFF
plt.rcParams["axes.facecolor"] = "#c7c7c7"

plt.rcParams["legend.numpoints"] = 3
plt.rcParams["legend.fontsize"] = 3.5 * COEFF
plt.rcParams["legend.facecolor"] = "#ffffff"

plt.rcParams["grid.linestyle"] = ":"
plt.rcParams["grid.linewidth"] = 0.5 * COEFF
plt.rcParams["grid.alpha"] = 0.5


plt.rcParams["figure.subplot.hspace"] = 0.2
plt.rcParams["figure.subplot.wspace"] = 0.2
plt.rcParams["figure.subplot.bottom"] = 0.11
plt.rcParams["figure.subplot.left"] = 0.14
plt.rcParams["figure.subplot.right"] = 0.95
plt.rcParams["figure.subplot.top"] = 0.82

plt.rcParams["figure.figsize"] = 2.913 * COEFF, 2.047 * COEFF
plt.rcParams["figure.facecolor"] = "#ffffff"

plt.rcParams["lines.markersize"] = 2.6 * COEFF
plt.rcParams["lines.markeredgewidth"] = 0.5 * COEFF
plt.rcParams["lines.linewidth"] = 0.5 * COEFF


def title1(mytitle, coef):
    """
    inclus le titre au document.
        @param mytitle: titre du document.
        @param coef : coefficient GFAT (renvoye par la fonction formatGFAT).
    """

    plt.figtext(
        0.5,
        0.95,
        mytitle,
        fontsize=6.5 * coef,
        fontweight="bold",
        horizontalalignment="center",
        verticalalignment="center",
    )
    return


def title2(mytitle, coef):
    """
    inclus le sous titre au document.
        @param mytitle: titre du document.
        @param coef : coefficient GFAT (renvoye par la fonction formatGFAT).
    """

    plt.figtext(
        0.5,
        0.89,
        mytitle,
        fontsize=5.5 * coef,
        horizontalalignment="center",
        verticalalignment="center",
    )
    return


def title3(mytitle, coef):
    """
    inclus le sous sous titre au document.
        @param mytitle: titre du document.
        @param coef : coefficient GFAT (renvoye par la fonction formatGFAT).
    """
    plt.figtext(
        0.5,
        0.85,
        mytitle,
        fontsize=4.5 * coef,
        horizontalalignment="center",
        verticalalignment="center",
    )
    return


def watermark(
    ax: Axes | np.ndarray = plt.gca(),
    zoom: float = 0.5,
    alpha=0.25,
    xpos=65,
    ypos=315,
    logofile="GFAT",
) -> None:
    """Place watermark in bottom right of figure.
    fig: figure handle
    ax: axes handle
    alpha: alpha channel, ie transparency
    xpos: horizontal location of the figure in pixel
    ypos: vertical location of the figure in pixel
    logofile: file path of the image to use as logo. Default: 'GFAT' redirects to the GFAT logo.
    """

    # Get the pixel dimensions of the figure
    # Import logo and scale accordingly
    if logofile == "GFAT":
        file_path = Path(__file__).parent.parent / "assets" / "LOGO_GFAT_150pp"
    else:
        raise NotImplementedError(f"{logofile} name not implemented")

    img = plt.imread(file_path, format='png')  # type: ignore
    image_box = OffsetImage(img, alpha=alpha, zoom=zoom)
    ao = AnchoredOffsetbox(
        "upper left", pad=0, borderpad=0.5, child=image_box, frameon=False
    )

    if isinstance(ax, Axes):
        ax.add_artist(ao)
    elif isinstance(ax, np.ndarray):
        _ax: Axes
        for _ax in ax.flat:
            _ax.add_artist(ao)

    # wm_width = int(width / scale)  # make the watermark 1/10 of the figure size
    # scaling = wm_width / float(img.size[0])
    # wm_height = int(float(img.size[1]) * float(scaling))
    # img = img.resize((wm_width, wm_height), Image.Resampling.LANCZOS)

    # # Place the watermark in the lower right of the figure
    # plt.figimage(img, xpos, ypos, alpha=alpha, zorder=1)


def tick():
    matplotlib.ticker.FuncFormatter(tmp_f)


def tmp_f(date_dt):
    """
    convert datetime to numerical date
    """
    return mdates.num2date(date_dt).strftime("%H")


def color_list(n, cmap="jet"):
    import matplotlib.pylab as pl
    #Get cmap from plt.cm
    cmap_ = getattr(pl.cm, cmap)
    #Get colors from cmap
    colors = cmap_(np.linspace(0, 1, n))
    # colors = pl.cm.jet(np.linspace(0, 1, n))
    return colors


def font_axes(ax, fontsize=14):
    for item in (
        [ax.title, ax.xaxis.label, ax.yaxis.label]
        + ax.get_xticklabels()
        + ax.get_yticklabels()
    ):
        item.set_fontsize(fontsize)


def gapsizer(ax, time, range, gapsize, colour="#c7c7c7"):
    """
    This function creates a rectangle of color 'colour' when time gap
    are found in the array 'time'.
    """
    # search for holes in data
    # --------------------------------------------------------------------
    dif_time = time[1:] - time[0:-1]
    for index, delta in enumerate(dif_time):
        if delta > dt.timedelta(minutes=gapsize):
            # missing hide bad data
            start = mdates.date2num(time[index])
            end = mdates.date2num(time[index + 1])
            width = end - start

            # Plot rectangle
            end = mdates.date2num(time[index + 1])
            rect = mpl.patches.Rectangle(
                (start, 0), width, np.nanmax(range), color=colour
            )
            ax.add_patch(rect)

def apply_gap_size(ax: matplotlib.axes.Axes, data_array) -> None:  # type: ignore
    """Apply gap size to the x-axis of the plot.

    Args:
        ax (matplotlib.axes.Axes): Axes object.
        data_array (_type_): Data array.
    """
    diff = data_array.time[1:].values - data_array.time[0:-1].values
    gap_size = 2 * int(
        np.ceil(
            np.median(np.median(diff).astype("timedelta64[s]").astype("float") / 60)
        )
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))  # Formato de hora: minutos

    gapsizer(
        ax,
        data_array.time.values.astype("M8[ms]").astype("O"),
        data_array.range.values,
        gap_size,
        "#c7c7c7",
    )

class OOMFormatter(mpl.ticker.ScalarFormatter):
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        mpl.ticker.ScalarFormatter.__init__(
            self, useOffset=offset, useMathText=mathText
        )

    def _set_orderOfMagnitude(self, nothing):
        self.orderOfMagnitude = self.oom

    def _set_format(self, vmin, vmax):
        self.format = self.fformat
        if self._useMathText:
            self.format = "$%s$" % mpl.ticker._mathdefault(self.format)
