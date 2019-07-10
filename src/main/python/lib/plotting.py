import multiprocessing

multiprocessing.freeze_support()

from global_variables import GlobalVariables as gvars
import matplotlib

from lib.math import estimate_bw

matplotlib.use("qt5agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
from matplotlib.collections import PatchCollection
from matplotlib.patches import Patch
import sklearn.neighbors
import lib.misc
import lib.math


class OOMFormatter(ScalarFormatter):
    """
    Formats the axes with a constant number.

    Example:
    ax.yaxis.set_major_formatter(OOMFormatter(4, "%1.1f"))
    """

    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        ScalarFormatter.__init__(self, useOffset=offset, useMathText=mathText)

    def _set_orderOfMagnitude(self, nothing):
        self.orderOfMagnitude = self.oom

    def _set_format(self, vmin, vmax):
        self.format = self.fformat
        if self._useMathText:
            self.format = "$%s$" % matplotlib.ticker._mathdefault(self.format)


def empty_imshow(img_ax):
    """
    Draws an empty canvas on a given image ax.
    """
    empty_arr = np.ndarray(shape=(200, 200))
    img_ax.tick_params(
        axis="both",
        which="both",
        bottom="off",
        top="off",
        labelbottom="off",
        right="off",
        left="off",
        labelleft="off",
    )
    img_ax.set_facecolor(gvars.color_hud_black)
    img_ax.imshow(empty_arr, alpha=0)
    img_ax.fill_between(
        [0, 200],
        [200, 200],
        color=None,
        hatch="/",
        edgecolor=gvars.color_hud_white,
        alpha=0,
    )
    img_ax.set_xlim(0, 200)
    img_ax.set_ylim(0, 200)
    return img_ax


def major_formatter_in_label(ax, axis, axis_label, major_formatter):
    """
    Places an exponent formatter (like 10e3) in the axis label.
    """
    if axis == "x":
        axis = ax.xaxis
    if axis == "y":
        axis = ax.yaxis

    axis.set_major_formatter(major_formatter)
    axis.offsetText.set_visible(False)
    exponent = axis.get_offset_text().get_text()

    axis.set_label_text(axis_label + " (" + exponent + ")")


def plot_rois(
    spots,
    img_ax,
    color,
    radius=gvars.roi_draw_radius,
    linewidths=gvars.roi_draw_linewidth,
):
    """
    Draws ROI overlays on a given image ax.
    """
    patches = []

    for spot in spots:
        y, x = spot
        c = plt.Circle((x, y), radius)
        patches.append(c)
    img_ax.add_collection(
        PatchCollection(
            patches, facecolors="None", edgecolors=color, linewidths=linewidths
        )
    )


def plot_roi_coloc(
    spots_coloc,
    img_ax,
    color1,
    color2,
    radius=gvars.roi_draw_radius,
    linewidths=gvars.roi_draw_linewidth,
):
    """
    Draws ROI overlays from colocalized spots on a given image ax.
    """

    patches1 = []
    patches2 = []

    for row in spots_coloc.itertuples():
        n, y1, x1, y2, x2 = row

        c1 = plt.Circle((x1, y1), color=color1, radius=radius)
        c2 = plt.Circle((x2, y2), color=color2, radius=radius)

        patches1.append(c1)
        patches2.append(c2)

        img_ax.annotate(
            n, (x1, y1), color="white", va="center", ha="center", fontsize=8
        )

    for patches, colors in zip((patches1, patches2), (color1, color2)):
        img_ax.add_collection(
            PatchCollection(
                patches,
                facecolors="None",
                edgecolors=colors,
                linewidths=linewidths,
            )
        )


def point_density(xdata, ydata, kernel="gaussian", bandwidth=0.1):
    """
    This function only evaluates density in the exact points. See contour_2d function for continuous KDE.

    Example
    -------
    c = point_density(xdata = xdata, ydata = ydata, kernel = "linear", bandwidth = 0.1)
    plt.scatter(xdata, ydata, c = c, cmap = "magma")
    """

    if kernel == "epa":
        kernel = "epanechnikov"

    if bandwidth == "auto":
        bandwidth = estimate_bw(n=len(xdata) + len(ydata), d=2, factor=0.25)

    positions = np.vstack([xdata.ravel(), ydata.ravel()])
    kernel_sk = sklearn.neighbors.KernelDensity(
        kernel=kernel, bandwidth=bandwidth
    ).fit(list(zip(*positions)))
    z = np.exp(kernel_sk.score_samples(list(zip(*positions))))
    return z


def plot_shaded_category(y, ax, alpha, colors=None):
    """
    Plots a color for every class segment in a timeseries

    Parameters
    ----------
    y:
        One-hot coded or categorical labels
    ax:
        Ax for plotting
    alpha:
        Alpha of shading
    colors:
        Colors to cycle through
    """
    if colors is None:
        colors = ("darkgrey", "red", "green", "orange", "royalblue", "purple")

    if len(y.shape) != 1:
        y_ = y.argmax(axis=1)  # one-hot to single
    else:
        y_ = y

    if len(colors) < len(set(y_)):
        raise ValueError("Must have at least a color for each class")

    adjs, lns = lib.math.count_adjacent_values(y_)
    position = range(len(y_))
    for idx, ln in zip(adjs, lns):
        label = y_[idx]
        ax.axvspan(
            xmin=position[idx],
            xmax=position[idx] + ln,
            alpha=alpha,
            facecolor=colors[label],
        )


def plot_predictions(yi_pred, ax):
    """
    Plots Keras predictions as probabilities with shaded argmax overlays
    """
    clrs = (
        "darkgrey",
        "red",
        "green",
        "orange",
        "royalblue",
        "mediumvioletred",
    )
    p, confidence = lib.math.seq_probabilities(yi_pred, skip_threshold=0.5)
    plot_shaded_category(y=yi_pred, ax=ax, colors=clrs, alpha=0.3)

    patches = []
    for i in range(yi_pred.shape[-1]):
        patch = Patch(
            color=clrs[i],
            label="{:.0f} %".format(p[i] * 100) if p[i] != 0 else None,
        )
        patches.append(patch)

        # plot_trace_and_preds predicted probabilities
        ax.plot(yi_pred[:, i], color=clrs[i])
        ax.annotate(
            s="confidence: {:.1f} %".format(p[[2, 3]].sum() * 100),
            xy=(0, 1),
            xytext=(8, -8),
            va="top",
            xycoords="axes fraction",
            textcoords="offset points",
            bbox=dict(
                boxstyle="round",
                alpha=0.25,
                facecolor="white",
                edgecolor="lightgrey",
            ),
        )
    ax.legend(handles=patches, loc="upper right", ncol=3)
    ax.set_ylabel("$p_i$")
