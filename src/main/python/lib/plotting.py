import multiprocessing

multiprocessing.freeze_support()

from global_variables import GlobalVariables as gvars
import matplotlib

from lib.math import estimate_bw

matplotlib.use("qt5agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from matplotlib.collections import PatchCollection
from matplotlib.patches import Patch
import sklearn.neighbors
import lib.misc
from matplotlib.colors import Normalize
import lib.math
import scipy.stats


def empty_imshow(img_ax):
    """
    Draws an empty canvas on a given image ax.
    """
    empty_arr = np.ndarray(shape=(200, 200))
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
    img_ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        labelbottom=False,
        right=False,
        left=False,
        labelleft=False,
    )
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


def set_axis_exp_ylabel(ax, label, values):
    """
    Formats axis y-label to hold the exponent for space saving
    """
    m = np.max(values)
    e = np.floor(np.log10(np.abs(m)))

    ax.ticklabel_format(style="sci", scilimits=(0, 0), axis="both")
    ax.yaxis.get_offset_text().set_visible(False)
    ax.yaxis.major.formatter._useMathText = True
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=False))
    ax.set_ylabel("{} ($10^{:.0f}$)".format(label, e))


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
    This function only evaluates density in the exact points. See contour_2d
    function for continuous KDE.

    Example
    -------
    c = point_density(xdata = xdata, ydata = ydata, kernel = "linear",
    bandwidth = 0.1)
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
    return np.exp(kernel_sk.score_samples(list(zip(*positions))))


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

    y_ = y.argmax(axis=1) if len(y.shape) != 1 else y
    if len(colors) < len(set(y_)):
        raise ValueError("Must have at least a color for each class")

    adjs, lns = lib.misc.count_adjacent_values(y_)
    position = range(len(y_))
    for idx, ln in zip(adjs, lns):
        label = y_[idx]
        ax.axvspan(
            xmin=position[idx],
            xmax=position[idx] + ln,
            alpha=alpha,
            facecolor=colors[label],
        )


def plot_simulation_category(y, ax, alpha=0.2, fontsize=6):
    """
    Plots a color for every class segment in a timeseries

    Parameters
    ----------
    y_:
        One-hot coded or categorical labels
    ax:
        Ax for plotting
    colors:
        Colors to cycle through
    """
    cls = {
        0: "bleached",
        1: "aggregate",
        2: "noisy",
        3: "scramble",
        4: "1-state",
        5: "2-state",
        6: "3-state",
        7: "4-state",
        8: "5-state",
    }

    colors = {
        0: "darkgrey",
        1: "red",
        2: "blue",
        3: "purple",
        4: "orange",
        5: "lightgreen",
        6: "green",
        7: "mediumseagreen",
        8: "darkolivegreen",
    }

    y_ = y.argmax(axis=1) if len(y.shape) != 1 else y
    y_ = y_.astype(int)  # type conversion to avoid float type labels
    if len(colors) < len(set(y_)):
        raise ValueError("Must have at least a color for each class")

    adjs, lns = lib.misc.count_adjacent_values(y_)
    position = range(len(y_))
    for idx, ln in zip(adjs, lns):
        label = y_[idx]
        ax.axvspan(
            xmin=position[idx],
            xmax=position[idx] + ln,
            alpha=alpha,
            facecolor=colors[label],
        )
    ax.plot([], label=cls[y_[0]], color=colors[y_[0]])
    ax.legend(loc="lower right", prop={"size": fontsize})


def plot_predictions(yi_pred, fig, ax):
    """
    Plots Keras predictions as probabilities with shaded argmax overlays
    """
    names = (
        "bleached",
        "aggregated",
        "noisy",
        "scrambled",
        "1-state",
        "2-state",
        "3-state",
        "4-state",
        "5-state",
    )

    clrs = (
        "darkgrey",
        "red",
        "royalblue",
        "mediumvioletred",
        "orange",
        "lightgreen",
        "springgreen",
        "limegreen",
        "green",
    )
    probability, confidence = lib.math.seq_probabilities(
        yi_pred, skip_threshold=0.5
    )
    plot_shaded_category(y=yi_pred, ax=ax, colors=clrs, alpha=0.1)

    # Upper right individual %
    patches = []
    for i in range(yi_pred.shape[-1]):
        p = probability[i] * 100
        label = "{:.0f}% {}".format(p, names[i])
        # Align with monospace if single digit prob
        if p < 10:
            label = " " + label
        ax.plot(yi_pred[:, i], color=clrs[i])
        patch = Patch(color=clrs[i], label=label)
        patches.append(patch)

    fig.legend(
        handles=patches, loc="center right", prop={"family": "monospace"}
    )

    # Upper left confidence %
    ax.annotate(
        s="confidence: {:.1f} %".format(confidence * 100),
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
    ax.set_ylabel("$p_i$")


def get_colors(cmap, n_colors):
    """Extracts n colors from a colormap"""
    norm = Normalize(vmin=0, vmax=n_colors)
    return [plt.get_cmap(cmap)(norm(i)) for i in range(n_colors)]


def plot_gaussian(mean, sigma, ax, x, weight=1, color=None):
    """
    Plots a single gaussian and returns the provided ax, along with computed
    y values
    """
    y = weight * scipy.stats.norm.pdf(x, mean, sigma)
    ax.plot(x, y, color=color)
    return ax, y


def plot_gaussian_mixture_to_ax(
    mixture_params,
    ax: plt.Axes,
    xpts=None,
    color_means=None,
    color_joint=None,
    plot_sum=True,
    plot_means=True,
):
    sum_ = []
    if xpts is None:
        xpts = np.linspace(0, 1, 2000)
    for i, gauss_params in enumerate(mixture_params):
        m, s, w = gauss_params
        ax.plot(xpts, w * scipy.stats.norm.pdf(xpts, m, s))
        sum_.append(np.array(w * scipy.stats.norm.pdf(xpts, m, s)))
        if plot_means:
            ax.axvline(
                m,
                ls="--",
                lw=0.5,
                c="xkcd:dark pink" if color_means is None else color_means,
                # label=rf'{m:.2f}',
                # label=rf'$\mu_{i} = {m:.2f}$',
            )
    if plot_sum:
        joint = np.sum(sum_, axis=0)
        ax.plot(
            xpts,
            joint,
            color="grey" if color_joint is None else color_joint,
            alpha=1,
            zorder=10,
            ls="--",
        )

    pass
