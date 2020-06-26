# coding=utf-8
import matplotlib
import numpy as np
import pandas as pd
import scipy.optimize
import sklearn.cluster
from matplotlib import pyplot as plt

import lib.math
import lib.plotting
from global_variables import GlobalVariables as gvars
from ui._MenuBar import Ui_MenuBar
from ui._TransitionDensityWindow import Ui_TransitionDensityWindow
from widgets.base_window import BaseWindow


class TransitionDensityWindow(BaseWindow):
    """
    Window for displaying all transitions available as fitted by the the
    Hidden Markov Model.
    """

    def __init__(self):
        super().__init__()
        self.currDir = None
        self.selected_data = None
        self.colors = None

        # dynamically created once plot is refreshed
        self.tdp_ax = None
        self.hist_axes = None

        self.ui = Ui_TransitionDensityWindow()
        self.ui.setupUi(self)
        self.ui.nClustersSpinBox.valueChanged.connect(self.refreshPlot)

        self.setupFigureCanvas(ax_type="plot", width=2, height=2)
        self.setupPlot()

    def enablePerWindow(self):
        """
        Disables specific commands that should be unavailable for certain
        window types. Export commands should be accessible from all windows
        (no implicit behavior).
        """
        self.ui: Ui_MenuBar

        menulist = (self.ui.actionFormat_Plot,)
        for menu in menulist:
            menu.setEnabled(True)

    def showDensityWindowInspector(self):
        """
        Opens the inspector (modal) window to format the current plot
        """
        if self.isActiveWindow():
            self.inspectors[gvars.DensityWindowInspector].show()

    def unCheckAll(self):
        """
        Unchecks all list elements. Attribute check because they override
        "select", which is normally reserved for text fields
        """
        super().unCheckAll()
        if self.isVisible():
            self.refreshPlot()

    def checkAll(self):
        """
        Checks all list elements.
        """
        super().checkAll()
        if self.isVisible():
            self.refreshPlot()

    def savePlot(self):
        """
        Saves plot with colors suitable for export (e.g. white background)
        for TransitionWindow.
        """
        self.setSavefigrcParams()
        self.canvas.defaultImageName = "TDP"
        self.canvas.toolbar.save_figure()
        self.refreshPlot()

    def setupPlot(self):
        """
        Set up plot for HistogramWindow.
        """
        self.canvas.fig.set_facecolor(gvars.color_gui_bg)

        for ax in self.canvas.axes:
            ax.tick_params(
                colors=gvars.color_gui_text,
                width=0.5,
                axis="both",
                which="both",
                bottom=False,
                top=False,
                left=False,
                right=False,
            )

            for spine in ax.spines.values():
                spine.set_edgecolor(gvars.color_gui_text)
                spine.set_linewidth(0.5)

    def plotDefaultElementsLeft(self):
        """
        Re-plot non-persistent plot settings for left (TDP)
        """
        try:
            smax = max(self.data.tdpData.state_before)
        except TypeError:
            smax = 1.0
        self.tdp_ax.set_xlim(-0.15, smax + 0.15)
        self.tdp_ax.set_ylim(-0.15, smax + 0.15)
        self.tdp_ax.set_xlabel(xlabel="Before", color=gvars.color_gui_text)
        self.tdp_ax.set_ylabel(ylabel="After", color=gvars.color_gui_text)
        self.tdp_ax.tick_params(axis="both", colors=gvars.color_gui_text)

    def plotDefaultElementsRight(self):
        """
        Re-plot non-persistent plot settings for right (histograms)
        """
        for ax in self.hist_axes:
            ax.set_xticks(())
            ax.set_yticks(())

    def plotDefaultElements(self):
        """
        Re-plots non-persistent plot settings for all axes
        """
        for ax in self.canvas.axes:
            for spine in ax.spines.values():
                spine.set_edgecolor(gvars.color_gui_text)
                spine.set_linewidth(0.5)

    def setClusteredTransitions(self):
        if self.data.tdpData.state_before is not None:
            n_clusters = self.ui.nClustersSpinBox.value()

            tdp_df = pd.DataFrame(
                {
                    "e_before": self.data.tdpData.state_before,
                    "e_after": self.data.tdpData.state_after,
                    "lifetime": self.data.tdpData.state_after,
                }
            )

            tdp_df.dropna(inplace=True)

            up_diag = tdp_df[tdp_df["e_before"] < tdp_df["e_after"]]  # 0
            lw_diag = tdp_df[tdp_df["e_before"] > tdp_df["e_after"]]  # 1

            halves = up_diag, lw_diag
            for n, half in enumerate(halves):
                m = sklearn.cluster.KMeans(n_clusters=n_clusters)
                m.fit(half[["e_before", "e_after"]])
                half["label"] = m.labels_ + n_clusters * n

            diags = pd.concat(halves)
            tdp_df["label"] = diags["label"]

            self.data.tdpData.df = tdp_df

    def setupDynamicGridAxes(self):
        """
        Sets up dynamic gridspec for changing number of subplots on the fly.
        Remember to add the axes at the end of plotting
        """

        n_rows = self.ui.nClustersSpinBox.value()

        # Outer grid
        self.gs = matplotlib.gridspec.GridSpec(1, 2, wspace=0.1, hspace=0.1)

        # Left grid (1)
        tdp_subplot = matplotlib.gridspec.GridSpecFromSubplotSpec(
            nrows=1, ncols=1, subplot_spec=self.gs[0], wspace=0, hspace=0
        )
        self.tdp_ax = plt.Subplot(self.canvas.fig, tdp_subplot[0])

        # Right grid (2-column dynamic)
        n_hists = n_rows * 2
        hist_subplots = matplotlib.gridspec.GridSpecFromSubplotSpec(
            nrows=n_rows, ncols=2, subplot_spec=self.gs[1], wspace=0, hspace=0
        )
        self.hist_axes = [
            plt.Subplot(self.canvas.fig, hist_subplots[n])
            for n in range(n_hists)
        ]
        self.canvas.axes = self.hist_axes + [self.tdp_ax]

    def plotTransitionDensity(self, params):
        """
        Plots TDP contents
        """
        bandwidth, resolution, n_colors, overlay_pts, pts_alpha = params

        try:
            smax = max(self.data.tdpData.state_before)
        except TypeError:
            smax = 1.0

        self.tdp_ax.plot(
            [-0.15, smax + 0.15],
            [-0.15, smax + 0.15],
            color="lightgrey",
            ls="--",
        )
        if (
            self.data.tdpData.state_before is not None
            and len(self.data.tdpData.state_before) > 0
        ):
            cont = lib.math.contour_2d(
                xdata=self.data.tdpData.state_before,
                ydata=self.data.tdpData.state_after,
                bandwidth=bandwidth / 200,
                resolution=resolution,
                kernel="linear",
                n_colors=n_colors,
            )
            self.tdp_ax.contourf(*cont, cmap="viridis")
            self.tdp_ax.text(
                x=0,
                y=smax - 0.1,
                s="N = {}\n"
                "{} transitions\n".format(
                    self.data.histData.n_samples,
                    len(self.data.tdpData.state_after),
                ),
                color=gvars.color_gui_text,
            )

            tdp_df_grp = self.data.tdpData.df.groupby("label")
            self.colors = lib.plotting.get_colors(
                "viridis", tdp_df_grp.ngroups * 2
            )

            for (
                i,
                cluster,
            ) in (
                tdp_df_grp
            ):  # TODO: figx this so it doesn't need to call multiple times to data.tdpf_df
                xi = cluster["e_before"]
                yi = cluster["e_after"]

                self.tdp_ax.scatter(
                    x=xi,
                    y=yi,
                    s=20,
                    color=self.colors[i],
                    edgecolor="black",
                    alpha=pts_alpha / 20 if overlay_pts else 0,
                )
        else:
            self.tdp_ax.plot([])
            self.tdp_ax.set_xticks(())
            self.tdp_ax.set_yticks(())

        self.canvas.fig.add_subplot(self.tdp_ax)

    def plotHistograms(self):
        """
        Plots histogram contents
        """
        np.random.seed(1)  # make sure histograms don't change on every re-fit

        if self.data.tdpData.df is not None:
            fret_lifetimes = self.data.tdpData.df["lifetime"]
            max_lifetime = np.max(fret_lifetimes)
            bw = lib.math.estimate_binwidth(fret_lifetimes)
            bins = np.arange(0, max_lifetime, bw)

            for k, cluster in self.data.tdpData.df.groupby("label"):
                try:
                    hx, hy, *_ = lib.math.histpoints_w_err(
                        data=cluster["lifetime"],
                        bins=bins,
                        density=False,
                        least_count=1,
                    )
                    popt, pcov = scipy.optimize.curve_fit(
                        lib.math.exp_function, xdata=hx, ydata=hy
                    )
                    perr = np.sqrt(np.diag(pcov))

                    rate = popt[1]
                    rate_err = perr[1]

                    if rate_err > 3 * rate:
                        rate_err = np.inf

                    self.hist_axes[k].plot(
                        bins,
                        lib.math.exp_function(bins, *popt),
                        "--",
                        color="black",
                        label="label: {}\n"
                        "$\lambda$:  ${:.2f} \pm {:.2f}$\n"
                        "lifetime: ${:.2f}$".format(
                            k, rate, rate_err, 1 / rate
                        ),
                    )
                except RuntimeError:  # drop fit if it doesn't converge
                    pass

                histy, *_ = self.hist_axes[k].hist(
                    cluster["lifetime"],
                    bins=bins,
                    color=self.colors[k],
                    density=False,
                )
                self.hist_axes[k].set_xlim(0, max(bins))
                self.hist_axes[k].set_ylim(0, max(histy) * 1.1)
                self.hist_axes[k].legend(loc="upper right")
        else:
            for ax in self.hist_axes:
                ax.plot([])

        for ax in self.hist_axes:
            self.canvas.fig.add_subplot(ax)

    def refreshPlot(self):
        """
        Refreshes plot for TransitionWindow
        """
        self.canvas.fig.clear()
        self.setupDynamicGridAxes()

        try:
            self.setPooledLifetimes()
            params = self.inspectors[
                gvars.DensityWindowInspector
            ].returnInspectorValues()
            self.inspectors[gvars.DensityWindowInspector].setInspectorConfigs(
                params
            )

            self.setClusteredTransitions()
            self.plotTransitionDensity(params)
            self.plotHistograms()

        except (KeyError, AttributeError):
            for ax in self.canvas.axes:
                ax.plot([])
                self.canvas.fig.add_subplot(ax)

        self.plotDefaultElementsLeft()
        self.plotDefaultElementsRight()
        self.plotDefaultElements()

        self.canvas.draw()

    def _debug(self):
        self.refreshPlot()
