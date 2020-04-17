from functools import partial

import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QFileDialog

import lib.math
import lib.misc
import lib.plotting
from global_variables import GlobalVariables as gvars
from lib.container import HistogramData
from ui._HistogramWindow import Ui_HistogramWindow
from ui._MenuBar import Ui_MenuBar
from widgets.base import BaseWindow


class HistogramWindow(BaseWindow):
    def __init__(self):
        super().__init__()
        # Histogram data
        # TODO: these should all be merged to be part of HistogramData container
        self.E = None
        self.E_un = None
        self.S = None
        self.S_un = None
        self.DD, self.DA, self.corrs, = None, None, None

        self.alpha = None
        self.delta = None
        self.beta = None
        self.gamma = None

        # Plotting parameters
        self.marg_bins = np.arange(-0.3, 1.3, 0.02)
        self.xpts = np.linspace(-0.3, 1.3, 300)

        self.ui = Ui_HistogramWindow()
        self.ui.setupUi(self)

        self.da_label = r"$\mathbf{DA}$"
        self.dd_label = r"$\mathbf{DD}$"

        self.setupFigureCanvas(ax_type="histwin")

        self.setupPlot()
        self.connectUi()

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

    def formatPlotInspector(self):
        """
        Opens the inspector (modal) window to format the current plot
        """
        if self.isActiveWindow():
            self.inspector.show()

    def exportHistogramData(self):
        """
        Exports histogram data for selected traces
        """
        exp_txt, date_txt = self.returnInfoHeader()

        directory = (
            self.getConfig(gvars.key_lastOpenedDir) + "/E_S_Histogram.txt"
        )
        path, _ = QFileDialog.getSaveFileName(
            self, directory=directory
        )  # type: str, str

        self.getHistogramData()

        if path != "":
            if not path.split("/")[-1].endswith(".txt"):
                path += ".txt"

            # Exports all the currently plotted datapoints
            if self.ui.applyCorrectionsCheckBox.isChecked():
                E, S = self.E, self.S
            else:
                E, S = self.E_un, self.S_un

            if E is not None:
                df = pd.DataFrame({"E": E, "S": S}).round(4)
            else:
                df = pd.DataFrame({"E": [], "S": []})

            with open(path, "w") as f:
                f.write(
                    "{0}\n"
                    "{1}\n\n"
                    "{2}".format(
                        exp_txt,
                        date_txt,
                        df.to_csv(index=False, sep="\t", na_rep="NaN"),
                    )
                )

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

    def connectUi(self):
        for f in (
            partial(self.fitGaussians, "auto"),
            partial(self.refreshPlot, True),
        ):
            self.ui.gaussianAutoButton.clicked.connect(f)

        for f in (self.fitGaussians, self.refreshPlot):
            self.ui.gaussianSpinBox.valueChanged.connect(f)

        for f in (self.fitGaussians, self.refreshPlot):
            self.ui.applyCorrectionsCheckBox.clicked.connect(f)

        for f in (self.fitGaussians, self.refreshPlot):
            self.ui.framesSpinBox.valueChanged.connect(f)

    def savePlot(self):
        """
        Saves plot with colors suitable for export (e.g. white background)
        for HistogramWindow.
        """
        self.setSavefigrcParams()
        self.canvas.defaultImageName = "2D histogram"
        self.canvas.toolbar.save_figure()
        self.refreshPlot()

    def setupPlot(self):
        """
        Set up plot for HistogramWindow.
        """
        self.canvas.fig.set_facecolor(gvars.color_gui_bg)

        for ax in self.canvas.axes:
            for spine in ax.spines.values():
                spine.set_edgecolor(gvars.color_gui_text)
                spine.set_linewidth(0.5)
            ax.tick_params(axis="both", colors=gvars.color_gui_text, width=0.5)

            # ax.tick_params(
            #     axis="both",
            #     which="both",
            #     bottom=False,
            #     top=False,
            #     left=False,
            #     right=False,
            #     # labelbottom=False,
            #     # labeltop=False,
            #     # labelleft=False,
            #     # labelright=False,
            # )

    def getHistogramData(self, n_first_frames="spinbox"):
        """
        Returns pooled E and S_app data before bleaching, for each trace.
        The loops take approx. 0.1 ms per trace, and it's too much trouble
        to lower it further.
        Also return DD, DA, and Pearson correlation data.
        """
        if n_first_frames == "all":
            n_first_frames = None
        elif n_first_frames == "spinbox":
            n_first_frames = self.ui.framesSpinBox.value()
        else:
            raise ValueError("n_first_frames must be either 'all' or 'spinbox'")

        # TODO: we might use this inline more often, but keep self.attr = None
        # TODO: in the init, to keep PyCharm inspections
        # for attr in ("E", "S", "DD", "DA", "corrs", "E_un", "S_un"):
        #     setattr(self, attr, None)
        self.E, self.S, self.DD, self.DA, self.corrs, self.E_un, self.S_un = (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

        checkedTraces = [
            trace for trace in self.data.traces.values() if trace.is_checked
        ]

        self.data.histData.n_samples = len(checkedTraces)
        alpha = self.getConfig(gvars.key_alphaFactor)
        delta = self.getConfig(gvars.key_deltaFactor)

        self.data.histData.trace_median_len = int(
            np.median(
                [
                    trace.first_bleach
                    if trace.first_bleach is not None
                    else trace.frames_max
                    for trace in checkedTraces
                ]
            )
        )

        DA, DD, E_app, S_app, lengths, corrs = [], [], [], [], [], []
        for trace in checkedTraces:
            E, S = lib.math.drop_bleached_frames(
                intensities=trace.get_intensities(),
                bleaches=trace.get_bleaches(),
                alpha=alpha,
                delta=delta,
                max_frames=n_first_frames,
            )
            E_app.extend(E)
            S_app.extend(S)
            _, I_DD, I_DA, I_AA = lib.math.correct_DA(trace.get_intensities())
            trace.calculate_stoi()
            DD.append(I_DD[: trace.first_bleach])
            DA.append(I_DA[: trace.first_bleach])
            lengths.append(len(trace.fret[: trace.first_bleach]))
            if self.getConfig(gvars.key_medianPearsonCorr):
                n_lags_pcorr = self.data.histData.trace_median_len
            else:
                n_lags_pcorr = self.getConfig(gvars.key_lagsPearsonCorr)

            n_lags_pcorr = int(n_lags_pcorr)

            c = lib.math.corrcoef_lags(
                I_DD[: trace.first_bleach],
                I_DA[: trace.first_bleach],
                n_lags=n_lags_pcorr,
            )
            corrs.append(c)

        self.DD = np.concatenate(DD).flatten()
        self.DA = np.concatenate(DA).flatten()

        self.data.histData.lengths = np.array(lengths)
        corrs = np.array(corrs)
        if len(corrs.shape) == 1:  # a nested array instead of one big array
            corrs = lib.math.correct_corrs(corrs)
        self.corrs = corrs

        self.E_un, self.S_un = lib.math.trim_ES(E_app, S_app)

        # Skip ensemble correction if stoichiometry is missing
        if not lib.math.contains_nan(self.S_un):
            if len(self.E_un) > 0:
                beta, gamma = lib.math.beta_gamma_factor(
                    E_app=self.E_un, S_app=self.S_un
                )
                E_real, S_real, = [], []
                for trace in checkedTraces:
                    E, S = lib.math.drop_bleached_frames(
                        intensities=trace.get_intensities(),
                        bleaches=trace.get_bleaches(),
                        alpha=alpha,
                        delta=delta,
                        beta=beta,
                        gamma=gamma,
                        max_frames=n_first_frames,
                    )
                    E_real.extend(E)
                    S_real.extend(S)
                self.E, self.S = lib.math.trim_ES(E_real, S_real)
                self.beta = beta
                self.gamma = gamma
        else:
            self.E = self.E_un

        self.alpha = alpha
        self.delta = delta

        self.updatePooledData()

    def updatePooledData(self):
        """
        Sets all variables from getHistogramData to be in a HistogramData container.
        This way we can access the data from other windows as well.
        This is only a temporary solution, over time this should be moved to DataContainer and this method should be removed
        """
        self.data.histData = HistogramData()
        self.data.histData.alpha = self.alpha
        self.data.histData.delta = self.delta
        self.data.histData.beta = self.beta
        self.data.histData.gamma = self.gamma

        (
            self.data.histData.E,
            self.data.histData.S,
            self.data.histData.DD,
            self.data.histData.DA,
            self.data.histData.corrs,
            self.data.histData.E_un,
            self.data.histData.S_un,
        ) = (self.E, self.S, self.DD, self.DA, self.corrs, self.E_un, self.S_un)

    def fitGaussians(self, states):
        """
        Fits multiple gaussians to the E data
        """
        corrected = self.ui.applyCorrectionsCheckBox.isChecked()
        E = self.E if corrected else self.E_un

        if E is not None:
            n_components = (
                (1, 6) if states == "auto" else self.ui.gaussianSpinBox.value()
            )

            best_model, params = lib.math.fit_gaussian_mixture(
                X=E,
                min_n_components=np.min(n_components),
                max_n_components=np.max(n_components),
            )

            self.data.histData.gauss_params = params
            self.data.histData.best_k = best_model.n_components
            self.ui.gaussianSpinBox.setValue(self.data.histData.best_k)
            self.ui.gaussianSpinBox.repaint()

    def plotDefaultElements(self):
        """
        Re-plot non-persistent plot settings (otherwise will be overwritten
        by ax.clear())
        """
        self.canvas.tl_ax_ctr.set_xlim(-0.1, 1.1)
        self.canvas.tl_ax_ctr.set_ylim(-0.1, 1.1)

        for ax in self.canvas.axes_marg:
            for tk in ax.get_xticklabels():
                tk.set_visible(False)
            for tk in ax.get_yticklabels():
                tk.set_visible(False)

        self.canvas.tr_ax_ctr.yaxis.set_major_formatter(
            lib.misc.format_string_to_k
        )
        self.canvas.tr_ax_ctr.xaxis.set_major_formatter(
            lib.misc.format_string_to_k
        )
        self.canvas.br_ax.xaxis.set_major_formatter(lib.misc.format_string_to_k)

        self.canvas.bl_ax_b.axhline(
            0,
            *self.canvas.bl_ax_b.get_xlim(),
            ls="--",
            color="black",
            alpha=0.3,
        )
        self.canvas.tr_ax_top.set_xlabel(self.dd_label)
        self.canvas.tr_ax_top.xaxis.set_label_position("top")

        self.canvas.tr_ax_rgt.set_ylabel(self.da_label)
        self.canvas.tr_ax_rgt.yaxis.set_label_position("right")

        self.canvas.tl_ax_top.set_xlabel(r"$\mathbf{E}_{FRET}$")
        self.canvas.tl_ax_top.xaxis.set_label_position("top")

        self.canvas.tl_ax_rgt.set_ylabel(r"$\mathbf{S}$")
        self.canvas.tl_ax_rgt.yaxis.set_label_position("right")

        self.canvas.bl_ax_b.set_xlabel("Frames")
        self.canvas.bl_ax_b.set_ylabel(r"$\operatorname{{E}}[\rho_{{DD, DA}}]$")

    def plotTopLeft_TopMarginal(self, corrected):
        """
        Plots the top left top marginal histogram (E).
        """
        E = self.E if corrected else self.E_un

        if E is not None:
            self.canvas.tl_ax_top.clear()
            self.canvas.tl_ax_top.hist(
                E,
                bins=self.marg_bins,
                color=gvars.color_orange,
                alpha=0.8,
                density=True,
                histtype="stepfilled",
            )
        if self.data.histData.gauss_params is not None:
            joint_dist = []
            xpts = self.xpts
            for (m, s, w) in self.data.histData.gauss_params:
                _, y = lib.plotting.plot_gaussian(
                    mean=m, sigma=s, weight=w, x=xpts, ax=self.canvas.tl_ax_top
                )
                joint_dist.append(y)

            # Sum of all gaussians (joint distribution)
            joint_dist = np.sum(joint_dist, axis=0)

            self.canvas.tl_ax_top.plot(
                xpts,
                joint_dist,
                color=gvars.color_grey,
                alpha=1,
                zorder=10,
                ls="--",
            )

    def plotTopLeft_RightMarginal(self, corrected):
        """
        Plots the top left right marginal histogram (S).
        """
        S = self.S if corrected else self.S_un
        # self.canvas.tl_ax_rgt : plt.Axes
        self.canvas.tl_ax_rgt.set_ylabel("Stoichiometry")
        self.canvas.tl_ax_rgt.yaxis.set_label_position("right")
        if S is not None:
            self.canvas.tl_ax_rgt.clear()
            self.canvas.tl_ax_rgt.hist(
                S,
                bins=self.marg_bins,
                color=gvars.color_purple,
                alpha=0.8,
                density=True,
                histtype="stepfilled",
                orientation="horizontal",
            )

    def plotTopLeft_CenterContour(self, corrected):
        """
        Plots the top left center E+S contour plot.
        """
        S = self.S if corrected else self.S_un
        E = self.E if corrected else self.E_un

        params = self.inspector.returnInspectorValues()
        bandwidth, resolution, n_colors, overlay_pts, pts_alpha = params
        self.inspector.setInspectorConfigs(params)

        self.canvas.tl_ax_ctr.clear()

        n_equals_txt = "N = {}\n".format(self.data.histData.n_samples)
        if self.data.histData.trace_median_len is not None:
            n_equals_txt += "(median length {:.0f})".format(
                self.data.histData.trace_median_len
            )

        self.canvas.tl_ax_ctr.text(
            x=0, y=0.9, s=n_equals_txt, color=gvars.color_gui_text
        )

        if self.data.histData.gauss_params is not None:
            for n, gauss_params in enumerate(self.data.histData.gauss_params):
                m, s, w = gauss_params
                self.canvas.tl_ax_ctr.text(
                    x=0.6,
                    y=0.15 - 0.05 * n,
                    s=r"$\mu_{}$ = {:.2f} $\pm$ {:.2f} ({:.2f})".format(
                        n + 1, m, s, w
                    ),
                    color=gvars.color_gui_text,
                    zorder=10,
                )

        for n, (factor, name) in enumerate(
            zip(
                (self.alpha, self.delta, self.beta, self.gamma),
                ("alpha", "delta", "beta", "gamma"),
            )
        ):
            if factor is not None:
                self.canvas.tl_ax_ctr.text(
                    x=0.0,
                    y=0.15 - 0.05 * n,
                    s=r"$\{}$ = {:.2f}".format(name, factor),
                    color=gvars.color_gui_text,
                    zorder=10,
                )

        if S is not None:
            c = lib.math.contour_2d(
                xdata=E,
                ydata=S,
                bandwidth=bandwidth / 200,
                resolution=resolution,
                kernel="linear",
                n_colors=n_colors,
            )
            self.canvas.tl_ax_ctr.contourf(*c, cmap="plasma")

            if overlay_pts:
                self.canvas.tl_ax_ctr.scatter(
                    E, S, s=20, color="black", zorder=1, alpha=pts_alpha / 20
                )  # Conversion factor, because sliders can't do [0,1]
            self.canvas.tl_ax_ctr.axhline(
                0.5, color="black", alpha=0.3, lw=0.5, ls="--", zorder=2
            )

    def plotTopRight_RightMarginal(self):
        """
        Plots the top right right marginal histogram (DA).
        """
        self.canvas.tr_ax_rgt.set_ylabel(self.da_label)
        self.canvas.tr_ax_rgt.yaxis.set_label_position("right")
        if self.DA is not None:
            self.canvas.tr_ax_rgt.clear()
            self.canvas.tr_ax_rgt.hist(
                self.DA,
                bins=100,
                density=True,
                alpha=0.5,
                label=self.da_label,
                color=gvars.color_red,
                orientation="horizontal",
            )

    def plotTopRight_TopMarginal(self):
        """
        Plots the top right top marginal histogram (DD).
        """
        if self.DD is not None:
            self.canvas.tr_ax_top.clear()
            self.canvas.tr_ax_top.hist(
                self.DD,
                bins=100,
                density=True,
                alpha=0.5,
                label=self.dd_label,
                color=gvars.color_green,
            )

    def plotTopRight_CenterContour(self):
        da_contour_x = self.DD
        if da_contour_x is not None:
            da_contour_y = self.DA
            cont = lib.math.contour_2d(
                xdata=da_contour_x,
                ydata=da_contour_y,
                # bandwidth='auto',
                extend_grid=0,
                bandwidth=150,
                resolution=50,
                kernel="linear",
                n_colors=20,
                diagonal=True,
            )
            self.canvas.tr_ax_ctr.contourf(
                *cont, cmap="magma"
            )  # cmap="viridis")

    def plotBottomLeft_Duration(self):
        """
        Plots the bottom left top half Histogram, as well as an exponential fit of lifetimes.
        """
        lengths = self.data.histData.lengths
        if lengths is not None:
            self.canvas.bl_ax_t.clear()
            self.canvas.bl_ax_t.hist(
                lengths, histtype="step", density=True, bins=20,
            )
            lifetime_dict = lib.math.fit_and_compare_exp_funcs(lengths, x0=None)
            xlim = self.canvas.bl_ax_t.get_xlim()
            xarr = np.linspace(0.1, xlim[1], 1000)

            single_param = lifetime_dict["SINGLE_PARAM"]
            yarr_1 = lib.math.func_exp(xarr, single_param)
            self.canvas.bl_ax_t.plot(
                xarr,
                yarr_1,
                c="r",
                label=f"Lifetimes \n"
                + lib.misc.nice_string_output(
                    [r"$\tau$"], [f"{1. / single_param[0]:.2f}",],
                ),
                alpha=0.5,
            )
            self.canvas.bl_ax_t.legend()

        for tk in self.canvas.bl_ax_t.get_yticklabels():
            tk.set_visible(False)

    def plotBottomLeft_Pearson(self, plot_errors=False):
        """
        Plots Mean Pearson Correlation Coefficients in bottom left bottom as an errorbar plot.
        """

        if self.corrs is None:
            return
        self.canvas.bl_ax_b.clear()
        maxlen = len(self.corrs[0])
        pr_xs = np.arange(maxlen // 2 + 1)
        pr_ys = np.zeros(len(pr_xs))
        pr_er = np.zeros_like(pr_ys)
        for i in range(maxlen):
            idx = i - maxlen // 2
            if idx < 0:
                continue
            else:
                _corr = self.corrs[:, i]
                pr_ys[idx] = np.nanmean(_corr)
                pr_er[idx] = np.nanstd(_corr)

        self.canvas.bl_ax_b.errorbar(
            x=pr_xs,
            y=pr_ys,
            yerr=pr_er if plot_errors else 0,
            label=r"$\operatorname{{E}}[\rho_{{DD, DA}}]$",
        )
        self.canvas.bl_ax_b.legend()

    def plotBottomRight(self):
        """
        Plots 1d histogram of DA and DD for comparison purposes
        """
        if self.DA is not None:
            self.canvas.br_ax.clear()

            self.canvas.br_ax.hist(
                self.DD,
                bins=100,
                density=True,
                alpha=0.5,
                label=self.dd_label,
                color=gvars.color_green,
            )
            self.canvas.br_ax.hist(
                self.DA,
                bins=100,
                density=True,
                alpha=0.5,
                label=self.da_label,
                color=gvars.color_red,
            )
            self.canvas.br_ax.legend()

        for tk in self.canvas.br_ax.get_yticklabels():
            tk.set_visible(False)

    def plotBottomLeft(self):
        self.plotBottomLeft_Duration()
        self.plotBottomLeft_Pearson()

    def plotTopLeft(self, corrected):
        self.plotTopLeft_TopMarginal(corrected)
        if self.S is not None:
            self.plotTopLeft_CenterContour(corrected)
            self.plotTopLeft_RightMarginal(corrected)

    def plotTopRight(self):
        self.plotTopRight_CenterContour()
        self.plotTopRight_TopMarginal()
        self.plotTopRight_RightMarginal()

    def plotAll(self, corrected):
        self.plotTopLeft(corrected)
        self.plotTopRight()
        self.plotBottomLeft()
        self.plotBottomRight()
        self.canvas.draw()

    def refreshPlot(self, autofit=False):
        """
        Refreshes plot with currently selected traces. Plot to refresh can be
        top, right (histograms) or center (scatterplot), or all.
        """
        corrected = self.ui.applyCorrectionsCheckBox.isChecked()
        try:
            self.getHistogramData()
            for ax in self.canvas.axes:
                ax.clear()
            if self.E is not None:
                # Force unchecked
                if self.S is None:
                    self.ui.applyCorrectionsCheckBox.setChecked(False)
                    corrected = False
                self.plotDefaultElements()
                self.plotAll(corrected)
            else:
                self.plotDefaultElements()
        except (AttributeError, ValueError):
            pass

        self.plotDefaultElements()
        self.canvas.draw()

    def _debug(self):
        pass
