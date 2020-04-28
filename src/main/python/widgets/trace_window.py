# coding=utf-8
import os
import warnings
from functools import partial

import numpy as np
import pandas as pd
from PyQt5.QtCore import QModelIndex, Qt, pyqtSlot
from PyQt5.QtGui import QStandardItem
from PyQt5.QtWidgets import QFileDialog

import lib.math
import lib.plotting
from global_variables import GlobalVariables as gvars
from lib.container import TraceContainer
from ui._MenuBar import Ui_MenuBar
from ui._TraceWindow import Ui_TraceWindow
from widgets.misc import ProgressBar
from widgets.base_window import BaseWindow
from widgets.histogram_window import HistogramWindow
from widgets.transition_density_window import TransitionDensityWindow


class TraceWindow(BaseWindow):
    """
    Window for displaying currently obtained traces.
    """

    def __init__(self):
        super().__init__()

        # Initialize UI states
        self.currName = None
        self.currRow = None
        self.currDir = None
        self.currChk = None
        self.loadCount = 0

        # Initialize HMM type
        self.hmmModel = None

        # Initialize interface
        # Instantiate but do not show
        self.ui = Ui_TraceWindow()
        self.ui.setupUi(self)
        self.setupListView(use_layoutbox=False)
        self.setupFigureCanvas(
            ax_type="trace", use_layoutbox=False,
        )
        self.setupPlot()
        self.setupSplitter(layout=self.ui.layoutBox)

        # Canvas event handler
        self.cid = self.canvas.mpl_connect(
            "button_press_event", self.selectCorrectionFactorRange
        )

    @pyqtSlot(QModelIndex)
    def onChecked(self, index):
        """
        Refreshes other windows when a trace is checked.
        """
        histogram_window = self.windows[
            gvars.HistogramWindow
        ]  # type: HistogramWindow

        transition_density_window = self.windows[
            gvars.TransitionDensityWindow
        ]  # type: TransitionDensityWindow

        item = self.listModel.itemFromIndex(index)  # type: QStandardItem

        if self.currName is not None:
            self.currentTrace().is_checked = (
                True if item.checkState() == Qt.Checked else False
            )

        if item.checkState() in (Qt.Checked, Qt.Unchecked):
            histogram_window.getHistogramData()
            histogram_window.gauss_params = None
            if histogram_window.isVisible():
                histogram_window.refreshPlot()

            histogram_window.setPooledLifetimes()
            transition_density_window.setClusteredTransitions()

            if transition_density_window.isVisible():
                transition_density_window.refreshPlot()

    def enablePerWindow(self):
        """
        Disables specific commands that should be unavailable for certain
        window types. Export commands should be accessible from all windows
        (no implicit behavior).
        """
        self.ui: Ui_MenuBar

        menulist = (
            self.ui.actionRemove_File,
            self.ui.actionRemove_All_Files,
            self.ui.actionClear_Traces,
            self.ui.actionAdvanced_Sort,
            self.ui.actionSort_by_Ascending,
            self.ui.actionSort_by_Green_Bleach,
            self.ui.actionSort_by_Red_Bleach,
            self.ui.actionSort_by_Equal_Stoichiometry,
            self.ui.actionUncheck_All,
            self.ui.actionCheck_All,
            self.ui.actionGet_alphaFactor,
            self.ui.actionGet_deltaFactor,
            self.ui.actionColor_Red,
            self.ui.actionColor_Yellow,
            self.ui.actionColor_Green,
            self.ui.actionClear_Color,
            self.ui.actionClear_All_Colors,
            self.ui.actionCorrectionFactorsWindow,
            self.ui.actionClear_Correction_Factors,
            self.ui.actionSelect_Bleach_Red_Channel,
            self.ui.actionSelect_Bleach_Green_Channel,
            self.ui.actionFit_Hmm_Selected,
            self.ui.actionPredict_Selected_Traces,
            self.ui.actionPredict_All_traces,
            self.ui.actionClear_All_Predictions,
            self.ui.actionClear_and_Rerun,
            self.ui.actionCheck_All_Traces,
            self.ui.actionUncheck_All_Traces,
        )
        for menu in menulist:
            menu.setEnabled(True)

    def clearTraceAndRerun(self):
        self.windows[gvars.VideoWindow].clearTraceAndRerun(self)

    def returnContainerInstance(self):
        """Returns appropriate data container for implemented windows"""
        return self.data.traces

    def showCorrectionFactorInspector(self):
        """
        Opens the inspector (modal) window to set correction factors
        """
        if self.isVisible():
            self.inspectors[gvars.CorrectionFactorInspector].show()

    def openFile(self, *args):
        """
        Loads ASCII files directly into the TraceWindow.
        """
        directory = self.getLastOpenedDir()

        filenames, selectedFilter = QFileDialog.getOpenFileNames(
            self,
            caption="Open File",
            filter="Trace files (*.txt *.dat)",
            directory=directory,
        )

        if len(filenames) != 0:
            select_top_row = True if len(self.data.traces) == 0 else False
            update_every_n = int(10) if len(filenames) > 10 else int(2)
            progressbar = ProgressBar(
                loop_len=len(filenames) / update_every_n, parent=self
            )
            rows = []
            for n, full_filename in enumerate(filenames):
                if progressbar.wasCanceled():
                    break

                self.currDir = os.path.dirname(full_filename)
                try:
                    newTrace = TraceContainer(
                        filename=full_filename, loaded_from_ascii=True
                    )

                except AttributeError:  # if a non-trace file was selected
                    warnings.warn(
                        f"This file could not be read: \n{full_filename}",
                        UserWarning,
                    )
                    continue
                if (n % update_every_n) == 0:
                    progressbar.increment()

                # If file wasn't loaded properly, skip
                if newTrace.load_successful is False:
                    continue
                # Don't load duplicates
                if newTrace.name in self.data.traces.keys():
                    continue

                self.data.traces[newTrace.name] = newTrace
                item = QStandardItem(newTrace.name)
                rows.append(item)

            # Don't touch the GUI until loading is done
            [self.listModel.appendRow(row) for row in rows]
            [
                self.listModel.item(i).setCheckable(True)
                for i in range(self.listModel.rowCount())
            ]
            self.listView.repaint()
            progressbar.close()

            if select_top_row:
                self.selectListViewTopRow()
            self.getCurrentListObject()
            self.setConfig(gvars.key_lastOpenedDir, self.currDir)

    def selectCorrectionFactorRange(self, event):
        """
        Manually click to select correction factor range.
        """
        if self.currName is not None:
            for n, ax in enumerate(self.canvas.axes):
                if ax == event.inaxes and n in [1, 2]:
                    # Register xdata
                    clickedx = int(event.xdata)

                    self.currentTrace().xdata.append(clickedx)
                    # Third click resets the (min, max) range
                    if len(self.currentTrace().xdata) == 3:
                        self.currentTrace().xdata = []

                    self.refreshPlot()

    def fitCheckedTracesHiddenMarkovModel(self):
        """
        Fits all selected traces with a Hidden Markov Model (HMM)
        """
        self.processEvents()
        traces = [
            trace for trace in self.data.traces.values() if trace.is_checked
        ]
        if not traces:
            warnings.warn("No traces were selected!", UserWarning)
        print(
            "Fitting HMM with {} setting".format(
                self.getConfig(gvars.key_hmmMode)
            )
        )
        lDD, lDA, lAA, lE, llengths = [], [], [], [], []
        for trace in traces:
            _, I_DD, I_DA, I_AA = lib.math.correct_DA(trace.get_intensities())
            lDD.append(I_DD[: trace.first_bleach])
            lDA.append(I_DA[: trace.first_bleach])
            lAA.append(I_AA[: trace.first_bleach])
            lE.append(trace.fret[: trace.first_bleach])
            llengths.append(len(I_DD[: trace.first_bleach]))

        E = np.array(lE)

        if self.getConfig(gvars.key_hmmMode) == "DA":
            X = []
            for ti in range(len(lDD)):
                _x = np.column_stack((lDD[ti], lDA[ti], lAA[ti], lE[ti]))
                X.append(_x)

            if lib.math.contains_nan([np.sum(aa) for aa in X[:][2]]):
                X = [np.concatenate((_x[:, :2], _x[:, 3:]), axis=1) for _x in X]

            X = np.array(X)
        else:
            X = E.copy()

        E_flat = np.concatenate(E)

        best_mixture_model, params = lib.math.fit_gaussian_mixture(
            E_flat,
            min_n_components=1,
            max_n_components=6,
            strict_bic=self.getConfig(gvars.key_hmmBICStrictness),
            verbose=True,
        )
        n_components = best_mixture_model.n_components
        self.hmmModel = lib.math.get_hmm_model(X, n_components=n_components)

        log_transmat = self.hmmModel.dense_transition_matrix()
        n_states = (
            self.hmmModel.node_count() - 2
        )  # minus virtual start and end state
        transmat = log_transmat[:n_states, :n_states]

        state_dict = {}
        for i, state in enumerate(self.hmmModel.states):
            try:
                if self.getConfig(gvars.key_hmmMode) == "DA":
                    state_dict[
                        f"{state.name}".replace("s", "")
                    ] = state.distribution.parameters[0][-1].parameters
                else:
                    state_dict[
                        f"{state.name}".replace("s", "")
                    ] = state.distribution.parameters
            except AttributeError:
                continue
        means = np.array([v[0] for v in state_dict.values()])
        sigs = np.array([v[1] for v in state_dict.values()])

        print("Transition matrix:\n", np.round(transmat, 2))
        print("State means:\n", means)
        print("State sigmas:\n", sigs)

        for ti, trace in enumerate(traces):
            _X = X[ti]
            tf = pd.DataFrame()
            tf["e_obs"] = trace.fret[: trace.first_bleach]
            tf["state"] = np.array(self.hmmModel.predict(_X)).astype(int)
            tf["e_pred_global"] = (
                tf["state"]
                .astype(str)
                .replace(
                    {
                        k: v[0]
                        for (k, v) in zip(
                            state_dict.keys(), state_dict.values()
                        )
                    },
                    inplace=False,
                )
            )
            tf["e_pred_local"] = tf.groupby(["state"], as_index=False)[
                "e_obs"
            ].transform("mean")

            tf["time"] = tf["e_pred_local"].index + 1

            trace.hmm_state = tf["state"].values
            trace.hmm_local_fret = tf["e_pred_local"].values
            trace.hmm_global_fret = tf["e_pred_global"].values
            trace.hmm_idx = tf["time"].values

            trace.calculate_transitions()

        if self.windows["TransitionDensityWindow"].isVisible():
            self.windows["TransitionDensityWindow"].refreshPlot()

    @staticmethod
    def setClassifications(trace, yi_pred):
        """Assign predicted trace classifications to trace"""
        trace.y_pred = yi_pred
        trace.y_class, trace.confidence = lib.math.seq_probabilities(
            trace.y_pred
        )
        trace.first_bleach = lib.math.find_bleach(
            p_bleach=trace.y_pred[:, 0], threshold=0.5, window=7
        )
        for c in trace.channels:
            c.bleach = trace.first_bleach

    def classifyTraces(self, single=False, checked_only=False):
        """
        Classifies checked traces with deep learning model.
        """
        self.processEvents()

        alpha = self.getConfig(gvars.key_alphaFactor)
        delta = self.getConfig(gvars.key_deltaFactor)

        if single:
            traces = [self.currentTrace()]
            if traces == [None]:
                traces.clear()

        elif checked_only:
            traces = [
                trace for trace in self.data.traces.values() if trace.is_checked
            ]

        else:
            traces = [trace for trace in self.data.traces.values()]

        if len(traces) > 0:
            batch_size = 256
            batches = (len(traces) // batch_size) + 1
            progressbar = ProgressBar(loop_len=batches, parent=self)

            if not single:
                all_lengths_eq = lib.math.all_equal(
                    [trace.frames_max for trace in traces]
                )
                all_features_eq = lib.math.all_equal(
                    [lib.math.contains_nan(trace.red.int) for trace in traces]
                )
            else:
                all_lengths_eq = False
                all_features_eq = None

            if all((all_lengths_eq, all_features_eq)):
                # shape is (n_traces) if traces have uneven length
                X = np.array(
                    [
                        lib.math.correct_DA(
                            trace.get_intensities(), alpha=alpha, delta=delta
                        )
                        for trace in traces
                    ]
                )

                # Swap from (samples, features, time) to
                # (samples, time, features)
                X = np.swapaxes(X, 1, 2)

                X = (
                    X[..., [1, 2]]
                    if lib.math.contains_nan(X[..., -1])
                    else X[..., [1, 2, 3]]
                )
                # Normalize tensor
                X = lib.math.sample_max_normalize_3d(X)

                # Fix single sample dimension
                if len(X.shape) == 2:
                    X = X[np.newaxis, :, :]

                model = (
                    self.keras_two_channel_model
                    if X.shape[-1] == 2
                    else self.keras_three_channel_model
                )

                Y = lib.math.predict_batch(
                    X=X,
                    model=model,
                    progressbar=progressbar,
                    batch_size=batch_size,
                )
            else:
                Y = []
                for n, trace in enumerate(traces):
                    xi = np.column_stack(
                        lib.math.correct_DA(
                            trace.get_intensities(), alpha=alpha, delta=delta
                        )
                    )
                    if lib.math.contains_nan(xi[..., -1]):
                        model = self.keras_two_channel_model
                        xi = xi[..., [1, 2]]
                    else:
                        model = self.keras_three_channel_model
                        xi = xi[..., [1, 2, 3]]

                    xi = lib.math.sample_max_normalize_3d(X=xi)

                    yi = lib.math.predict_single(xi=xi, model=model)
                    Y.append(yi)
                    if n % batch_size == 0:
                        progressbar.increment()

            for n, trace in enumerate(traces):
                self.setClassifications(trace=trace, yi_pred=Y[n])

            self.resetCurrentName()

    def clearAllClassifications(self):
        """
        Clears all classifications for a trace. This will also clear
        predicted bleaching.
        """
        traces = [trace for trace in self.data.traces.values()]
        for trace in traces:  # type: TraceContainer
            trace.y_pred = None
            trace.y_class = None
            trace.first_bleach = None
            for c in trace.channels:
                c.bleach = None

    def triggerBleach(self, color):
        """
        Sets the appropriate matplotlib event handler for bleaching.
        """
        # Disconnect general handler for correction factors
        self.cid = self.canvas.mpl_disconnect(self.cid)

        # Connect for bleach
        self.cid = self.canvas.mpl_connect(
            "button_press_event", partial(self.selectBleach, color)
        )
        self.setCursor(Qt.CrossCursor)

    def selectBleach(self, color, event):
        """
        Manually click to select bleaching event
        """
        if self.currName is not None:
            for n, ax in enumerate(self.canvas.axes):
                if ax == event.inaxes and n in [1, 2]:
                    # Register xdata
                    clickedx = int(event.xdata)

                    if color == "green":
                        self.currentTrace().grn.bleach = clickedx
                    elif color == "red":
                        self.currentTrace().red.bleach = clickedx
                        self.currentTrace().acc.bleach = clickedx
                    self.currentTrace().first_bleach = lib.math.min_real(
                        self.currentTrace().get_bleaches()
                    )
                    self.currentTrace().hmm = None
                    self.refreshPlot()

        # Disconnect bleach event handler
        self.canvas.mpl_disconnect(self.cid)

        # Reset to original event handler
        self.cid = self.canvas.mpl_connect(
            "button_press_event", self.selectCorrectionFactorRange
        )
        self.setCursor(Qt.ArrowCursor)

    def clearListModel(self):
        """
        Clears the internal data of the listModel. Call this before total
        refresh of interface.listView.
        """
        self.listModel.removeRows(0, self.listModel.rowCount())
        self.data.traces = []

    def returnTracenamesAllVideos(self, names_only=True):
        """
        Obtains all available tracenames from videos and returns them as a list.
        """
        if names_only:
            tracenames = [trace for trace in self.data.traces.keys()]
        else:
            tracenames = [trace for trace in self.data.traces.values()]

        return tracenames

    def showAdvancedSortInspector(self):
        """
        Show advanced sort inspector
        """
        self.inspectors[gvars.AdvancedSortInspector].show()

    def sortListByCondition(self, setup):
        """
        Checks all traces where the red fluorophore (acceptor or direct
        excitation) channel is bleached.
        """
        # Only used for inspector currently
        params = self.inspectors[
            gvars.AdvancedSortInspector
        ].returnInspectorValues()

        (
            S_med_lo,
            S_med_hi,
            E_med_lo,
            E_med_hi,
            min_n_frames,
            confidence,
            dynamics,
            bleached_only,
        ) = params

        for index in range(self.listModel.rowCount()):
            item = self.listModel.item(index)
            trace = self.getTrace(item)

            if setup in ("red", "green"):
                channel = trace.grn if setup == "green" else trace.red
                pass_all = (
                    channel.bleach is not None
                    and channel.bleach >= min_n_frames
                )

            elif setup == "S":
                S = trace.stoi[:10]
                cond1 = 0.4 < np.median(S) < 0.6
                cond2 = all((S > 0.3) & (S < 0.7))
                pass_all = all((cond1, cond2))

            elif setup == "advanced":
                conditions = []

                S = trace.stoi[: trace.first_bleach]
                E = trace.fret[: trace.first_bleach]

                # TODO: write a warning that stoichiometry will be ignored?
                if lib.math.contains_nan(trace.red.int):
                    cond1 = True
                else:
                    cond1 = S_med_lo < np.median(S) < S_med_hi
                cond2 = E_med_lo < np.median(E) < E_med_hi
                cond3 = (
                    True
                    if trace.first_bleach is None
                    else trace.first_bleach >= min_n_frames
                )

                conditions += [cond1, cond2, cond3]

                if trace.y_pred is not None and confidence > 0:
                    cond4 = trace.confidence > confidence
                    conditions += [cond4]
                    if dynamics > 0:
                        cond5 = trace.y_class >= dynamics
                        conditions += [cond5]

                cond6 = (
                    False
                    if bleached_only and trace.first_bleach is None
                    else True
                )

                conditions += [cond6]

                pass_all = all(conditions)
            else:
                raise ValueError

            if pass_all:
                item.setCheckState(Qt.Checked)
                trace.is_checked = True
            else:
                item.setCheckState(Qt.Unchecked)
                trace.is_checked = False

        self.sortListByChecked()
        self.selectListViewTopRow()

        self.inspectors[gvars.AdvancedSortInspector].setInspectorConfigs(params)

    def currentTrace(self) -> TraceContainer:
        """
        Get metadata for currently selected trace.
        """
        if self.currName is not None:
            try:
                return self.data.traces[self.currName]
            except KeyError:
                self.currName = None

    def getTrace(self, name) -> TraceContainer:
        """Gets metadata for specified trace."""
        if type(name) == QStandardItem:
            name = name.text()
        return self.data.traces.get(name)

    def savePlot(self):
        """
        Saves plot with colors suitable for export (e.g. white background).
        """
        self.setSavefigrcParams()

        # Change colors of output to look better for export
        if self.currName is not None:
            self.canvas.defaultImageName = self.currentTrace().get_tracename()

        else:
            self.canvas.defaultImageName = "Blank"

        for ax in self.canvas.axes:
            ax.tick_params(axis="both", colors=gvars.color_hud_black)
            ax.yaxis.label.set_color(gvars.color_hud_black)
            for spine in ax.spines.values():
                spine.set_edgecolor(gvars.color_hud_black)

        self.canvas.toolbar.save_figure()

        # Afterwards, reset figure colors to GUI after saving
        for ax in self.canvas.axes:
            ax.tick_params(axis="both", colors=gvars.color_hud_white)
            for spine in ax.spines.values():
                spine.set_edgecolor(gvars.color_gui_text)

        for ax, label in self.canvas.axes_c:
            ax.yaxis.label.set_color(gvars.color_gui_text)

        self.refreshPlot()

    def setupPlot(self):
        """
        Plot setup for TraceWindow.
        """
        self.canvas.fig.set_facecolor(gvars.color_gui_bg)
        for ax in self.canvas.axes:
            ax.yaxis.set_label_coords(-0.05, 0.5)

            for spine in ax.spines.values():
                spine.set_edgecolor(gvars.color_gui_text)
                spine.set_linewidth(0.5)

        self.canvas.ax_acc.yaxis.set_label_coords(1.05, 0.5)

    def getCorrectionFactors(self, factor):
        """
        The correction factor for leakage (alpha) is determined by Equation 5
        using the FRET efficiency of the D-only population
        """
        trace = self.currentTrace()

        if self.currName is not None and len(trace.xdata) == 2:
            xmin, xmax = sorted(trace.xdata)

            I_DD = trace.grn.int[xmin:xmax]
            I_DA = trace.acc.int[xmin:xmax]
            I_AA = trace.red.int[xmin:xmax]

            if factor == "alpha":
                trace.a_factor = lib.math.alpha_factor(I_DD, I_DA)
            elif factor == "delta":
                trace.d_factor = lib.math.delta_factor(I_DD, I_DA, I_AA)
            self.refreshPlot()

    def clearCorrectionFactors(self):
        """Zeros alpha and delta correction factors"""
        if self.currName is not None:
            self.currentTrace().a_factor = np.nan
            self.currentTrace().d_factor = np.nan

        self.clearMarkerLines()
        self.refreshPlot()

    def getCurrentLines(self, ax):
        """Get labels of currently plotted lines"""
        if self.currName is not None:
            return [l.get_label() for l in ax.lines]

    def clearMarkerLines(self):
        """Clears the lines drawn to decide xmin/xmax. This also clears the
        property"""
        if self.currName is not None:
            self.currentTrace().xdata = []

    def refreshPlot(self):
        """
        Refreshes plot for TraceWindow.
        """
        self.canvas.fig.legends = []
        trace = self.currentTrace()

        if trace is not None and len(self.data.traces) > 0:
            if self.getConfig(gvars.key_hmmLocal):
                trace.hmm_idealized_config = "local"
            else:
                trace.hmm_idealized_config = "global"

            alpha = self.getConfig(gvars.key_alphaFactor)
            delta = self.getConfig(gvars.key_deltaFactor)
            factors = alpha, delta
            F_DA, I_DD, I_DA, I_AA = lib.math.correct_DA(
                trace.get_intensities(), *factors
            )
            zeros = np.zeros(len(F_DA))

            for ax in self.canvas.axes:
                ax.clear()
                ax.tick_params(
                    axis="both", colors=gvars.color_gui_text, width=0.5
                )
                if len(trace.xdata) == 2:
                    xmin, xmax = sorted(trace.xdata)
                    ax.set_xlim(xmin, xmax)
                else:
                    ax.set_xlim(1, trace.frames_max)

            channels = [trace.grn, trace.acc, trace.red]
            colors = [gvars.color_green, gvars.color_red, gvars.color_red]

            for (ax, label), channel, color in zip(
                self.canvas.axes_c, channels, colors
            ):
                if label == "A":
                    signal = F_DA
                    ax = self.canvas.ax_acc
                elif label == "A-direct":
                    signal = I_AA
                    ax = self.canvas.ax_red
                elif label == "D":
                    signal = I_DD
                    ax = self.canvas.ax_grn

                    af = (
                        r"$\alpha$ = {:.2f}".format(trace.a_factor)
                        if not np.isnan(trace.a_factor)
                        else ""
                    )
                    df = (
                        r"$\delta$ = {:.2f}".format(trace.d_factor)
                        if not np.isnan(trace.d_factor)
                        else ""
                    )
                    ax.annotate(
                        s=af + "\n" + df,
                        xy=(0.885, 0.90),
                        xycoords="figure fraction",
                        color=gvars.color_grey,
                    )
                else:
                    signal = channel.int
                if signal is None:
                    continue

                if label != "A":
                    # Don't zero plot background for A, because it's already
                    # plotted from D
                    ax.plot(trace.frames, zeros, color="black", **gvars.bg_p)
                    ax.axvspan(
                        trace.first_bleach,
                        trace.frames_max,
                        color="darkgrey",
                        alpha=0.3,
                        zorder=0,
                    )

                ax.plot(trace.frames, signal, color=color)

                # Try to make both signals land on top of each other, and zero
                try:
                    ax.set_ylim(0 - signal.max() * 0.1, signal.max() * 1.1)
                except ValueError:
                    ax.set_ylim(0, 1.1)
                ax.yaxis.label.set_color(gvars.color_gui_text)

                if not lib.math.contains_nan(signal):
                    lib.plotting.set_axis_exp_ylabel(
                        ax=ax, label=label, values=signal
                    )
                else:
                    ax.set_ylabel("")
                    ax.set_yticks(())

            # Continue drawing FRET specifics
            fret = lib.math.calc_E(trace.get_intensities(), *factors)
            stoi = lib.math.calc_S(trace.get_intensities(), *factors)

            for signal, ax, color, label in zip(
                (fret, stoi),
                (self.canvas.ax_fret, self.canvas.ax_stoi),
                (gvars.color_orange, gvars.color_purple),
                ("E", "S"),
            ):
                ax.plot(trace.frames, signal, color=color)
                ax.axvspan(
                    trace.first_bleach,
                    trace.frames_max,
                    color="darkgrey",
                    alpha=0.4,
                    zorder=0,
                )

                ax.set_ylim(-0.1, 1.1)
                ax.set_ylabel(label)
                ax.set_yticks([0.5])
                ax.axhline(
                    0.5, color="black", alpha=0.3, lw=0.5, ls="--", zorder=2,
                )

            if trace.hmm is not None:
                self.canvas.ax_fret.plot(
                    trace.hmm_idx, trace.hmm, color=gvars.color_blue, zorder=3,
                )

            # If clicking on the trace
            if len(trace.xdata) == 1:
                self.canvas.ax_grn.axvline(
                    trace.xdata[0],
                    ls="-",
                    alpha=0.2,
                    zorder=10,
                    color=gvars.color_red,
                )
                self.canvas.ax_red.axvline(
                    trace.xdata[0],
                    ls="-",
                    alpha=0.2,
                    zorder=10,
                    color=gvars.color_red,
                )
            elif len(trace.xdata) == 2:
                xmin, xmax = sorted(trace.xdata)

                for ax in self.canvas.axes:
                    if ax != self.canvas.ax_acc:
                        ax.axvline(
                            xmin,
                            color=gvars.color_red,
                            zorder=10,
                            lw=30,
                            alpha=0.2,
                        )
                        ax.axvline(
                            xmax,
                            color=gvars.color_red,
                            zorder=10,
                            lw=30,
                            alpha=0.2,
                        )

            if hasattr(self.canvas, "ax_ml") and trace.y_pred is not None:
                lib.plotting.plot_predictions(
                    yi_pred=trace.y_pred,
                    fig=self.canvas.fig,
                    ax=self.canvas.ax_ml,
                )
        else:
            for ax in self.canvas.axes:
                ax.clear()
                ax.tick_params(axis="both", colors=gvars.color_gui_bg)

        self.canvas.draw()

    def _debug(self):
        pass
