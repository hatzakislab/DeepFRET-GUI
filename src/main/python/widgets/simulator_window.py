# coding=utf-8
import os
from functools import partial

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import lib.math
import lib.utils
import lib.plotting
from global_variables import GlobalVariables as gvars
from lib.container import TraceContainer
from ui._SimulatorWindow import Ui_SimulatorWindow
from widgets.misc import ExportDialog, ProgressBar
from widgets.base_window import BaseWindow


class SimulatorWindow(BaseWindow):
    """
    smFRET trace simulator window
    """

    def __init__(self):
        super().__init__()
        # self.data.examples = {}
        self.df = pd.DataFrame()
        self.ui = Ui_SimulatorWindow()
        self.ui.setupUi(self)

        self.setupFigureCanvas(ax_type="dynamic", use_layoutbox=True)
        self.connectUi()

    def connectUi(self):
        """
        Connects interface to function. Most of it is done automatically, based
        on ui element names
        """
        # Checkboxes
        for element in dir(self.ui):
            if element.startswith("checkBox"):
                [
                    getattr(self.ui, element).clicked.connect(f)
                    for f in (self.refreshUi, self.refreshPlot)
                ]

        # Input fields
        for element in dir(self.ui):
            if element.startswith("input"):
                if element.startswith("inputFretStateMeans"):
                    continue
                else:
                    getattr(self.ui, element).textChanged.connect(
                        self.refreshPlot
                    )

        # Functions will be carried out in the given order when export button
        # is pushed
        for f in (
            self.valuesFromGUI,
            partial(self.generateTraces, False),
            self.exportTracesToAscii,
        ):
            self.ui.pushButtonExport.clicked.connect(f)

        # Other inputs not handled above
        self.ui.inputFretStateMeans.returnPressed.connect(self.refreshPlot)
        self.ui.examplesComboBox.currentTextChanged.connect(self.refreshPlot)

    def exportTracesToAscii(self, checked_only=False):
        """
        Exports all traces as ASCII files to selected directory.
        Maintains compatibility with iSMS and older pySMS scripts.
        """
        traces = (
            self.data.simulated_traces
        )  # TODO : maybe generated traces should be stored at .traces?

        if checked_only:
            selected = [trace for trace in traces.values() if trace.is_checked]
        else:
            selected = [trace for trace in traces.values()]

        directory = self.getConfig(gvars.key_lastOpenedDir)
        if not os.path.exists(directory):
            directory = ""

        diag = ExportDialog(init_dir=directory, accept_label="Export")

        if diag.exec():
            path = diag.selectedFiles()[0]
            self.processEvents()

            for trace in selected:
                trace.export_trace_to_txt(dir_to_join=path)

    def savePlot(self):
        """
        Saves plot for simulated traces
        """
        self.setSavefigrcParams()
        self.canvas.defaultImageName = "Simulated Traces"
        self.canvas.toolbar.save_figure()
        self.refreshPlot()

    def refreshUi(self):
        """Refreshes UI to e.g. disable some input boxes"""
        for inputBox, checkBox in (
            (self.ui.inputDonorMeanLifetime, self.ui.checkBoxDlifetime),
            (self.ui.inputAcceptorMeanLifetime, self.ui.checkBoxALifetime),
            (
                self.ui.inputTransitionProbabilityHi,
                self.ui.checkBoxTransitionProbability,
            ),
            (self.ui.inputFretStateMeans, self.ui.checkBoxRandomState),
            (self.ui.inputNoiseHi, self.ui.checkBoxNoise),
            (self.ui.inputMismatchHi, self.ui.checkBoxMismatch),
            (self.ui.inputScalerHi, self.ui.checkBoxScaler),
            (self.ui.inputBleedthroughHi, self.ui.checkBoxBleedthrough),
        ):
            inputBox.setDisabled(checkBox.isChecked())

        self.ui.inputMaxRandomStates.setEnabled(
            self.ui.checkBoxRandomState.isChecked()
        )

    def valuesFromGUI(self):
        """
        Fetch values from GUI
        """
        # Number of traces to export
        # Number of examples
        self.n_examples = (
            int(self.ui.examplesComboBox.currentText().split("x")[0]) ** 2
        )

        self.n_traces = int(self.ui.inputNumberOfTraces.value())

        # Trace length
        self.trace_len = int(self.ui.inputTraceLength.value())

        # Scramble probability
        self.scramble_prob = float(self.ui.inputScrambleProbability.value())

        # Scramble decouple probability
        self.scramble_decouple_prob = float(
            self.ui.inputScrambleDecoupleProbability.value()
        )

        # Aggregation probability
        self.aggregate_prob = float(self.ui.inputAggregateProbability.value())

        # Max aggregate size
        self.max_aggregate_size = int(self.ui.inputMaxAggregateSize.value())

        # FRET state means
        if self.ui.checkBoxRandomState.isChecked():
            self.fret_means = "random"
        else:
            if hasattr(self, "fret_means"):
                old_fret_means = self.fret_means
            else:
                old_fret_means = float(self.ui.inputFretStateMeans.text())

            new_fret_means = sorted(
                lib.utils.numstring_to_ls(self.ui.inputFretStateMeans.text())
            )
            if not new_fret_means:
                self.fret_means = 0

            if new_fret_means != old_fret_means:
                self.fret_means = new_fret_means

        # Max number of random states
        self.max_random_states = int(self.ui.inputMaxRandomStates.value())

        # Minimum FRET state difference
        self.min_fret_diff = float(self.ui.inputMinFretStateDiff.value())

        # Donor mean lifetime
        if self.ui.checkBoxDlifetime.isChecked():
            self.donor_lifetime = None
        else:
            self.donor_lifetime = int(self.ui.inputDonorMeanLifetime.value())

        # Acceptor mean lifetime
        if self.ui.checkBoxALifetime.isChecked():
            self.acceptor_lifetime = None
        else:
            self.acceptor_lifetime = int(
                self.ui.inputAcceptorMeanLifetime.value()
            )

        # Blinking probability
        self.blinking_prob = float(self.ui.inputBlinkingProbability.value())

        # Fall-off probability
        self.fall_off_prob = float(self.ui.inputFalloffProbability.value())

        # Fall-off lifetime
        self.fall_off_lifetime = int(self.ui.inputFalloffMeanLifetime.value())

        # Transition Probability
        if self.ui.checkBoxTransitionProbability.isChecked():
            self.transition_prob = float(
                self.ui.inputTransitionProbabilityLo.value()
            )
        else:
            self.transition_prob = (
                float(self.ui.inputTransitionProbabilityLo.value()),
                float(self.ui.inputTransitionProbabilityHi.value()),
            )

        # Noise
        if self.ui.checkBoxNoise.isChecked():
            self.noise = float(self.ui.inputNoiseLo.value())
        else:
            self.noise = (
                float(self.ui.inputNoiseLo.value()),
                float(self.ui.inputNoiseHi.value()),
            )

        # Acceptor-only mismatch
        if self.ui.checkBoxMismatch.isChecked():
            self.aa_mismatch = float(self.ui.inputMismatchLo.value())
        else:
            self.aa_mismatch = (
                float(self.ui.inputMismatchLo.value()),
                float(self.ui.inputMismatchHi.value()),
            )

        # Donor Bleedthrough
        if self.ui.checkBoxBleedthrough.isChecked():
            self.bleed_through = float(self.ui.inputBleedthroughLo.value())
        else:
            self.bleed_through = (
                float(self.ui.inputBleedthroughLo.value()),
                float(self.ui.inputBleedthroughHi.value()),
            )

        # Scaler
        if self.ui.checkBoxScaler.isChecked():
            self.scaling_factor = float(self.ui.inputScalerLo.value())
        else:
            self.scaling_factor = (
                float(self.ui.inputScalerLo.value()),
                float(self.ui.inputScalerHi.value()),
            )

    def getTrace(self, idx) -> TraceContainer:
        """Returns a trace, given assigned index from df"""
        return self.data.simulated_traces[idx]

    def generateTraces(self, examples: bool):
        """Generate traces to show in the GUI (examples) or for export"""
        n_traces = self.n_examples if examples else self.n_traces

        if n_traces > 50:
            # every number of traces gets 20 updates in total
            # but closes if less
            update_every_nth = n_traces // 20
            progressbar = ProgressBar(
                parent=self, loop_len=n_traces / update_every_nth
            )
        else:
            update_every_nth = None
            progressbar = None

        if examples:
            self.data.example_traces.clear()
        else:
            self.data.simulated_traces.clear()

        df = lib.math.generate_traces(
            n_traces=n_traces,
            aa_mismatch=self.aa_mismatch,
            state_means=self.fret_means,
            min_state_diff=self.min_fret_diff,
            random_k_states_max=self.max_random_states,
            max_aggregate_size=self.max_aggregate_size,
            aggregation_prob=self.aggregate_prob,
            scramble_prob=self.scramble_prob,
            scramble_decouple_prob=self.scramble_decouple_prob,
            trace_length=self.trace_len,
            trans_prob=self.transition_prob,
            blink_prob=self.blinking_prob,
            bleed_through=self.bleed_through,
            noise=self.noise,
            D_lifetime=self.donor_lifetime,
            A_lifetime=self.acceptor_lifetime,
            au_scaling_factor=self.scaling_factor,
            falloff_prob=self.fall_off_prob,
            falloff_lifetime=self.fall_off_lifetime,
            discard_unbleached=False,
            progressbar_callback=progressbar,
            callback_every=update_every_nth,
            return_matrix=False,
            reduce_memory=False,
            run_headless_parallel=False,
            merge_state_labels=True,
        )

        df.index = np.arange(0, len(df), 1) // int(self.trace_len)
        for n, (idx, trace_df) in enumerate(df.groupby(df.index)):
            self.data.simulated_traces[idx] = TraceContainer(
                filename="trace_{}.txt".format(idx),
                loaded_from_ascii=False,
                n=idx,
            )

            self.getTrace(idx).set_from_df(df=trace_df)

        if progressbar is not None:
            progressbar.close()

    def refreshPlot(self):
        """Refreshes preview plots"""
        self.valuesFromGUI()

        try:
            for ax in self.canvas.axes:
                self.canvas.fig.delaxes(ax)
        except KeyError:
            pass

        # generate at least enough traces to show required number of examples
        if self.n_traces < self.n_examples:
            self.n_traces = self.n_examples

        self.generateTraces(examples=True)

        n_subplots = self.n_examples
        nrows = int(self.n_examples ** (1 / 2))
        ncols = nrows
        outer_grid = matplotlib.gridspec.GridSpec(
            nrows, ncols, wspace=0.1, hspace=0.1
        )  # 2x2 grid

        self.canvas.axes = []

        for i in range(n_subplots):
            trace = self.getTrace(i)
            inner_subplot = matplotlib.gridspec.GridSpecFromSubplotSpec(
                nrows=5,
                ncols=1,
                subplot_spec=outer_grid[i],
                wspace=0,
                hspace=0,
                height_ratios=[3, 3, 3, 3, 1],
            )
            axes = [
                plt.Subplot(self.canvas.fig, inner_subplot[n]) for n in range(5)
            ]
            self.canvas.axes.extend(axes)

            ax_g_r, ax_red, ax_frt, ax_sto, ax_lbl = axes
            bleach = trace.first_bleach
            tmax = len(trace.grn.int)
            fret_states = np.unique(trace.fret_true)
            fret_states = fret_states[fret_states != -1]

            ax_g_r.plot(trace.grn.int, color="seagreen")
            ax_g_r.plot(trace.acc.int, color="salmon")
            ax_red.plot(trace.red.int, color="red")
            ax_frt.plot(trace.fret, color="orange")
            ax_frt.plot(
                trace.fret_true, color="black", ls="-", alpha=0.3
            )  # TODO: fix in init

            for state in fret_states:
                ax_frt.plot([0, bleach], [state, state], color="red", alpha=0.2)

            ax_sto.plot(trace.stoi, color="purple")

            lib.plotting.plot_simulation_category(
                y=trace.y_class,  # TODO: fix in init
                ax=ax_lbl,
                alpha=0.4,
                fontsize=max(10, 36 // self.n_examples),
                model_has_states=False,
            )

            for ax in ax_frt, ax_sto:
                ax.set_ylim(-0.15, 1.15)

            for ax, s in zip((ax_g_r, ax_red), (trace.grn.int, trace.red.int)):
                ax.set_ylim(s.max() * -0.15)
                ax.plot([0] * len(s), color="black", ls="--", alpha=0.5)

            for ax in axes:
                for spine in ax.spines.values():
                    spine.set_edgecolor("darkgrey")

                if bleach is not None:
                    ax.axvspan(bleach, tmax, color="black", alpha=0.1)

                ax.set_xticks(())
                ax.set_yticks(())
                ax.set_xlim(0, tmax)
                self.canvas.fig.add_subplot(ax)

        self.canvas.draw()
