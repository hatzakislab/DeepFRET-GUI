# coding=utf-8
import multiprocessing
import warnings

import scipy.optimize
import sklearn.cluster

from PyQt5.QtCore import Qt, qWarning
from PyQt5.QtGui import QStandardItem
from PyQt5.QtWidgets import QFileDialog
from matplotlib import pyplot as plt

from ui._MenuBar import Ui_MenuBar
from ui._TransitionDensityWindow import Ui_TransitionDensityWindow
from widgets.base import BaseWindow, AboutWindow, PreferencesWindow
from widgets.histogram import HistogramWindow
from widgets.simulator import SimulatorWindow

multiprocessing.freeze_support()

import os
import sys
from functools import partial
from typing import Union
from lib.misc import timeit
import matplotlib
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from configobj import ConfigObj

import lib.imgdata
import lib.math
import lib.misc
import lib.plotting
from global_variables import GlobalVariables as gvars

matplotlib.use("qt5agg")
import pandas as pd
import numpy as np
from tensorflow_core.python.keras.models import load_model, Model

from ui._MainWindow import Ui_MainWindow
from ui._TraceWindow import Ui_TraceWindow
from ui._CorrectionFactorInspector import Ui_CorrectionFactorInspector
from ui._DensityWindowInspector import Ui_DensityWindowInspector
from ui._TraceWindowInspector import Ui_TraceWindowInspector

from ui.misc import ProgressBar, SheetInspector
from lib.container import (
    VideoContainer,
    ImageChannel,
    TraceChannel,
    TraceContainer,
)

from fbs_runtime.application_context.PyQt5 import ApplicationContext

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 10000)


class MainWindow(BaseWindow):
    """
    Main UI for the application.
    """

    def __init__(self):
        super().__init__()

        # Initialize UI states
        self.currName = None
        self.currRow = None
        self.currRoiSize = gvars.roi_draw_radius
        self.donor_first = self.getConfig(gvars.key_firstFrameIsDonor)
        self.donor_is_left = self.getConfig(gvars.key_donorLeft)
        self.bg_correction = self.getConfig(gvars.key_illuCorrect)

        self.batchLoaded = (
            False  # If videos have been batchloaded, disable some controls
        )

        # Initialize interface
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Spot counter labels
        self.labels = (
            self.ui.labelColocGreenRedSpots,
            self.ui.labelGreenSpots,
            self.ui.labelRedSpots,
        )

        self.setupListView(use_layoutbox=False)
        self.setupFigureCanvas(ax_type="img", use_layoutbox=False)
        self.setupPlot()
        self.setupSplitter(layout=self.ui.LayoutBox)

        # Spinbox triggers
        self.spinBoxes = (self.ui.spotsGrnSpinBox, self.ui.spotsRedSpinBox)

        self.ui.spotsGrnSpinBox.valueChanged.connect(
            partial(self.displaySpotsSingle, "green")
        )
        self.ui.spotsRedSpinBox.valueChanged.connect(
            partial(self.displaySpotsSingle, "red")
        )

        # Contrast boxes
        self.contrastBoxesLo = (
            self.ui.contrastBoxLoGreen,
            self.ui.contrastBoxLoRed,
        )
        self.contrastBoxesHi = (
            self.ui.contrastBoxHiGreen,
            self.ui.contrastBoxHiRed,
        )
        for contrastBox in self.contrastBoxesLo + self.contrastBoxesHi:
            contrastBox.valueChanged.connect(self.refreshPlot)
        self.ui.contrastBoxHiGreen.setValue(
            self.getConfig(gvars.key_contrastBoxHiGrnVal)
        )
        self.ui.contrastBoxHiRed.setValue(
            self.getConfig(gvars.key_contrastBoxHiRedVal)
        )

        self.show()

    def returnContainerInstance(self):
        """Returns appropriate data container for implemented windows"""
        return self.data.videos

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
            self.ui.actionColocalize_All,
            self.ui.actionFind_Show_Traces,
            self.ui.actionClear_Traces,
            self.ui.actionClear_and_Rerun,
        )
        for menu in menulist:
            menu.setEnabled(True)

        self.ui.actionClose.setEnabled(False)

    def currentVideo(self) -> VideoContainer:
        """
        Quick interface to obtain file data (or names will be extremely long).
        Add type hinting in metadata to improve autocomplete.
        """
        return self.data.get(self.currName)

    def newTraceFromVideo(self, n) -> TraceContainer:
        """
        Shortcut to obtain trace, and create a new one if it doesn't exist.
        """
        tracename = self.currName + "_" + str(n)
        if tracename not in self.data.traces:
            self.data.traces[tracename] = TraceContainer(
                filename=tracename, video=self.currName, n=n
            )
        return self.data.traces[tracename]

    def trace_c(self, n, channel) -> TraceChannel:
        """
        Shortcut to obtain trace channel. See newTraceFromVideo().
        """
        if channel == "green":
            return self.newTraceFromVideo(n).grn
        elif channel == "red":
            return self.newTraceFromVideo(n).red
        elif channel == "acc":
            return self.newTraceFromVideo(n).acc
        else:
            raise ValueError("Invalid channel")

    def openFile(self):
        """
        Open file to load in.
        """
        if not self.getConfig(gvars.key_batchLoadingMode):
            directory = self.getLastOpenedDir()

            filenames, selectedFilter = QFileDialog.getOpenFileNames(
                self,
                caption="Open File",
                filter="Video files (*.tif *fits)",
                directory=directory,
            )

            if len(filenames) > 0:
                progressbar = ProgressBar(loop_len=len(filenames), parent=self)
                for i, full_filename in enumerate(filenames):
                    if progressbar.wasCanceled():
                        break

                    # Make sure name is unique
                    uniqueName = lib.misc.generate_unique_name(
                        full_filename=full_filename,
                        array=self.data.videos.keys(),
                    )

                    self.data.load_video_data(
                        path=full_filename,
                        name=uniqueName,
                        donor_is_first=self.donor_first,
                        donor_is_left=self.donor_is_left,
                        bg_correction=self.bg_correction,
                    )

                    item = QStandardItem(uniqueName)
                    item.setCheckable(False)

                    self.currName = uniqueName
                    self.currDir = os.path.dirname(full_filename)
                    self.listModel.appendRow(item)
                    # refresh listView
                    self.listView.repaint()
                    progressbar.increment()

                # Select first video if loading into an empty listView
                if self.currRow is None:
                    self.currRow = 0
                    self.selectListViewTopRow()
                    index = self.listModel.index(self.currRow, 0)
                    name = self.listModel.data(index)
                    self.currName = name

                # Write most recently opened directory to getConfig file,
                # but only if a file was selected
                if self.currDir is not None:
                    self.setConfig(gvars.key_lastOpenedDir, self.currDir)
        else:
            self.batchOpen()

    def batchOpen(self):
        """Loads one video at a time and extracts traces, then clears the
        video from memory afterwards"""
        directory = self.getLastOpenedDir()

        filenames, selectedFilter = QFileDialog.getOpenFileNames(
            self,
            caption="Open File",
            filter="Video files (*.tif *.fits)",
            directory=directory,
        )

        if len(filenames) > 0:
            self.processEvents()
            progressbar = ProgressBar(loop_len=len(filenames), parent=self)
            for i, full_filename in enumerate(filenames):
                if progressbar.wasCanceled():
                    break
                else:
                    progressbar.increment()

                self.currName = os.path.basename(full_filename)
                if self.currName in self.data.videos:
                    continue

                self.data.load_video_data(
                    path=full_filename,
                    name=os.path.basename(full_filename),
                    donor_is_first=self.donor_first,
                    donor_is_left=self.donor_is_left,
                    bg_correction=self.bg_correction,
                )

                channels = ("green", "red")
                for c in channels:
                    if self.currentVideo().acc.exists:
                        self.colocalizeSpotsSingleVideo(
                            channel=c, find_npairs="auto"
                        )
                    else:
                        self.colocalizeSpotsSingleVideo(
                            channel=c, find_npairs="spinbox"
                        )
                self.getTracesSingleVideo()
                self.currentVideo().vid = None
                for c in self.currentVideo().channels + (
                    self.currentVideo().acc,
                ):
                    c.raw = None

                self.listModel.appendRow(QStandardItem(self.currName))
                self.listView.repaint()

                self.currDir = os.path.dirname(full_filename)
                if self.currDir is not None:
                    self.setConfig(gvars.key_lastOpenedDir, self.currDir)

        if len(self.data.traces) > 0:
            currently_loaded = TraceWindow_.returnCurrentListViewNames()
            # Iterate over all filenames and add to list
            for name in self.data.traces.keys():
                # If name is already in list, skip it
                if name in currently_loaded:
                    continue
                item = QStandardItem(name)
                item.setCheckable(True)
                TraceWindow_.listModel.appendRow(item)

            TraceWindow_.selectListViewTopRow()
            TraceWindow_.getCurrentListObject()
            TraceWindow_.show()

            self.batchLoaded = True
            self.refreshInterface()
            self.selectListViewTopRow()
            self.refreshPlot()

    def colocalizeSpotsSingleVideo(self, channel, find_npairs="spinbox"):
        """
        Find and colocalize spots for a single currentVideo (not displayed).
        """
        vid = self.currentVideo()
        tolerance_type = self.getConfig(gvars.key_colocTolerance)
        tolerance_value = gvars.roi_coloc_tolerances[tolerance_type]

        if channel == "green":
            channel = vid.grn
            spinBox = self.ui.spotsGrnSpinBox
            pairs = ((vid.grn, vid.red),)
            colocs = (vid.coloc_grn_red,)

        elif channel == "red":
            channel = vid.red
            spinBox = self.ui.spotsRedSpinBox
            pairs = ((vid.grn, vid.red),)
            colocs = (vid.coloc_grn_red,)
        else:
            raise ValueError("Invalid color")

        if find_npairs == "spinbox":
            find_npairs = spinBox.value()
        elif find_npairs == "auto":
            find_npairs = self.getConfig(gvars.key_autoDetectPairs)
        else:
            raise ValueError("Select from 'spinbox' or 'auto'")

        if self.getConfig(gvars.key_fitSpots):
            if find_npairs > 0:
                # hardcoded value threshold until I come up with something
                # better
                spots = lib.imgdata.find_spots(
                    channel.mean_nobg, value=20, method="laplacian_of_gaussian"
                )

                # Sort spots based on intensity
                real_spots = []
                for spot in spots:
                    masks = lib.imgdata.circle_mask(
                        yx=spot, indices=vid.indices, **gvars.cmask_p
                    )
                    intensity, bg = lib.imgdata.tiff_stack_intensity(
                        channel.mean_nobg, *masks, raw=True
                    )
                    if intensity > bg * 1.05:
                        real_spots.append(spot)

                channel.n_spots = len(real_spots)
                channel.spots = real_spots

        else:
            if find_npairs > 0:
                channel.spots = lib.imgdata.find_spots(
                    channel.mean_nobg,
                    value=find_npairs,
                    method="peak_local_max",
                )
                channel.n_spots = len(channel.spots)

        if channel == "red" and self.getConfig(gvars.key_unColocRed):
            vid.grn.spots = vid.red.spots
            vid.acc.spots = vid.red.spots
            vid.grn.n_spots = vid.red.n_spots
            vid.acc.n_spots = vid.red.n_spots

        for (c1, c2), coloc in zip(pairs, colocs):
            if all((c1.n_spots, c2.n_spots)) > 0:
                coloc.spots = lib.imgdata.colocalize_rois(
                    c1.spots,
                    c2.spots,
                    color1=coloc.color1,
                    color2=coloc.color2,
                    tolerance=tolerance_value,
                )
                coloc.n_spots = len(coloc.spots)

        vid.coloc_all.spots = vid.coloc_grn_red.spots
        vid.coloc_all.n_spots = vid.coloc_grn_red.n_spots

    def displaySpotsSingle(self, channel):
        """
        Displays colocalized spot for a single video.
        """
        self.getCurrentListObject()

        if self.currName is not None:
            self.colocalizeSpotsSingleVideo(channel)
            self.refreshPlot()

    def colocalizeSpotsAllVideos(self):
        """
        Colocalizes spots for all videos, with the same threshold. Use this
        method instead for progress bar.
        """
        progressbar = ProgressBar(
            loop_len=len(self.data.videos.keys()), parent=self
        )
        for name in self.data.videos.keys():
            self.currName = name
            for c in "green", "red":
                self.colocalizeSpotsSingleVideo(c)
            progressbar.increment()

        self.resetCurrentName()
        self.refreshPlot()

    def getTracesSingleVideo(self):
        """
        Gets traces from colocalized ROIs, for a single video.
        """
        if self.currName is None:
            return
        vid = self.currentVideo()

        # Clear all traces previously held traces whenever this is called
        vid.traces = {}

        if vid.coloc_grn_red.spots is None:
            for c in "green", "red":
                self.colocalizeSpotsSingleVideo(c)
        else:
            for n, *row in vid.coloc_grn_red.spots.itertuples():
                yx_grn, yx_red = lib.misc.pairwise(row)

                trace = self.newTraceFromVideo(n)

                # Green
                if vid.grn.exists and yx_grn is not None:
                    masks_grn = lib.imgdata.circle_mask(
                        yx=yx_grn, indices=vid.indices, **gvars.cmask_p
                    )
                    (
                        trace.grn.int,
                        trace.grn.bg,
                    ) = lib.imgdata.tiff_stack_intensity(
                        vid.grn.raw, *masks_grn, raw=True
                    )

                # Red
                masks_red = lib.imgdata.circle_mask(
                    yx=yx_red, indices=vid.indices, **gvars.cmask_p
                )
                trace.red.int, trace.red.bg = lib.imgdata.tiff_stack_intensity(
                    vid.red.raw, *masks_red, raw=True
                )

                # Acceptor (if FRET)
                if vid.acc.exists:
                    (
                        trace.acc.int,
                        trace.acc.bg,
                    ) = lib.imgdata.tiff_stack_intensity(
                        vid.acc.raw, *masks_red, raw=True
                    )
                    trace.fret = lib.math.calc_E(trace.get_intensities())
                    trace.stoi = lib.math.calc_S(trace.get_intensities())

                trace.frames = np.arange(1, len(trace.red.int) + 1)
                trace.frames_max = max(trace.frames)

    def getTracesAllVideos(self):
        """
        Gets the traces from all videos that have colocalized ROIs.
        """
        for name in self.data.videos.keys():
            self.currName = name
            for c in "green", "red":
                self.colocalizeSpotsSingleVideo(c)

            self.getTracesSingleVideo()

        self.resetCurrentName()

    def savePlot(self):
        """
        Saves plot with colors suitable for export (e.g. white background).
        """
        self.setSavefigrcParams()

        if self.currName is not None:
            self.canvas.defaultImageName = self.currName
            self.canvas.defaultImageName = self.currName
        else:
            self.canvas.defaultImageName = "Blank"

        for ax in self.canvas.axes_all:
            ax.tick_params(axis="both", colors=gvars.color_hud_black)
            for spine in ax.spines.values():
                spine.set_edgecolor(gvars.color_hud_black)

        self.canvas.toolbar.save_figure()

        # Reset figure colors after plotting
        for ax in self.canvas.axes_all:
            ax.tick_params(axis="both", colors=gvars.color_hud_white)
            for spine in ax.spines.values():
                spine.set_edgecolor(gvars.color_hud_white)

        self.refreshPlot()

    def setupPlot(self):
        """
        Set up plot for MainWindow.
        """
        self.canvas.fig.set_facecolor(gvars.color_hud_black)
        self.canvas.fig.set_edgecolor(gvars.color_hud_white)

        for ax in self.canvas.axes_all:
            for spine in ax.spines.values():
                spine.set_edgecolor(gvars.color_hud_white)

    def refreshPlot(self):
        """
        Refreshes plot with selected list item.
        """
        for ax in self.canvas.axes_all:
            ax.tick_params(axis="both", colors=gvars.color_hud_white)
            ax.clear()

        if self.currName is not None:
            vid = self.currentVideo()
            roi_radius = vid.roi_radius

            contrast_lo = (
                self.ui.contrastBoxLoGreen,
                self.ui.contrastBoxLoRed,
            )
            keys = (
                gvars.key_contrastBoxHiGrnVal,
                gvars.key_contrastBoxHiRedVal,
            )
            contrast_hi = (
                self.ui.contrastBoxHiGreen,
                self.ui.contrastBoxHiRed,
            )
            # Rescale values according to UI first
            # might break image if too high
            sensitivity = 250

            for c, lo, hi in zip(
                vid.channels, contrast_lo, contrast_hi
            ):  # type: ImageChannel, QDoubleSpinBox, QDoubleSpinBox
                clip_lo = float(lo.value() / sensitivity)
                clip_hi = float(hi.value() / sensitivity)
                c.rgba = lib.imgdata.rescale_intensity(
                    c.mean, range=(clip_lo, clip_hi)
                )

            # Save contrast settings
            for hi, cfg in zip(contrast_hi, keys):
                self.setConfig(cfg, hi.value())

            # Single channels
            for img, ax in zip(vid.channels, self.canvas.axes_single):
                if img.rgba is not None:
                    # Avoid imshow showing blank if clipped too much
                    if np.isnan(img.rgba).any():
                        img.rgba.fill(1)
                    ax.imshow(img.rgba, cmap=img.cmap, vmin=0)
                else:
                    lib.plotting.empty_imshow(ax)

            c1, c2 = vid.grn, vid.red
            if c1.rgba is not None and c2.rgba is not None:
                self.canvas.ax_grn_red.imshow(
                    lib.imgdata.light_blend(
                        c1.rgba, c2.rgba, cmap1=c1.cmap, cmap2=c2.cmap
                    ),
                    vmin=0,
                )
            else:
                lib.plotting.empty_imshow(self.canvas.axes_blend.ax)

            for ax in self.canvas.axes_all:
                ax.set_xticks(())
                ax.set_yticks(())

            # Green spots
            if vid.grn.n_spots > 0:
                lib.plotting.plot_rois(
                    vid.grn.spots,
                    self.canvas.ax_grn,
                    color=gvars.color_white,
                    radius=roi_radius,
                )

            # Red spots
            if vid.red.n_spots > 0:
                lib.plotting.plot_rois(
                    vid.red.spots,
                    self.canvas.ax_red,
                    color=gvars.color_white,
                    radius=roi_radius,
                )

            # Colocalized spots
            if vid.coloc_grn_red.spots is not None:
                lib.plotting.plot_roi_coloc(
                    vid.coloc_grn_red.spots,
                    img_ax=self.canvas.ax_grn_red,
                    color1=gvars.color_green,
                    color2=gvars.color_red,
                    radius=roi_radius,
                )

        else:
            for ax in self.canvas.axes_all:
                lib.plotting.empty_imshow(ax)

        self.canvas.draw()
        self.refreshInterface()

    def refreshInterface(self):
        """
        Repaints UI labels to match the current plot shown.
        """
        if self.currName is not None:
            vid = self.currentVideo()

            channels = (
                vid.coloc_grn_red,
                vid.grn,
                vid.red,
            )

            for channel, label in zip(channels, self.labels):
                label.setText(str(channel.n_spots))

            self.ui.spotsGrnSpinBox.setDisabled(not vid.grn.exists)
            self.ui.spotsRedSpinBox.setDisabled(not vid.red.exists)

            if self.batchLoaded:
                self.disableSpinBoxes(("green", "red",))

        else:
            for label in self.labels:
                label.setText(str("-"))

        # Repaint all labels
        for label in self.labels:
            label.repaint()

    def findTracesAndShow(self):
        """
        Gets the name of currently stored traces and puts them into the
        trace listView.
        """
        if len(self.data.traces) == 0:
            # Load all traces into their respective videos, and generate a
            # list of traces
            self.getTracesAllVideos()
            currently_loaded = self.returnCurrentListViewNames()
            # Iterate over all filenames and add to list
            for (name, trace) in self.data.traces.items():
                if name in currently_loaded:
                    continue
                item = QStandardItem(trace.name)
                TraceWindow_.listModel.appendRow(item)
                TraceWindow_.currName = trace.name
                item.setCheckable(True)
            TraceWindow_.selectListViewTopRow()
        else:
            TraceWindow_.show()

        TraceWindow_.refreshPlot()
        TraceWindow_.show()

    def newTraceFromContainer(self, trace, n):
        """
        Creates an empty video object to load traces into by transplanting a
        list of loaded TraceContainers
        """
        tracename = self.currName + "_" + str(n)

        trace.currentVideo = self.currName
        trace.name = tracename
        trace.n = n

        if tracename not in self.data.videos[self.currName].traces:
            self.data.videos[self.currName].traces[tracename] = trace

    def disableSpinBoxes(self, channel):
        """
        Disables all spinboxes. Order must be as below, or a Qt bug will
        re-enable the boxes.
        """
        if "red" in channel:
            self.ui.spotsRedSpinBox.setDisabled(True)
            self.ui.spotsRedSpinBox.repaint()

        if "green" in channel:
            self.ui.spotsGrnSpinBox.setDisabled(True)
            self.ui.spotsGrnSpinBox.repaint()

    def _debug(self):
        """Debug for MainWindow."""
        pass


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
        item = self.listModel.itemFromIndex(index)  # type: QStandardItem

        if self.currName is not None:
            self.currentTrace().is_checked = (
                True if item.checkState() == Qt.Checked else False
            )

        if item.checkState() in (Qt.Checked, Qt.Unchecked):
            HistogramWindow_.getHistogramData()
            HistogramWindow_.gauss_params = None
            if HistogramWindow_.isVisible():
                HistogramWindow_.refreshPlot()

            TransitionDensityWindow_.setPooledLifetimes()
            TransitionDensityWindow_.setClusteredTransitions()

            if TransitionDensityWindow_.isVisible():
                TransitionDensityWindow_.refreshPlot()

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

    def returnContainerInstance(self):
        """Returns appropriate data container for implemented windows"""
        return self.data.traces

    def correctionFactorInspector(self):
        """
        Opens the inspector (modal) window to set correction factors
        """
        if self.isVisible():
            CorrectionFactorInspector_.show()

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

        if TransitionDensityWindow_.isVisible():
            TransitionDensityWindow_.refreshPlot()

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

    def sortListByCondition(self, setup):
        """
        Checks all traces where the red fluorophore (acceptor or direct
        excitation) channel is bleached.
        """
        # Only used for inspector currently
        params = self.inspector.returnInspectorValues()

        for index in range(self.listModel.rowCount()):
            item = self.listModel.item(index)
            trace = self.getTrace(item)

            if setup in ("red", "green"):
                channel = trace.grn if setup == "green" else trace.red
                pass_all = channel.bleach is not None and channel.bleach >= 10

            elif setup == "S":
                S = trace.stoi[:10]
                cond1 = 0.4 < np.median(S) < 0.6
                cond2 = all((S > 0.3) & (S < 0.7))
                pass_all = all((cond1, cond2))

            elif setup == "advanced":
                conditions = []

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
        self.inspector.setInspectorConfigs(params)

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


class DensityWindowInspector(SheetInspector):
    """
    Window for managing slider sheets for windows that use density plots. Use similar template for other sliders.
    See also refreshPlot for usage, because it needs a try/except clause to function properly.
    """

    def __init__(self, parent):
        super().__init__(parent=parent)

        self.getConfig = parent.getConfig
        self.setConfig = parent.setConfig
        self.ui = Ui_DensityWindowInspector()
        self.ui.setupUi(self)

        if isinstance(
            parent, HistogramWindow
        ):  # Avoids an explicit reference in parent class, for easier copy-paste
            self.keys = gvars.keys_hist
            parent.inspector = self
        elif isinstance(parent, TransitionDensityWindow):
            self.keys = gvars.keys_tdp
            parent.inspector = self
        else:
            raise NotImplementedError

        self.setUi()
        self.connectUi(parent)

    def connectUi(self, parent):
        """Connect Ui to parent functions"""
        if hasattr(
            parent, "canvas"
        ):  # Avoid refreshing canvas before it's even instantiated on the parent
            for slider in (
                self.ui.smoothingSlider,
                self.ui.resolutionSlider,
                self.ui.colorSlider,
                self.ui.pointAlphaSlider,
            ):

                # Avoids an explicit reference in parent class,
                # for easier copy-paste
                if isinstance(parent, HistogramWindow) or isinstance(
                    parent, TransitionDensityWindow
                ):
                    slider.valueChanged.connect(parent.refreshPlot)
                else:
                    raise NotImplementedError

            self.ui.overlayCheckBox.clicked.connect(parent.refreshPlot)

    def setUi(self):
        """
        Setup UI according to last saved preferences.
        """
        bandwidth, resolution, n_colors, overlay_pts, pts_alpha = [
            self.getConfig(key) for key in self.keys
        ]

        self.ui.smoothingSlider.setValue(bandwidth)
        self.ui.resolutionSlider.setValue(resolution)
        self.ui.colorSlider.setValue(n_colors)
        self.ui.overlayCheckBox.setChecked(bool(overlay_pts))
        self.ui.pointAlphaSlider.setValue(pts_alpha)

    def returnInspectorValues(self):
        """
        Returns values from inspector window to be used in parent window.
        """
        bandwidth = self.ui.smoothingSlider.value()
        resolution = self.ui.resolutionSlider.value()
        n_colors = self.ui.colorSlider.value()
        overlay_pts = self.ui.overlayCheckBox.isChecked()
        pts_alpha = self.ui.pointAlphaSlider.value()

        return bandwidth, resolution, n_colors, overlay_pts, pts_alpha


class CorrectionFactorInspector(SheetInspector):
    def __init__(self, parent):
        super().__init__(parent=parent)

        self.getConfig = parent.getConfig

        self.ui = Ui_CorrectionFactorInspector()
        self.ui.setupUi(self)

        self.alphaFactor = self.getConfig(gvars.key_alphaFactor)
        self.deltaFactor = self.getConfig(gvars.key_deltaFactor)

        self.ui.alphaFactorBox.setValue(self.alphaFactor)
        self.ui.deltaFactorBox.setValue(self.deltaFactor)

        self.ui.alphaFactorBox.valueChanged.connect(
            partial(self.setCorrectionFactors, "alpha")
        )
        self.ui.deltaFactorBox.valueChanged.connect(
            partial(self.setCorrectionFactors, "delta")
        )

    def setCorrectionFactors(self, factor):
        """
        Sets the global correction factors
        """
        parent = self.parent
        self.alphaFactor = self.ui.alphaFactorBox.value()
        self.deltaFactor = self.ui.deltaFactorBox.value()

        if factor == "alpha":
            parent.setConfig(gvars.key_alphaFactor, self.alphaFactor)
        elif factor == "delta":
            parent.setConfig(gvars.key_deltaFactor, self.deltaFactor)
        if TraceWindow_.isVisible():
            TraceWindow_.refreshPlot()

        if HistogramWindow_.isVisible():
            HistogramWindow_.refreshPlot()

    def showEvent(self, event):
        self.ui.alphaFactorBox.setValue(self.alphaFactor)
        self.ui.deltaFactorBox.setValue(self.deltaFactor)
        self.ui.alphaFactorBox.repaint()
        self.ui.deltaFactorBox.repaint()


class TraceWindowInspector(SheetInspector):
    """
    Inspector for the advanced sorting sheet.
    """

    def __init__(self, parent):
        super().__init__(parent=parent)

        self.getConfig = parent.getConfig
        self.setConfig = parent.setConfig
        self.ui = Ui_TraceWindowInspector()
        self.ui.setupUi(self)
        self.keys = gvars.keys_trace

        self.spinBoxes = (
            self.ui.spinBoxStoiLo,
            self.ui.spinBoxStoiHi,
            self.ui.spinBoxFretLo,
            self.ui.spinBoxFretHi,
            self.ui.spinBoxMinFrames,
            self.ui.spinBoxConfidence,
            self.ui.spinBoxDynamics,
        )

        if isinstance(parent, TraceWindow):
            parent.inspector = self

        self.connectUi(parent)
        self.setUi()

    def setUi(self):
        """Setup UI according to last saved preferences."""
        for spinBox, key in zip(self.spinBoxes, self.keys):
            spinBox.setValue(self.getConfig(key))

    def connectUi(self, parent):
        """Connect Ui to parent functions."""
        self.ui.pushButtonFind.clicked.connect(self.findPushed)

    def findPushed(self):
        """Sorts list by conditions set by spinBoxes and closes."""
        TraceWindow_.sortListByCondition(setup="advanced")
        if HistogramWindow_.isVisible():
            HistogramWindow_.refreshPlot(autofit=True)
        self.close()

    def returnInspectorValues(self):
        """Returns inspector values to parent window"""
        (
            S_med_lo,
            S_med_hi,
            E_med_lo,
            E_med_hi,
            min_n_frames,
            confidence,
            dynamics,
        ) = [spinBox.value() for spinBox in self.spinBoxes]

        bleached_only = self.ui.checkBoxBleach.isChecked()
        return (
            S_med_lo,
            S_med_hi,
            E_med_lo,
            E_med_hi,
            min_n_frames,
            confidence,
            dynamics,
            bleached_only,
        )


class AppContext(ApplicationContext):
    """
    Entry point for running the application. Only loads resources and holds
    the event loop.
    """

    def __init__(self):
        super().__init__()

        self.keras_two_channel_model = None
        self.keras_three_channel_model = None
        self.config = None
        self.app_version = None
        self.load_resources()

    def load_resources(self):
        """
        Loads initial resources from disk to application
        """
        # model_experimental is better but undocumented
        self.keras_two_channel_model = load_model(
            self.get_resource("FRET_2C_experimental.h5")
        )  # type: Model
        self.keras_three_channel_model = load_model(
            self.get_resource("FRET_3C_experimental.h5")
        )  # type: Model
        self.config = ConfigObj(self.get_resource("config.ini"))

    def getConfig(self, key: str) -> Union[bool, str, float]:
        """
        Shortcut for reading config from file and returning key
        """
        self.config.reload()

        value = self.config.get(key)

        if value is None:
            qWarning(
                "{} = {} returned NoneType. Ensure that the correct value is "
                "set in the GUI".format(key, value)
            )
            return 0

        # To handle 0/1/True/False as ints
        if value in gvars.boolMaps:
            value = gvars.boolMaps[value]
        else:
            try:
                value = float(value)
            except ValueError:
                value = str(value)
        return value

    def setConfig(self, key, value):
        """
        Shortcut for writing config to file.
        """
        self.config[key] = value
        self.config.write()

    @staticmethod
    def focusWindow(window):  # TODO: this method could also live in lib.misc
        """
        Focuses selected window and brings it to front.
        """
        window.show()
        window.setWindowState(
            window.windowState() & ~Qt.WindowMinimized | Qt.WindowActive
        )
        window.raise_()
        window.activateWindow()

    def bringToFront(self, window):
        """
        Select which windows to focus and bring to front
        """
        if window == "MainWindow":
            self.focusWindow(MainWindow_)

        elif window == "TraceWindow":
            TraceWindow_.refreshPlot()
            self.focusWindow(TraceWindow_)

        elif window == "HistogramWindow":
            HistogramWindow_.refreshPlot()
            self.focusWindow(HistogramWindow_)

        elif window == "TransitionDensityWindow":
            TransitionDensityWindow_.refreshPlot()
            self.focusWindow(TransitionDensityWindow_)

        elif window == "SimulatorWindow":
            SimulatorWindow_.refreshPlot()
            self.focusWindow(SimulatorWindow_)
        else:
            raise ValueError("Window doesn't exist")

    def assign(self):
        """
        Assigns resources and functions to the right windows
        before they're instantiated
        """
        # Assigns global config variables to BaseWindow and PreferencesWindow
        # so they both point to the base functions in AppContext
        for Window in (BaseWindow, PreferencesWindow):
            Window.config = self.config
            Window.processEvents = self.app.processEvents
            Window.getConfig = self.getConfig
            Window.setConfig = self.setConfig
            Window.bringToFront = self.bringToFront

        AboutWindow.app_version = self.config["appVersion"]
        BaseWindow.keras_two_channel_model = self.keras_two_channel_model
        BaseWindow.keras_three_channel_model = self.keras_three_channel_model

    def run(self):
        """
        Returns main loop exit code to be put into sys.exit()
        """
        return self.app.exec_()


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

    def formatPlotInspector(self):
        """
        Opens the inspector (modal) window to format the current plot
        """
        if self.isActiveWindow():
            self.inspector.show()

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

            params = self.inspector.returnInspectorValues()
            self.inspector.setInspectorConfigs(params)

            self.setClusteredTransitions()
            self.plotTransitionDensity(params)
            self.plotHistograms()

            # Adjust defaults
            self.plotDefaultElementsLeft()
            self.plotDefaultElementsRight()
            self.plotDefaultElements()

            self.canvas.draw()

        except AttributeError:
            pass

    def showEvent(self, QShowEvent):
        """
        No idea why this window writes incorrect correction factors,
        but this hotfix overrides the bug
        """
        # TODO: fix the real culprit somewhere in TransitionDensityWindow
        self.setConfig(
            gvars.key_alphaFactor, CorrectionFactorInspector_.alphaFactor
        )
        self.setConfig(
            gvars.key_deltaFactor, CorrectionFactorInspector_.deltaFactor
        )

    def _debug(self):
        self.refreshPlot()


if __name__ == "__main__":
    # Fixes https://github.com/mherrmann/fbs/issues/87
    multiprocessing.freeze_support()
    # Create the app
    # Load app
    ctxt = AppContext()
    ctxt.assign()

    # Windows
    MainWindow_ = MainWindow()
    TraceWindow_ = TraceWindow()
    HistogramWindow_ = HistogramWindow()
    TransitionDensityWindow_ = TransitionDensityWindow()
    SimulatorWindow_ = SimulatorWindow()

    # Inspector sheets
    HistogramInspector_ = DensityWindowInspector(HistogramWindow_)
    TransitionDensityInspector_ = DensityWindowInspector(
        TransitionDensityWindow_
    )
    CorrectionFactorInspector_ = CorrectionFactorInspector(TraceWindow_)
    TraceWindowInspector_ = TraceWindowInspector(TraceWindow_)

    #

    exit_code = ctxt.run()
    sys.exit(exit_code)
