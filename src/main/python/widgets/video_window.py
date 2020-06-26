import os
from functools import partial

import numpy as np
from PyQt5.QtGui import QStandardItem
from PyQt5.QtWidgets import QDoubleSpinBox, QFileDialog

import lib.imgdata
import lib.math
import lib.utils
import lib.plotting
from global_variables import GlobalVariables as gvars
from lib.container import (
    ImageChannel,
    TraceChannel,
    TraceContainer,
    VideoContainer,
)
from lib.utils import timeit
from ui._MainWindow import Ui_MainWindow
from ui._MenuBar import Ui_MenuBar
from widgets.misc import ProgressBar
from widgets.base_window import BaseWindow
from widgets.trace_window import TraceWindow


class VideoWindow(BaseWindow):
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
                    uniqueName = lib.utils.generate_unique_name(
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
        trace_window = self.windows[gvars.TraceWindow]  # type: TraceWindow

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
            currently_loaded = trace_window.returnCurrentListViewNames()
            # Iterate over all filenames and add to list
            for name in self.data.traces.keys():
                # If name is already in list, skip it
                if name in currently_loaded:
                    continue
                item = QStandardItem(name)
                item.setCheckable(True)
                trace_window.listModel.appendRow(item)

            trace_window.selectListViewTopRow()
            trace_window.getCurrentListObject()
            trace_window.show()

            self.batchLoaded = True
            self.refreshInterface()
            self.selectListViewTopRow()
            self.refreshPlot()

    @timeit
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
                yx_grn, yx_red = lib.utils.pairwise(row)

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
        trace_window = self.windows[gvars.TraceWindow]

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
                trace_window.listModel.appendRow(item)
                trace_window.currName = trace.name
                item.setCheckable(True)
            trace_window.selectListViewTopRow()
        else:
            trace_window.show()

        trace_window.refreshPlot()
        trace_window.show()

    def clearTraceAndRerun(self):
        """
        Clears everything and reruns on selected videos.
        Loads all traces into their respective videos, and generates a list
        of traces
        """
        trace_window = self.windows[gvars.TraceWindow]

        self.clearTraces()
        self.getTracesAllVideos()

        # Iterate over all filenames and add to list
        for (name, trace) in self.data.traces.items():
            item = QStandardItem(trace.name)
            trace_window.listModel.appendRow(item)
            trace_window.currName = trace.name
            item.setCheckable(True)

        if trace_window.isVisible():
            trace_window.selectListViewTopRow()
            trace_window.refreshPlot()

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
