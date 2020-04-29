import os
import time
from functools import partial
from typing import Dict, Union

import matplotlib
import numpy as np
import pandas as pd
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from global_variables import GlobalVariables as gvars
from lib.container import DataContainer, TraceContainer
from mpl_layout import PlotWidget
from ui._AboutWindow import Ui_About
from ui._MenuBar import Ui_MenuBar
from ui._PreferencesWindow import Ui_Preferences
from widgets.misc import ExportDialog, ListView


class BaseWindow(QMainWindow):
    """
    Superclass for shared window functions.
    """

    # Place here to share state between all windows
    data = DataContainer()

    # Every window is tracked globally, so they can all reference each other
    windows = {}

    keras_two_channel_model = None
    keras_three_channel_model = None
    config = None  # Overidden by AppContext

    def __init__(self):
        super().__init__()
        self.trackWindowInstance()

        # Each window tracks its own inspector instances
        self.inspectors = {}

        self.ui = Ui_MenuBar()
        self.ui.setupUi(self)

        self.currDir = None
        self.currName = None
        self.currRow = None
        self.batchLoaded = False

        self.setupMenuBarActions()
        self.enablePerWindow()

        self.PreferencesWindow_ = PreferencesWindow()
        self.AboutWindow_ = AboutWindow()

    def trackWindowInstance(self):
        """
        Tracks every new window that is instantiated, so they can be accessed
        in subclasses
        """
        self.__class__.windows[self.__class__.__name__] = self

    def setupMenuBarActions(self):
        """
        Setup menubar actions to be inherited (and thus shared) between all
        windows. Override BaseWindow methods to override action, or disable
        for certain windows with isinstance(self, WindowType).
        """

        # Special (macOS only)
        self.ui.actionAbout.triggered.connect(self.showAboutWindow)

        # This will show up correctly under the application name on MacOS,
        # as  long as the key is Cmd+,
        self.ui.actionPreferences.triggered.connect(self.showPreferenceWindow)

        # File
        # Open file
        self.ui.actionOpen.triggered.connect(self.openFile)
        # Save plot
        self.ui.actionSave.triggered.connect(self.savePlot)
        # Close window
        self.ui.actionClose.triggered.connect(self.close)
        # Exports CHECKED traces
        self.ui.actionExport_Selected_Traces.triggered.connect(
            partial(self.exportTracesToAscii, True)
        )
        # Exports ALL traces
        self.ui.actionExport_All_Traces.triggered.connect(
            partial(self.exportTracesToAscii, False)
        )
        # Exports colocalized spots
        self.ui.actionExport_Colocalization.triggered.connect(
            self.exportColocalization
        )
        self.ui.actionExport_Correction_Factors.triggered.connect(
            self.exportCorrectionFactors
        )
        self.ui.actionExport_ES_Histogram_Data.triggered.connect(
            self.exportHistogramData
        )
        self.ui.actionExport_Transition_Density_Data.triggered.connect(
            self.exportTransitionDensityData
        )

        # Edit
        # Delete from listView
        self.ui.actionRemove_File.triggered.connect(self.deleteSingleListObject)
        self.ui.actionRemove_All_Files.triggered.connect(
            self.deleteAllListObjects
        )

        self.ui.actionCheck_All_Traces.triggered.connect(self.checkAll)
        self.ui.actionUncheck_All_Traces.triggered.connect(self.unCheckAll)

        # View
        self.ui.actionFormat_Plot.triggered.connect(
            self.showDensityWindowInspector
        )
        # ---
        self.ui.actionAdvanced_Sort.triggered.connect(
            self.showAdvancedSortInspector
        )
        self.ui.actionSort_by_Ascending.triggered.connect(
            self.sortListAscending
        )
        self.ui.actionSort_by_Red_Bleach.triggered.connect(
            partial(self.sortListByCondition, "red")
        )
        self.ui.actionSort_by_Green_Bleach.triggered.connect(
            partial(self.sortListByCondition, "green")
        )
        self.ui.actionSort_by_Equal_Stoichiometry.triggered.connect(
            partial(self.sortListByCondition, "S")
        )

        # Analyze

        # Colocalize all spots with current settings
        self.ui.actionColocalize_All.triggered.connect(
            self.colocalizeSpotsAllVideos
        )
        self.ui.actionClear_Traces.triggered.connect(self.clearTraces)
        self.ui.actionFind_Show_Traces.triggered.connect(self.findTracesAndShow)
        self.ui.actionClear_and_Rerun.triggered.connect(self.clearTraceAndRerun)

        # ---
        self.ui.actionColor_Red.triggered.connect(self.colorListObjectRed)
        self.ui.actionColor_Yellow.triggered.connect(self.colorListObjectYellow)
        self.ui.actionColor_Green.triggered.connect(self.colorListObjectGreen)
        self.ui.actionClear_Color.triggered.connect(self.resetListObjectColor)
        self.ui.actionClear_All_Colors.triggered.connect(
            self.resetListObjectColorAll
        )
        # ---
        self.ui.actionCorrectionFactorsWindow.triggered.connect(
            self.showCorrectionFactorInspector
        )
        self.ui.actionGet_alphaFactor.triggered.connect(
            partial(self.getCorrectionFactors, "alpha")
        )
        self.ui.actionGet_deltaFactor.triggered.connect(
            partial(self.getCorrectionFactors, "delta")
        )
        self.ui.actionClear_Correction_Factors.triggered.connect(
            self.clearCorrectionFactors
        )

        for f in (partial(self.triggerBleach, "red"), self.refreshPlot):
            self.ui.actionSelect_Bleach_Red_Channel.triggered.connect(f)

        for f in (partial(self.triggerBleach, "green"), self.refreshPlot):
            self.ui.actionSelect_Bleach_Green_Channel.triggered.connect(f)

        # self.ui.actionFit_Hmm_Current.triggered.connect(
        #     partial(self.fitSingleTraceHiddenMarkovModel, True)
        # )
        for f in (self.fitCheckedTracesHiddenMarkovModel, self.refreshPlot):
            self.ui.actionFit_Hmm_Selected.triggered.connect(f)

        for f in (partial(self.classifyTraces, True), self.refreshPlot):
            self.ui.actionPredict_Selected_Traces.triggered.connect(f)

        for f in (partial(self.classifyTraces, False), self.refreshPlot):
            self.ui.actionPredict_All_traces.triggered.connect(f)

        for f in (self.clearAllClassifications, self.refreshPlot):
            self.ui.actionClear_All_Predictions.triggered.connect(f)

        # Window
        # Minimizes window
        self.ui.actionMinimize.triggered.connect(self.showMinimized)
        self.ui.actionMainWindow.triggered.connect(
            partial(self.bringToFront, "MainWindow")
        )
        self.ui.actionTraceWindow.triggered.connect(
            partial(self.bringToFront, "TraceWindow")
        )
        self.ui.actionHistogramWindow.triggered.connect(
            partial(self.bringToFront, "HistogramWindow")
        )
        self.ui.actionTransitionDensityWindow.triggered.connect(
            partial(self.bringToFront, "TransitionDensityWindow")
        )
        self.ui.actionTraceSimulatorWindow.triggered.connect(
            partial(self.bringToFront, "SimulatorWindow")
        )

        # Help
        # Open URL in help menu
        self.ui.actionGet_Help_Online.triggered.connect(self.openUrl)
        self.ui.actionDebug.triggered.connect(self._debug)

    # not used fcn, uncommented to make sure this holds
    # def disableOpenFileMenu(self):
    #     """
    #     Disables the Open menu to prevent loading additional videos if
    #     setting has been changed
    #     """
    #     self.ui: Ui_MenuBar
    #
    #     if isinstance(self, MainWindow):
    #         self.ui.actionOpen.setEnabled(False)

    def focusWindow(self):
        """
        Focuses selected window and brings it to front.
        """
        self.show()
        self.setWindowState(
            self.windowState() & ~Qt.WindowMinimized | Qt.WindowActive
        )
        self.raise_()
        self.activateWindow()

    @classmethod
    def bringToFront(cls, window):
        """
        Select which windows to focus and bring to front
        """
        for instance in cls.windows.keys():
            if instance == window:
                cls.windows[instance].focusWindow()

    def resetCurrentName(self):
        """
        Resets the current index that was used for currName object() iteration.
        """
        item = self.listModel.itemFromIndex(self.listView.currentIndex())

        self.currName = item.text() if item is not None else None

    def setSavefigrcParams(self):
        """
        Sets up some global matplotlib savefig rcParams. See individual
        savePlot() methods for further customization.
        """
        if self.currDir is None:
            current_dir = self.getConfig(gvars.key_lastOpenedDir)
        else:
            current_dir = self.currDir

        matplotlib.rcParams["savefig.directory"] = current_dir
        matplotlib.rcParams["savefig.facecolor"] = "white"
        matplotlib.rcParams["savefig.edgecolor"] = gvars.color_hud_black
        matplotlib.rcParams["savefig.transparent"] = True

    def setupSplitter(self, layout, width_left=300, width_right=1400):
        """
        Sets up a splitter between the listView and the plotWidget to make
        the list resizable. This has to be set up AFTER listView and
        plotWidget have been created, to place them in the layout
        """
        # noinspection PyAttributeOutsideInit
        self.splitter = QSplitter()
        self.splitter.addWidget(self.listView)
        self.splitter.addWidget(self.plotWidget)
        self.splitter.setSizes((width_left, width_right))
        layout.addWidget(self.splitter)

    def setupListView(self, use_layoutbox=True):
        """
        Sets up the left-hand listview.
        """
        # This is the underlying container of the listView. Deals with
        # handling the data labels.
        # noinspection PyAttributeOutsideInit
        self.listModel = QStandardItemModel()

        # override old listview
        # noinspection PyAttributeOutsideInit
        self.listView = ListView(self)
        self.listView.setEditTriggers(QAbstractItemView.NoEditTriggers)

        if use_layoutbox:
            self.ui.list_LayoutBox.addWidget(self.listView)

        # This is the UI part of the listView. Deals with the
        # user-interaction (e.g. clicking, deleting)
        self.listView.setModel(self.listModel)
        self.listView.checked.connect(self.onChecked)

        # Listview triggers
        # Need to grab the selection model inside the listview (which is
        # singleselect here)
        # noinspection PyAttributeOutsideInit
        self.listViewSelectionModel = self.listView.selectionModel()
        # to connect it to the selection
        self.listViewSelectionModel.currentChanged.connect(
            self.getCurrentListObject
        )
        # Don't connect listview twice!
        self.listView.clicked.connect(self.getCurrentListObject)

    @pyqtSlot(QModelIndex)
    def onChecked(self, index):
        pass

    @pyqtSlot(QListView)
    def keyPressEvent(self, QKeyEvent):
        if QKeyEvent.key() in [Qt.UpArrow, Qt.DownArrow]:
            self.refreshPlot()

    def returnCurrentListViewNames(self):
        """
        Creates a list of current objects in ListView, to avoid duplicate
        loading
        """
        in_listModel = []
        for index in range(self.listModel.rowCount()):
            item = self.listModel.item(index)
            name = item.text()
            in_listModel.append(name)
        return in_listModel

    def getCurrentListObject(self):
        """
        Obtains single list object details.
        """
        container = self.returnContainerInstance()
        currentIndex = self.listView.selectedIndexes()

        try:
            for obj_index in currentIndex:
                item = self.listModel.itemFromIndex(obj_index)
                row = item.row()
                index = self.listModel.index(row, 0)
                name = self.listModel.data(index)
                self.currName = name
                self.currRow = row

            if len(container) == 0:
                self.currName = None
                self.currRow = None

            self.refreshPlot()

        except AttributeError:
            pass

    def deleteSingleListObject(self):
        """
        Deletes single list object.
        """
        container = self.returnContainerInstance()

        if len(container) != 0:
            container.pop(self.currName)

            if len(container) == 0:
                self.currName = None

            self.listModel.removeRow(self.currRow)
            self.getCurrentListObject()

    def deleteAllListObjects(self):
        """
        Deletes all files from listview.
        """
        self.listModel.clear()
        container = self.returnContainerInstance()
        container.clear()

        self.currName = None
        self.currRow = None
        self.batchLoaded = False

        self.refreshPlot()
        self.refreshInterface()

    def sortListByChecked(self):
        """
        Sorts the listModel with checked objects on top.
        Only works with checkable elements.
        """
        self.listModel.setSortRole(Qt.CheckStateRole)
        self.listModel.sort(Qt.Unchecked, Qt.DescendingOrder)

    def sortListAscending(self):
        """
        Sorts the listModel in ascending order.
        """
        self.listModel.setSortRole(Qt.AscendingOrder)
        self.listModel.sort(Qt.AscendingOrder)

    def clearListModel(self):
        """
        Clears the internal data of the listModel. Call this before total
        refresh of interface.listView.
        """
        self.listModel.removeRows(0, self.listModel.rowCount())

    def selectListViewTopRow(self):
        """
        Marks the first element in the listview.
        """
        self.listView.setCurrentIndex(self.listModel.index(0, 0))
        self.getCurrentListObject()

    def selectListViewRow(self, row):
        """
        Selects a specific row in the listview.
        """
        self.listView.setCurrentIndex(self.listModel.index(row, 0))

    def unCheckAll(self):
        """
        Unchecks all list elements. Attribute check because they override
        "select", which is normally reserved for text fields
        """
        if not hasattr(self, "listModel"):
            return

        for index in range(self.listModel.rowCount()):
            item = self.listModel.item(index)
            item.setCheckState(Qt.Unchecked)
            trace = self.getTrace(item)
            trace.is_checked = False

        histogram_window = self.windows[gvars.HistogramWindow]
        transition_density_window = self.windows[gvars.HistogramWindow]
        for window in histogram_window, transition_density_window:
            if window.isVisible():
                window.refreshPlot()

    def checkAll(self):
        """
        Checks all list elements.
        """
        if not hasattr(self, "listModel"):
            return

        for index in range(self.listModel.rowCount()):
            item = self.listModel.item(index)
            item.setCheckState(Qt.Checked)
            trace = self.getTrace(item)
            trace.is_checked = True

        histogram_window = self.windows[gvars.HistogramWindow]
        transition_density_window = self.windows[gvars.HistogramWindow]
        for window in histogram_window, transition_density_window:
            if window.isVisible():
                window.refreshPlot()

    # noinspection PyAttributeOutsideInit
    def setupFigureCanvas(self, ax_type, use_layoutbox=True, **kwargs):
        """
        Creates a canvas with a given ax layout.
        """
        self.plotWidget = PlotWidget(ax_type=ax_type, **kwargs)
        self.canvas = self.plotWidget.canvas

        if use_layoutbox:
            try:
                self.ui.mpl_LayoutBox.addWidget(self.canvas)
            except AttributeError:
                qWarning(
                    "Canvas must be placed in a Q-Layout named 'mpl_LayoutBox', "
                    "which is currently missing from the .interface file for "
                    "{}".format(type(self))
                )

        # make font editable in PDF
        matplotlib.rcParams["pdf.fonttype"] = 42
        matplotlib.rcParams["savefig.format"] = "pdf"

        self.refreshPlot()

    def openUrl(self):
        """
        Opens URL in default browser.
        """
        url = QUrl("https://github.com/komodovaran/DeepFRET-GUI/issues")
        if not QDesktopServices.openUrl(url):
            QMessageBox.warning(self, "Open Url", "Could not open url")

    def resetListObjectColor(self):
        """
        Resets color of currently selected listview object.
        """
        item = self.listModel.itemFromIndex(self.listView.currentIndex())
        item.setForeground(Qt.black)
        item.setBackground(Qt.white)

    def resetListObjectColorAll(self):
        """
        Resets color of all listview objects.
        """
        for index in range(self.listModel.rowCount()):
            item = self.listModel.itemFromIndex(index)
            item.setForeground(Qt.black)
            item.setBackground(Qt.white)

    def colorListObjectRed(self):
        """
        Colors the currently selected listview object red.
        """
        item = self.listModel.itemFromIndex(self.listView.currentIndex())
        item.setForeground(Qt.white)
        item.setBackground(Qt.darkRed)

    def colorListObjectYellow(self):
        """
        Colors the currently selected listview object yellow.
        """
        item = self.listModel.itemFromIndex(self.listView.currentIndex())
        item.setForeground(Qt.white)
        item.setBackground(Qt.darkYellow)

    def colorListObjectGreen(self):
        """
        Colors the currently selected listview object green.
        """
        item = self.listModel.itemFromIndex(self.listView.currentIndex())
        item.setForeground(Qt.white)
        item.setBackground(Qt.darkGreen)

    def showAboutWindow(self):
        """
        Shows "About" window.
        """
        self.AboutWindow_.show()

    def showPreferenceWindow(self):
        """
        Shows "Preference" window.
        """
        self.PreferencesWindow_.show()

    @staticmethod
    def newListItem(other, name):
        """
        Adds a new, single object to ListView.
        """
        item = QStandardItem(name)
        other.listModel.appendRow(item)
        other.listView.repaint()

    def openFile(self):
        """Override in subclass."""
        pass

    def savePlot(self):
        """Override in subclass."""
        pass

    @staticmethod
    def returnInfoHeader():
        """
        Generates boilerplate header text for every exported file.
        """
        exp_txt = "Exported by DeepFRET"
        date_txt = "Date: {}".format(time.strftime("%Y-%m-%d, %H:%M"))

        return exp_txt, date_txt

    @staticmethod
    def traceHeader(
        trace_df, exp_txt, id_txt, video_filename, bleaches_at, is_simulated
    ):
        """Returns the string to use for saving the trace as txt"""
        date_txt = "Date: {}".format(time.strftime("%Y-%m-%d, %H:%M"))
        bl_txt = "Bleaches at {}".format(bleaches_at)
        vid_txt = "Video filename: {}".format(video_filename)

        if is_simulated:
            exp_txt += " (Simulated)"

        return (
            "{0}\n"
            "{1}\n"
            "{2}\n"
            "{3}\n"
            "{4}\n\n"
            "{5}".format(
                exp_txt,
                date_txt,
                vid_txt,
                id_txt,
                bl_txt,
                trace_df.to_csv(index=False, sep="\t", na_rep="NaN"),
            )
        )

    def exportColocalization(self):
        """
        Exports colocalized ROI information for all currently colocalized
        videos. Does not automatically trigger ColocalizeAll beforehand.
        """
        exp_txt, date_txt = self.returnInfoHeader()

        directory = os.path.join(self.getLastOpenedDir(), "Colocalization.txt")

        path, _ = QFileDialog.getSaveFileName(
            self, directory=directory
        )  # type: str, str

        if path != "":
            if not path.split("/")[-1].endswith(".txt"):
                path += ".txt"

            columns = ("green", "red", "green/red", "filename")
            d = {c: [] for c in columns}  # type: Dict[str, list]

            assert columns[-1] == "filename"

            # for name, vid in MainWindow_.videos.items(): # old version
            for name, vid in self.data.videos.items():

                data = (
                    vid.grn.n_spots,
                    vid.red.n_spots,
                    vid.coloc_grn_red.n_spots,
                    name,
                )

                for c, da in zip(columns, data):
                    d[c].append(da)

            df = pd.DataFrame.from_dict(d)

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

    def exportTracesToAscii(self, checked_only=False):
        """
        Exports all traces as ASCII files to selected directory.
        Maintains compatibility with iSMS and older pySMS scripts.
        """
        traces = self.data.traces

        if checked_only:
            selected = [trace for trace in traces.values() if trace.is_checked]
        else:
            selected = [trace for trace in traces.values()]

        directory = self.getLastOpenedDir()

        diag = ExportDialog(init_dir=directory, accept_label="Export")

        if diag.exec():
            path = diag.selectedFiles()[0]
            self.processEvents()

            for trace in selected:
                trace.export_trace_to_txt(dir_to_join=path)

    def returnPooledCorrectionFactors(self):
        """
        Obtains global correction factors to be used
        """
        columns = "alpha", "delta", "filename"
        assert columns[-1] == "filename"

        d = {c: [] for c in columns}  # type: Dict[str, list]

        for trace in self.data.traces.values():
            if trace.a_factor is np.nan and trace.d_factor is np.nan:
                continue
            else:
                data = trace.a_factor, trace.d_factor, trace.name

                for c, da in zip(columns, data):
                    d[c].append(da)

        return pd.DataFrame.from_dict(d).round(4)

    def exportCorrectionFactors(self):
        """
        Exports all available correction factors for each trace to table.
        """

        exp_txt, date_txt = self.returnInfoHeader()

        directory = os.path.join(
            self.getLastOpenedDir(), "CorrectionFactors.txt"
        )

        path, _ = QFileDialog.getSaveFileName(
            self, directory=directory
        )  # type: str, str

        if path != "":
            if not path.split("/")[-1].endswith(".txt"):
                path += ".txt"

            df = self.returnPooledCorrectionFactors()

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

    def exportHistogramData(self):
        """
        Exports histogram data for selected traces if current window is not HistogramWindow
        Overridden in HistogramWindow
        """
        exp_txt, date_txt = self.returnInfoHeader()
        directory = (
            self.getConfig(gvars.key_lastOpenedDir) + "/E_S_Histogram.txt"
        )
        path, _ = QFileDialog.getSaveFileName(
            self, directory=directory
        )  # type: str, str
        if path != "":
            if not path.split("/")[-1].endswith(".txt"):
                path += ".txt"

            if self.data.histData.E is not None:
                df = pd.DataFrame(
                    {"E": self.data.histData.E, "S": self.data.histData.S}
                ).round(4)
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

    def setPooledLifetimes(self):
        """
        Return pooled lifetimes from Hidden Markov Model fits.
        """
        checkedTraces = [
            trace for trace in self.data.traces.values() if trace.is_checked
        ]
        self.data.histData.n_samples = len(checkedTraces)

        try:
            transitions = pd.concat(
                [trace.transitions for trace in checkedTraces]
            )
            transitions.reset_index(inplace=True)

            self.data.tdpData.state_lifetime = transitions["lifetime"]
            self.data.tdpData.state_before = transitions["e_before"]
            self.data.tdpData.state_after = transitions["e_after"]

        except ValueError:
            self.data.tdpData.state_lifetime = None
            self.data.tdpData.state_before = None
            self.data.tdpData.state_after = None

    def exportTransitionDensityData(self):
        """
        Exports TDP data for selected traces
        """
        exp_txt, date_txt = self.returnInfoHeader()

        directory = (
            self.getConfig(gvars.key_lastOpenedDir)
            + "/Transition_Densities.txt"
        )
        path, _ = QFileDialog.getSaveFileName(
            self, directory=directory
        )  # type: str, str

        self.setPooledLifetimes()

        if path != "":
            if not path.split("/")[-1].endswith(".txt"):
                path += ".txt"

            with open(path, "w") as f:
                df = self.data.tdpData.df

                f.write(
                    "{0}\n"
                    "{1}\n\n"
                    "{2}".format(
                        exp_txt,
                        date_txt,
                        df.to_csv(index=False, sep="\t", na_rep="NaN"),
                    )
                )

    def getLastOpenedDir(self):
        """
        Gets the most recent directory from the config file, and set this in new file dialogs
        """
        directory = self.getConfig(gvars.key_lastOpenedDir)
        if not os.path.exists(directory):
            directory = ""
        return directory

    """
    Functions below are not used in the BaseWindow, but they MUST be present
    in order to be triggered via the menubar. It's also a good way to keep
    naming consistent
    """

    def processEvents(self):
        """Set by ApplicationContext before any windows are initialized"""
        pass

    def showDensityWindowInspector(self):
        """Override in subclass."""
        pass

    def clearTraces(self):
        """Override in subclass."""
        pass

    def clearTraceAndRerun(self):
        pass

    def clearAllClassifications(self):
        """Override in subclass."""
        pass

    def colocalizeSpotsAllVideos(self):
        """Override in subclass."""
        pass

    def getTrace(self, name) -> TraceContainer:
        """Override in subclass."""

    def getTracesAllVideos(self):
        """Override in subclass."""
        pass

    def setupPlot(self):
        """Override in subclass."""
        pass

    def refreshPlot(self):
        """Override in subclass."""
        pass

    def refreshInterface(self):
        """Override in subclass."""
        pass

    def sortListByCondition(self, channel):
        """Override in subclass."""
        pass

    def getCorrectionFactors(self, factor):
        """Override in subclass."""
        pass

    def clearCorrectionFactors(self):
        """Override in subclass."""
        pass

    def showCorrectionFactorInspector(self):
        """
        Opens the inspector (modal) window to set correction factors.
        """
        pass

    def showAdvancedSortInspector(self):
        """Override in subclass"""
        pass

    def triggerBleach(self, color):
        """Override in subclass."""
        pass

    def fitCheckedTracesHiddenMarkovModel(self):
        """Override in subclass."""
        pass

    def classifyTraces(self, selected):
        """Override in subclass."""
        pass

    def batchOpen(self):
        """Override in subclass."""
        pass

    def findTracesAndShow(self):
        """Override in subclass."""
        pass

    def _debug(self):
        """Override in subclass."""
        pass

    def enablePerWindow(self):
        """
        Disables specific commands that should be unavailable for certain
        window types. Export commands should be accessible from all windows
        (no implicit behavior).
        Overridden in subclasses.
        """
        pass

    def returnContainerInstance(self):
        """
        Returns appropriate data container for implemented windows.
        Overridden in subclasses where relevant, TraceWindow and MainWindow
        """
        raise NotImplementedError

    def setConfig(self, key, value):
        """Overridden by AppContext"""
        pass

    def getConfig(self, key) -> Union[bool, str, float]:
        """Overridden by AppContext"""
        pass


class AboutWindow(QDialog):
    """
    'About this application' window with version numbering.
    """

    app_version = None  # This definition is overridden by AppContext

    def __init__(self):
        super().__init__()
        self.ui = Ui_About()
        self.ui.setupUi(self)
        self.setWindowTitle("")

        self.ui.label_APPNAME.setText(gvars.APPNAME)
        self.ui.label_APPVER.setText("version {}".format(self.app_version))
        self.ui.label_AUTHORS.setText(gvars.AUTHORS)
        self.ui.label_LICENSE.setText(gvars.LICENSE)


class PreferencesWindow(QDialog):
    """
    Editable preferences. If preferences shouldn't be editable (e.g. global
    color styles), set them in GLOBALs class.
    """

    # config = None  # This definition is overridden by AppContext

    def __init__(self):
        super().__init__()
        self.config.reload()
        self.setModal(True)
        self.ui = Ui_Preferences()
        self.ui.setupUi(self)

        self.globalCheckBoxes = (
            self.ui.checkBox_batchLoadingMode,
            self.ui.checkBox_unColocRed,
            self.ui.checkBox_illuCorrect,
            self.ui.checkBox_fitSpots,
            self.ui.checkBox_hmm_local,
            self.ui.checkBox_firstFrameIsDonor,
            self.ui.checkBox_donorLeft,
            self.ui.checkBox_medianPearsonCorr,
        )

        self.hmmRadioButtons = (
            self.ui.radioButton_hmm_fitE,
            self.ui.radioButton_hmm_fitDD,
        )

        if len(self.globalCheckBoxes) != len(gvars.keys_globalCheckBoxes):
            mismatch_error = (
                "Make sure widgets have the correct number of keys in gvars"
            )
            raise ValueError(mismatch_error)

        self.connectUi()

    def writeUiToConfig(self):
        """
        Checks all UI elements and writes their state to config
        """
        # All checkboxes
        for key, checkBox in zip(
            gvars.keys_globalCheckBoxes, self.globalCheckBoxes
        ):
            self.setConfig(key=key, value=checkBox.isChecked())

        # HMM radio buttons
        for hmmMode, radioButton in zip(
            gvars.keys_hmmModes, self.hmmRadioButtons
        ):
            if radioButton.isChecked():
                self.setConfig(key=gvars.key_hmmMode, value=hmmMode)

        # ROI detection tolerance
        self.setConfig(
            key=gvars.key_colocTolerance,
            value=self.ui.toleranceComboBox.currentText().lower(),
        )

        # Number of pairs to autodetect
        self.setConfig(
            key=gvars.key_autoDetectPairs,
            value=self.ui.spinBox_autoDetect.value(),
        )

        # BIC Strictness
        self.setConfig(
            key=gvars.key_hmmBICStrictness,
            value=self.ui.doubleSpinBox_hmm_BIC.value(),
        )

    def connectUi(self):
        """
        Connect widgets to config write functions
        """
        # All checkboxes
        for configKey, checkBox in zip(
            gvars.keys_globalCheckBoxes, self.globalCheckBoxes
        ):
            checkBox.clicked.connect(self.writeUiToConfig)

        # TODO: add that changing the hmmLocal checkbox should change the parameters for all traces
        #  both existing and new traces

        # HMM type radio buttons
        for radioButton in self.hmmRadioButtons:
            radioButton.clicked.connect(self.writeUiToConfig)

        # ROI detection tolerance
        self.ui.toleranceComboBox.currentTextChanged.connect(
            self.writeUiToConfig
        )

        # BIC strictness
        self.ui.doubleSpinBox_hmm_BIC.valueChanged.connect(self.writeUiToConfig)

        # Number of pairs to autodetect
        self.ui.spinBox_autoDetect.valueChanged.connect(self.writeUiToConfig)

        # Close modal window with Ctrl+W
        QShortcut(QKeySequence("Ctrl+W"), self, self.close)

    def loadConfigToGUI(self):
        """
        Read settings from the config file every time window is opened,
        and adjust UI elements accordingly.
        """
        self.config.reload()

        for configKey, checkBox in zip(
            gvars.keys_globalCheckBoxes, self.globalCheckBoxes
        ):
            checkBox.setChecked(bool(self.getConfig(configKey)))

        for radioButton, hmmMode in zip(
            self.hmmRadioButtons, gvars.keys_hmmModes
        ):
            if self.getConfig(gvars.key_hmmMode) == hmmMode:
                radioButton.setChecked(True)

        self.ui.toleranceComboBox.setCurrentText(
            self.getConfig(gvars.key_colocTolerance)
        )
        self.ui.spinBox_autoDetect.setValue(
            self.getConfig(gvars.key_autoDetectPairs)
        )
        self.ui.doubleSpinBox_hmm_BIC.setValue(
            self.getConfig(gvars.key_hmmBICStrictness)
        )

    def showEvent(self, QShowEvent):
        """
        Read settings on show.
        """
        self.loadConfigToGUI()
        self.show()

    def setConfig(self, key, value):
        """Overridden by AppContext"""
        pass

    def getConfig(self, key) -> Union[bool, str, float]:
        """Overridden by AppContext, only here for type annotations"""
        pass
