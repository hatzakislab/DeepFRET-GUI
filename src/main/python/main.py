import multiprocessing

from about_window import AboutWindow

multiprocessing.freeze_support()

import os
import sys
from functools import partial
from typing import Dict

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
import matplotlib.colors
import matplotlib.figure
import matplotlib.gridspec
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import re
import scipy.stats
import scipy.signal
import scipy.optimize
import scipy.special
import warnings
import sklearn.preprocessing
from tensorflow_core.python.keras.models import load_model

from ui._MenuBar import Ui_MenuBar
from ui._MainWindow import Ui_MainWindow
from ui._TraceWindow import Ui_TraceWindow
from ui._PreferencesWindow import Ui_Preferences
from ui._HistogramWindow import Ui_HistogramWindow
from ui._TransitionDensityWindow import Ui_TransitionDensityWindow
from ui._CorrectionFactorInspector import Ui_CorrectionFactorInspector
from ui._DensityWindowInspector import Ui_DensityWindowInspector
from ui._TraceWindowInspector import Ui_TraceWindowInspector

from ui.misc import (
    ProgressBar,
    RestartDialog,
    SheetInspector,
    ListView,
    ExportDialog,
)
from lib.container import (
    ImageContainer,
    ImageChannel,
    TraceChannel,
    TraceContainer,
    MovieData,
)
from mpl_layout import PlotWidget

from fbs_runtime.application_context.PyQt5 import ApplicationContext



class PreferencesWindow(QDialog):
    """
    Editable preferences. If preferences shouldn't be editable (e.g. global
    color styles), set them in GLOBALs class.
    """

    def __init__(self):
        super().__init__()
        self.setModal(True)
        self.ui = Ui_Preferences()
        self.ui.setupUi(self)
        self.readConfigFromFile()
        self.restartDialog = RestartDialog(self)

        QShortcut(QKeySequence("Ctrl+W"), self, self.close)

        self.globalCheckBoxes = (
            self.ui.checkBox_batchLoadingMode,
            self.ui.checkBox_unColocRed,
            self.ui.checkBox_illuCorrect,
            self.ui.checkBox_fitSpots,
        )

        self.imgModeRadioButtons = (
            self.ui.radioButton_dual,
            self.ui.radioButton_2_col,
            self.ui.radioButton_2_col_inv,
        )

        self.imgModes = "dual", "2-color", "2-color-inv"
        self.boolMaps = {"True": 1, "1": 1, "False": 0, "0": 0}

        assert len(self.globalCheckBoxes) == len(gvars.keys_globalCheckBoxes)
        assert len(self.imgModeRadioButtons) == len(self.imgModes)

    def getConfig(self, key):
        """
        Shortcut for reading config from file and returning key
        """
        self.readConfigFromFile()
        value = self.config.get(key)

        if value is None:
            qWarning(
                "{} = {} returned NoneType. Ensure that the correct value is "
                "set in the GUI".format(key, value)
            )
            return 0

        # To handle 0/1/True/False as ints
        if value in self.boolMaps:
            value = self.boolMaps[value]
        else:
            try:
                value = float(value)
            except ValueError:
                value = str(value)
        return value

    def setConfig(self, key, new_value):
        """
        Shortcut for writing config to file.
        """
        self.config[key] = new_value
        self.writeConfigToFile()

    def readConfigFromFile(self):
        """
        Read getConfig.ini file.
        """
        self.config = ctxt.config

    def writeConfigToFile(self):
        """
        Renamed to fit nomenclature.
        """
        self.config.write()

    def applyConfigToGUI(self):
        """
        Read settings from the config file every time window is opened,
        and adjust UI accordingly.
        """
        self.readConfigFromFile()

        for configKey, checkBox in zip(
            gvars.keys_globalCheckBoxes, self.globalCheckBoxes
        ):
            checkBox.setChecked(bool(self.getConfig(configKey)))

        for radioButton, imgMode in zip(
            self.imgModeRadioButtons, self.imgModes
        ):
            if self.getConfig(gvars.key_imgMode) == imgMode:
                radioButton.setChecked(True)

        self.ui.toleranceComboBox.setCurrentText(
            self.getConfig(gvars.key_colocTolerance)
        )
        self.ui.spinBox_autoDetect.setValue(
            self.getConfig(gvars.key_autoDetectPairs)
        )
        self.currentImgMode = self.getConfig(gvars.key_imgMode)

    def setConfigFromGUI(self):
        """
        Write getConfig from the GUI when the preference window is closed.
        """
        for configKey, checkBox in zip(
            gvars.keys_globalCheckBoxes, self.globalCheckBoxes
        ):
            self.setConfig(configKey, checkBox.isChecked())

        # Imaging Modes
        newImgMode = None
        for radioButton, imgMode in zip(
            self.imgModeRadioButtons, self.imgModes
        ):
            if radioButton.isChecked():
                newImgMode = imgMode

        if newImgMode != self.getConfig(gvars.key_imgMode):
            self.restartDialog.exec()
            if self.restartDialog.status == "Restart":
                self.setConfig(gvars.key_imgMode, newImgMode)
                ctxt.app.exit()

        self.setConfig(
            gvars.key_colocTolerance,
            self.ui.toleranceComboBox.currentText().lower(),
        )
        self.setConfig(
            gvars.key_autoDetectPairs, self.ui.spinBox_autoDetect.value()
        )

        self.writeConfigToFile()

    def showEvent(self, QShowEvent):
        """
        Read settings on show.
        """
        self.applyConfigToGUI()

    def closeEvent(self, QCloseEvent):
        """
        Write settings on close.
        """
        self.setConfigFromGUI()

    def reject(self):
        """
        Override Esc reject as a regular closeEvent.
        """
        self.close()


class BaseWindow(QMainWindow):
    """
    Superclass for shared window functions.
    """

    def __init__(self):
        super().__init__()
        self.ui = Ui_MenuBar()
        self.ui.setupUi(self)

        self.currName = None
        self.currRow = None
        self.batchLoaded = False

        self.setupMenuBarActions()
        self.enablePerWindow()

        self.AboutWindow_ = AboutWindow()
        self.PreferencesWindow_ = PreferencesWindow()

        self.getConfig = self.PreferencesWindow_.getConfig
        self.setConfig = self.PreferencesWindow_.setConfig

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
            partial(self.exportTraces, True)
        )
        # Exports ALL traces
        self.ui.actionExport_All_Traces.triggered.connect(
            partial(self.exportTraces, False)
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
        self.ui.actionSelect_All.triggered.connect(self.checkAll)
        self.ui.actionDeselect_All.triggered.connect(self.unCheckAll)

        # View
        self.ui.actionFormat_Plot.triggered.connect(self.formatPlotInspector)
        # ---
        self.ui.actionAdvanced_Sort.triggered.connect(self.configInspector)
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
            self.colocalizeSpotsAllMovies
        )
        self.ui.actionClear_Traces.triggered.connect(self.clearTraces)
        self.ui.actionFind_Show_Traces.triggered.connect(self.findTracesAndShow)

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
            self.correctionFactorInspector
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
        self.ui.actionSelect_Bleach_Red_Channel.triggered.connect(
            partial(self.triggerBleach, "red")
        )
        self.ui.actionSelect_Bleach_Green_Channel.triggered.connect(
            partial(self.triggerBleach, "green")
        )
        self.ui.actionFit_Hmm_Current.triggered.connect(
            partial(self.fitSingleTraceHiddenMarkovModel, True)
        )
        self.ui.actionFit_Hmm_All.triggered.connect(
            self.fitCheckedTracesHiddenMarkovModel
        )
        self.ui.actionPredict_Selected_Traces.triggered.connect(
            partial(self.predictTraces, True)
        )
        self.ui.actionPredict_All_traces.triggered.connect(
            partial(self.predictTraces, False)
        )
        self.ui.actionClear_All_Predictions.triggered.connect(
            self.clearAllPredictions
        )

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

        # Help
        # Open URL in help menu
        self.ui.actionGet_Help_Online.triggered.connect(self.openUrl)
        self.ui.actionDebug.triggered.connect(self._debug)

    def enablePerWindow(self):
        """
        Disables specific commands that should be unavailable for certain
        window types. Export commands should be accessible from all windows
        (no implicit behavior).
        """
        self.ui: Ui_MenuBar

        enableMenus_MainWindow = (
            self.ui.actionRemove_File,
            self.ui.actionRemove_All_Files,
            self.ui.actionColocalize_All,
            self.ui.actionFind_Show_Traces,
            self.ui.actionClear_Traces,
            self.ui.actionClear_and_Rerun,
            self.ui.actionSelect_All,
            self.ui.actionDeselect_All,
        )

        enableMenus_TraceWindow = (
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
            self.ui.actionFit_Hmm_Current,
            self.ui.actionFit_Hmm_All,
            self.ui.actionPredict_Selected_Traces,
            self.ui.actionPredict_All_traces,
            self.ui.actionClear_All_Predictions,
            self.ui.actionClear_and_Rerun,
            self.ui.actionSelect_All,
            self.ui.actionDeselect_All,
        )

        enableMenus_TransitionDensityWindow = (self.ui.actionFormat_Plot,)
        enableMenus_HistogramWindow = (self.ui.actionFormat_Plot,)

        instances = (
            MainWindow,
            TraceWindow,
            TransitionDensityWindow,
            HistogramWindow,
        )
        menus = (
            enableMenus_MainWindow,
            enableMenus_TraceWindow,
            enableMenus_TransitionDensityWindow,
            enableMenus_HistogramWindow,
        )

        # enable menus set above
        for instance, menulist in zip(instances, menus):
            if isinstance(self, instance):
                for menu in menulist:
                    menu.setEnabled(True)

        if isinstance(self, MainWindow):
            self.ui.actionClose.setEnabled(False)

    def disableOpenFileMenu(self):
        """
        Disables the Open menu to prevent loading additional videos if
        setting has been changed
        """
        self.ui: Ui_MenuBar

        if isinstance(self, MainWindow):
            self.ui.actionOpen.setEnabled(False)

    def bringToFront(self, window):
        """
        Focuses selected window and brings it to front.
        """

        def _focusWindow(window):
            window.show()
            window.setWindowState(
                window.windowState() & ~Qt.WindowMinimized | Qt.WindowActive
            )
            window.raise_()
            window.activateWindow()

        if window == "MainWindow":
            _focusWindow(MainWindow_)

        elif window == "TraceWindow":
            TraceWindow_.refreshPlot()
            _focusWindow(TraceWindow_)

        elif window == "HistogramWindow":
            HistogramWindow_.refreshPlot()
            _focusWindow(HistogramWindow_)

        elif window == "TransitionDensityWindow":
            hmm = [
                trace.hmm
                for trace in MainWindow_.data.traces.values()
                if trace.is_checked
            ]
            if len(hmm) > 0 and lib.misc.all_nonetype(hmm):
                TraceWindow_.fitCheckedTracesHiddenMarkovModel()
            TransitionDensityWindow_.refreshPlot()
            _focusWindow(TransitionDensityWindow_)

        else:
            raise ValueError("Window doesn't exist")

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
        matplotlib.rcParams["savefig.directory"] = self.currDir
        matplotlib.rcParams["savefig.facecolor"] = "white"
        matplotlib.rcParams["savefig.edgecolor"] = gvars.color_hud_black
        matplotlib.rcParams["savefig.transparent"] = True

    def setupSplitter(self, layout, width_left=300, width_right=1400):
        """
        Sets up a splitter between the listView and the plotWidget to make
        the list resizable. This has to be set up AFTER listView and
        plotWidget have been created, to place them in the layout
        """
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
        self.listModel = QStandardItemModel()

        # override old listview
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

    def returnCurrentListviewNames(self):
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

    def clearTraceAndRerun(self):
        """
        Clears everything and reruns on selected movies.
        """
        MainWindow_.clearTraces()

        # Load all traces into their respective movies, and generate a list
        # of traces
        MainWindow_.getTracesAllMovies()

        # Iterate over all filenames and add to list
        for (name, trace) in MainWindow_.data.traces.items():
            item = QStandardItem(trace.name)
            TraceWindow_.listModel.appendRow(item)
            TraceWindow_.currName = trace.name
            item.setCheckable(True)

        TraceWindow_.selectListViewTopRow()
        TraceWindow_.refreshPlot()
        TraceWindow_.show()

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

    def returnContainerInstance(self):
        """Returns appropriate data container for implemented windows"""
        if isinstance(self, MainWindow):
            container = MainWindow_.data.movies
        elif isinstance(self, TraceWindow):
            container = MainWindow_.data.traces
        else:
            raise NotImplementedError
        return container

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
        Deletes all Example_Files from listview.
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
        Unchecks all list elements.
        """
        for index in range(self.listModel.rowCount()):
            item = self.listModel.item(index)
            item.setCheckState(Qt.Unchecked)
            trace = self.getTrace(item)
            trace.is_checked = False

        if HistogramWindow_.isVisible():
            HistogramWindow_.refreshPlot()

        if TransitionDensityWindow_.isVisible():
            TransitionDensityWindow_.refreshPlot()

    def checkAll(self):
        """
        Checks all list elements.
        """
        for index in range(self.listModel.rowCount()):
            item = self.listModel.item(index)
            item.setCheckState(Qt.Checked)
            trace = self.getTrace(item)
            trace.is_checked = True

        if HistogramWindow_.isVisible():
            HistogramWindow_.refreshPlot()

        if TransitionDensityWindow_.isVisible():
            TransitionDensityWindow_.refreshPlot()

    def setupFigureCanvas(
        self, ax_setup, ax_window, use_layoutbox=True, **kwargs
    ):
        """
        Creates a canvas with a given ax layout.
        """
        self.plotWidget = PlotWidget(
            ax_setup=ax_setup, ax_window=ax_window, **kwargs
        )
        self.canvas = self.plotWidget.canvas

        if use_layoutbox:
            try:
                self.ui.mpl_LayoutBox.addWidget(self.plotWidget)
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

    def returnInfoHeader(self):
        """
        Generates boilerplate header text for every exported file.
        """
        exp_txt = "Exported by DeepFRET"
        date_txt = "Date: {}".format(time.strftime("%Y-%m-%d, %H:%M"))

        return exp_txt, date_txt

    def exportColocalization(self):
        """
        Exports colocalized ROI information for all currently colocalized
        movies. Does not automatically trigger ColocalizeAll beforehand.
        """
        exp_txt, date_txt = self.returnInfoHeader()

        directory = (
            self.getConfig(gvars.key_lastOpenedDir) + "/Colocalization.txt"
        )
        path, _ = QFileDialog.getSaveFileName(
            self, directory=directory
        )  # type: str, str

        if path != "":
            if not path.split("/")[-1].endswith(".txt"):
                path += ".txt"

            columns = ("green", "red", "green/red", "filename")
            d = {c: [] for c in columns}  # type: Dict[str, list]

            assert columns[-1] == "filename"

            for name, mov in MainWindow_.movies.items():

                data = (
                    mov.grn.n_spots,
                    mov.red.n_spots,
                    mov.coloc_grn_red.n_spots,
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
                        exp_txt, date_txt, df.to_csv(index=False, sep="\t")
                    )
                )

    def exportTraces(self, checked_only):
        """
        Exports all traces as ASCII Example_Files to selected directory.
        Maintains compatibility with iSMS and older pySMS scripts.
        """
        if checked_only:
            traces = [
                trace
                for trace in MainWindow_.data.traces.values()
                if trace.is_checked
            ]
        else:
            traces = [trace for trace in MainWindow_.data.traces.values()]

        exp_txt, date_txt = self.returnInfoHeader()
        diag = ExportDialog(
            init_dir=gvars.key_lastOpenedDir, accept_label="Export"
        )

        if diag.exec():
            path = diag.selectedFiles()[0]
            ctxt.app.processEvents()

            for trace in traces:
                if trace.y_pred is None:
                    df = pd.DataFrame(
                        {
                            "D-Dexc-bg": trace.grn.bg,
                            "A-Dexc-bg": trace.acc.bg,
                            "A-Aexc-bg": trace.red.bg,
                            "D-Dexc-rw": trace.grn.int,
                            "A-Dexc-rw": trace.acc.int,
                            "A-Aexc-rw": trace.red.int,
                            "S": trace.stoi,
                            "E": trace.fret,
                        }
                    ).round(4)
                else:
                    df = pd.DataFrame(
                        {
                            "D-Dexc-bg": trace.grn.bg,
                            "A-Dexc-bg": trace.acc.bg,
                            "A-Aexc-bg": trace.red.bg,
                            "D-Dexc-rw": trace.grn.int,
                            "A-Dexc-rw": trace.acc.int,
                            "A-Aexc-rw": trace.red.int,
                            "S": trace.stoi,
                            "E": trace.fret,
                            "p_blch": trace.y_pred[:, 0],
                            "p_aggr": trace.y_pred[:, 1],
                            "p_stat": trace.y_pred[:, 2],
                            "p_dyna": trace.y_pred[:, 3],
                            "p_nois": trace.y_pred[:, 4],
                            "p_scrm": trace.y_pred[:, 5],
                        }
                    ).round(4)

                mov_txt = "Movie filename: {}".format(trace.movie)
                id_txt = "FRET pair #{}".format(trace.n)
                bl_txt = (
                    "Donor bleaches at: {} - "
                    "Acceptor bleaches at: {}".format(
                        trace.grn.bleach, trace.red.bleach
                    )
                )

                savename = os.path.join(
                    path, TraceWindow_.generatePrettyTracename(trace)
                )

                with open(savename, "w") as f:
                    f.write(
                        "{0}\n"
                        "{1}\n"
                        "{2}\n"
                        "{3}\n"
                        "{4}\n\n"
                        "{5}".format(
                            exp_txt,
                            date_txt,
                            mov_txt,
                            id_txt,
                            bl_txt,
                            df.to_csv(index=False, sep="\t"),
                        )
                    )

    def exportCorrectionFactors(self):
        """
        Exports all available correction factors for each newTrace to table.
        """

        exp_txt, date_txt = self.returnInfoHeader()

        directory = (
            self.getConfig(gvars.key_lastOpenedDir) + "/CorrectionFactors.txt"
        )
        path, _ = QFileDialog.getSaveFileName(
            self, directory=directory
        )  # type: str, str

        if path != "":
            if not path.split("/")[-1].endswith(".txt"):
                path += ".txt"

            df = TraceWindow_.returnPooledCorrectionFactors()

            with open(path, "w") as f:
                f.write(
                    "{0}\n"
                    "{1}\n\n"
                    "{2}".format(
                        exp_txt, date_txt, df.to_csv(index=False, sep="\t")
                    )
                )

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

        HistogramWindow_.setPooledES()

        if path != "":
            if not path.split("/")[-1].endswith(".txt"):
                path += ".txt"

            # Exports all the currently plotted datapoints
            if HistogramWindow_.ui.applyCorrectionsCheckBox.isChecked():
                E, S = HistogramWindow_.E, HistogramWindow_.S
            else:
                E, S = HistogramWindow_.E_un, HistogramWindow_.S_un

            if E is not None:
                df = pd.DataFrame({"E": E, "S": S}).round(4)
            else:
                df = pd.DataFrame({"E": [], "S": []})

            with open(path, "w") as f:
                f.write(
                    "{0}\n"
                    "{1}\n\n"
                    "{2}".format(
                        exp_txt, date_txt, df.to_csv(index=False, sep="\t")
                    )
                )

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

        TransitionDensityWindow_.setPooledLifetimes()

        if path != "":
            if not path.split("/")[-1].endswith(".txt"):
                path += ".txt"

            with open(path, "w") as f:
                df = TransitionDensityWindow_.tdp_df

                f.write(
                    "{0}\n"
                    "{1}\n\n"
                    "{2}".format(
                        exp_txt, date_txt, df.to_csv(index=False, sep="\t")
                    )
                )

    def clearTraces(self):
        """Clears all currently obtained traces."""
        MainWindow_.data.traces.clear()
        TraceWindow_.listModel.clear()
        TraceWindow_.refreshPlot()

    def formatPlotInspector(self):
        """
        Opens the inspector (modal) window to format the current plot
        """
        if HistogramWindow_.isActiveWindow():
            HistogramInspector_.show()

        if TransitionDensityWindow_.isActiveWindow():
            TransitionDensityInspector_.show()

    def configInspector(self):
        """
        Opens the inspector (modal) window to do advanced actions
        """
        if TraceWindow_.isVisible():
            TraceWindowInspector_.show()

    def correctionFactorInspector(self):
        """
        Opens the inspector (modal) window to set correction factors
        """
        if TraceWindow_.isVisible():
            CorrectionFactorInspector_.show()

    def clearAllPredictions(self):
        """Override in subclass."""
        pass

    def colocalizeSpotsAllMovies(self):
        """Override in subclass."""
        pass

    def getTrace(self, name):
        """Override in subclass."""

    def getTracesAllMovies(self):
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

    def triggerBleach(self, color):
        """Override in subclass."""
        pass

    def fitSingleTraceHiddenMarkovModel(self, refresh):
        """Override in subclass."""
        pass

    def fitCheckedTracesHiddenMarkovModel(self):
        """Override in subclass."""
        pass

    def predictTraces(self, selected):
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


class MainWindow(BaseWindow):
    """
    Main UI for the application.
    """

    def __init__(self):
        super().__init__()

        # Initialize UI states
        self.currName = None
        self.currRow = None
        self.currDir = None
        self.currRoiSize = gvars.roi_draw_radius
        self.imgMode = self.getConfig(gvars.key_imgMode)
        self.batchLoaded = (
            False  # If movies have been batchloaded, disable some controls
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
        self.setupFigureCanvas(
            ax_setup=self.imgMode, ax_window="img", use_layoutbox=False
        )
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

        # Initialize DataHolder class
        self.data = MovieData()

        self.show()

    def currentMovie(self) -> ImageContainer:
        """
        Quick interface to obtain file data (or names will be extremely long).
        Add type hinting in metadata to improve autocomplete.
        """
        return self.data.get(self.currName)

    def newTrace(self, n) -> TraceContainer:
        """
        Shortcut to obtain newTrace, and create a new one if it doesn't exist.
        """
        tracename = self.currName + "_" + str(n)
        if tracename not in self.data.traces:
            self.data.traces[tracename] = TraceContainer(
                name=tracename, movie=self.currName, n=n
            )
        return self.data.traces[tracename]

    def trace_c(self, n, channel) -> TraceChannel:
        """
        Shortcut to obtain trace channel. See newTrace().
        """
        # if channel == "blue":
        #     return self.newTrace(n).blu
        if channel == "green":
            return self.newTrace(n).grn
        elif channel == "red":
            return self.newTrace(n).red
        elif channel == "acc":
            return self.newTrace(n).acc
        else:
            raise ValueError("Invalid channel")

    def openFile(self):
        """
        Open file to load in.
        """
        if not self.getConfig(gvars.key_batchLoadingMode):
            if self.getConfig(gvars.key_lastOpenedDir) == "None":
                directory = os.path.join(
                    os.path.join(os.path.expanduser("~")), "Desktop"
                )
            else:
                directory = self.getConfig(gvars.key_lastOpenedDir)

            filenames, selectedFilter = QFileDialog.getOpenFileNames(
                self,
                caption="Open File",
                filter="Tiff movie files (*.tif)",
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
                        array=self.data.movies.keys(),
                    )

                    self.data.load_img(
                        path=full_filename,
                        name=uniqueName,
                        setup=self.imgMode,
                        bg_correction=self.getConfig(gvars.key_illuCorrect),
                    )

                    item = QStandardItem(uniqueName)
                    item.setCheckable(False)

                    self.currName = uniqueName
                    self.currDir = os.path.dirname(full_filename)
                    self.listModel.appendRow(item)
                    # refresh listView
                    self.listView.repaint()
                    progressbar.increment()

                # Select first movie if loading into an empty listView
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
        """Loads one movie at a time and extracts traces, then clears the
        movie from memory afterwards"""
        # TraceWindow_.traces.clear()
        if self.getConfig(gvars.key_lastOpenedDir) == "None":
            directory = os.path.join(
                os.path.join(os.path.expanduser("~")), "Desktop"
            )
        else:
            directory = self.getConfig(gvars.key_lastOpenedDir)

        filenames, selectedFilter = QFileDialog.getOpenFileNames(
            self,
            caption="Open File",
            directory=directory,
            filter="Tiff movie files (*.tif)",
        )

        if len(filenames) > 0:
            ctxt.app.processEvents()
            progressbar = ProgressBar(
                loop_len=len(filenames), parent=MainWindow_
            )
            for i, full_filename in enumerate(filenames):
                if progressbar.wasCanceled():
                    break
                else:
                    progressbar.increment()

                self.currName = os.path.basename(full_filename)
                if self.currName in self.data.movies:
                    continue

                self.data.load_img(
                    path=full_filename,
                    name=os.path.basename(full_filename),
                    setup=self.imgMode,
                    bg_correction=self.getConfig(gvars.key_illuCorrect),
                )

                channels = ["blue", "green", "red"]
                if self.currentMovie().acc.exists:
                    # Can't have blue and FRET at the same time (at least not
                    # currently)
                    channels.remove("blue")

                    for c in channels:
                        self.colocalizeSpotsSingleMovie(
                            channel=c, find_npairs="auto"
                        )
                else:
                    for c in channels:
                        self.colocalizeSpotsSingleMovie(
                            channel=c, find_npairs="spinbox"
                        )

                self.getTracesSingleMovie()
                self.currentMovie().img = None
                for c in self.currentMovie().channels + (
                    self.currentMovie().acc,
                ):
                    c.raw = None

                self.listModel.appendRow(QStandardItem(self.currName))
                self.listView.repaint()

                self.currDir = os.path.dirname(full_filename)
                if self.currDir is not None:
                    self.setConfig(gvars.key_lastOpenedDir, self.currDir)

        if len(self.data.traces) > 0:
            currently_loaded = TraceWindow_.returnCurrentListviewNames()
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

    def colocalizeSpotsSingleMovie(self, channel, find_npairs="spinbox"):
        """
        Find and colocalize spots for a single currentMovie (not displayed).
        """
        mov = self.currentMovie()
        tolerance = gvars.roi_coloc_tolerances.get(
            self.getConfig(gvars.key_colocTolerance)
        )

        if channel == "green":
            channel = mov.grn
            spinBox = self.ui.spotsGrnSpinBox
            pairs = ((mov.grn, mov.red),)
            colocs = (mov.coloc_grn_red,)

        elif channel == "red":
            channel = mov.red
            spinBox = self.ui.spotsRedSpinBox
            pairs = ((mov.grn, mov.red),)
            colocs = (mov.coloc_grn_red,)
        else:
            raise ValueError("Invalid color")

        if find_npairs == "spinbox":
            find_npairs = spinBox.value()
        elif find_npairs == "auto":
            find_npairs = self.getConfig(gvars.key_autoDetectPairs)
        else:
            raise ValueError

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
                        yx=spot, indices=mov.indices, **gvars.cmask_p
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
            mov.grn.spots = mov.red.spots
            mov.acc.spots = mov.red.spots
            mov.grn.n_spots = mov.red.n_spots
            mov.acc.n_spots = mov.red.n_spots

        for (c1, c2), coloc in zip(pairs, colocs):
            if all((c1.n_spots, c2.n_spots)) > 0:
                coloc.spots = lib.imgdata.colocalize_rois(
                    c1.spots,
                    c2.spots,
                    color1=coloc.color1,
                    color2=coloc.color2,
                    tolerance=tolerance,
                )
                coloc.n_spots = len(coloc.spots)

        mov.coloc_all.spots = mov.coloc_grn_red.spots
        mov.coloc_all.n_spots = mov.coloc_grn_red.n_spots

    def displaySpotsSingle(self, channel):
        """
        Displays colocalized spot for a single movie.
        """
        self.getCurrentListObject()

        if self.currName is not None:
            self.colocalizeSpotsSingleMovie(channel)
            self.refreshPlot()

    def colocalizeSpotsAllMovies(self):
        """
        Colocalizes spots for all movies, with the same threshold. Use this
        method instead for progress bar.
        """
        progressbar = ProgressBar(
            loop_len=len(self.data.movies.keys()), parent=MainWindow_
        )
        for name in self.data.movies.keys():
            self.currName = name
            for c in "green", "red":
                self.colocalizeSpotsSingleMovie(c)
            progressbar.increment()

        self.resetCurrentName()
        self.refreshPlot()

    def getTracesSingleMovie(self):
        """
        Gets traces from colocalized ROIs, for a single movie.
        """
        if self.currName is None:
            return
        mov = self.currentMovie()

        # Clear all traces previously held traces whenever this is called
        mov.traces = {}

        if (
            mov.coloc_blu_grn.spots is not None
            and mov.coloc_grn_red.spots is not None
        ):
            mov.coloc_all.spots = lib.imgdata.colocalize_triple(
                mov.coloc_blu_grn.spots, mov.coloc_grn_red.spots
            )
        elif mov.acc.exists:
            mov.coloc_all.spots = mov.coloc_grn_red.spots
        else:
            mov.coloc_all.spots = mov.coloc_blu_red.spots

        if mov.coloc_all.spots is None:
            for c in "blue", "green", "red":
                self.colocalizeSpotsSingleMovie(c)
        else:
            for n, *row in mov.coloc_all.spots.itertuples():
                # if len(row) == 6:
                #     yx_blu, yx_grn, yx_red = lib.misc.pairwise(row)
                if mov.acc.exists:
                    yx_grn, yx_red = lib.misc.pairwise(row)
                    # yx_blu = None
                else:
                    yx_blu, yx_red = lib.misc.pairwise(row)
                    yx_grn = None

                trace = self.newTrace(n)

                # Green
                if mov.grn.exists and yx_grn is not None:
                    masks_grn = lib.imgdata.circle_mask(
                        yx=yx_grn, indices=mov.indices, **gvars.cmask_p
                    )
                    (
                        trace.grn.int,
                        trace.grn.bg,
                    ) = lib.imgdata.tiff_stack_intensity(
                        mov.grn.raw, *masks_grn, raw=True
                    )

                # Red
                masks_red = lib.imgdata.circle_mask(
                    yx=yx_red, indices=mov.indices, **gvars.cmask_p
                )
                trace.red.int, trace.red.bg = lib.imgdata.tiff_stack_intensity(
                    mov.red.raw, *masks_red, raw=True
                )

                # Acceptor (if FRET)
                if mov.acc.exists:
                    (
                        trace.acc.int,
                        trace.acc.bg,
                    ) = lib.imgdata.tiff_stack_intensity(
                        mov.acc.raw, *masks_red, raw=True
                    )
                    trace.fret = lib.math.calc_E(trace.get_intensities())
                    trace.stoi = lib.math.calc_S(trace.get_intensities())

                trace.frames = np.arange(1, len(trace.red.int) + 1)
                trace.frames_max = max(trace.frames)

    def getTracesAllMovies(self):
        """
        Gets the traces from all videos that have colocalized ROIs.
        """
        for name in self.data.movies.keys():
            self.currName = name
            for c in "green", "red":  # blue
                self.colocalizeSpotsSingleMovie(c)

            self.getTracesSingleMovie()

        self.resetCurrentName()
        # TraceWindow_.traces = TraceWindow_.returnTracenamesAllMovies()

    def savePlot(self):
        """
        Saves plot with colors suitable for export (e.g. white background).
        """
        self.setSavefigrcParams()

        if self.currName is not None:
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
            mov = self.currentMovie()
            roi_radius = mov.roi_radius

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
            sensitivity = 100 if self.imgMode == "bypass" else 250

            for c, lo, hi in zip(
                mov.channels, contrast_lo, contrast_hi
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
            for img, ax in zip(mov.channels, self.canvas.axes_single):
                if img.rgba is not None:
                    # Avoid imshow showing blank if clipped too much
                    if np.isnan(img.rgba).any():
                        img.rgba.fill(1)
                    ax.imshow(img.rgba, cmap=img.cmap, vmin=0)
                else:
                    lib.plotting.empty_imshow(ax)

            # Blended channels
            if len(self.canvas.axes_blend) == 3:
                pairs = (
                    (mov.blu, mov.grn),
                    (mov.blu, mov.red),
                    (mov.grn, mov.red),
                )
            else:
                pairs = ((mov.grn, mov.red),)

            for pair, ax in zip(pairs, self.canvas.axes_blend):
                c1, c2 = pair
                if c1.rgba is not None and c2.rgba is not None:
                    ax.imshow(
                        lib.imgdata.light_blend(
                            c1.rgba, c2.rgba, cmap1=c1.cmap, cmap2=c2.cmap
                        ),
                        vmin=0,
                    )
                else:
                    lib.plotting.empty_imshow(ax)

            for ax in self.canvas.axes_all:
                ax.set_xticks(())
                ax.set_yticks(())

            # Green spots
            if mov.grn.n_spots > 0:
                lib.plotting.plot_rois(
                    mov.grn.spots,
                    self.canvas.ax_grn,
                    color=gvars.color_white,
                    radius=roi_radius,
                )

            # Red spots
            if mov.red.n_spots > 0:
                lib.plotting.plot_rois(
                    mov.red.spots,
                    self.canvas.ax_red,
                    color=gvars.color_white,
                    radius=roi_radius,
                )

            # Colocalized spots
            if mov.coloc_grn_red.spots is not None:
                lib.plotting.plot_roi_coloc(
                    mov.coloc_grn_red.spots,
                    img_ax=self.canvas.ax_grn_red,
                    color1=gvars.color_green,
                    color2=gvars.color_red,
                    radius=roi_radius,
                )

            if (
                self.imgMode == "3-color"
                and mov.blu.exists
                or self.imgMode == "bypass"
            ):
                if mov.coloc_blu_grn.spots is not None:
                    lib.plotting.plot_roi_coloc(
                        mov.coloc_blu_grn.spots,
                        img_ax=self.canvas.ax_blu_grn,
                        color1=gvars.color_blue,
                        color2=gvars.color_green,
                        radius=roi_radius,
                    )

                if mov.coloc_blu_red.spots is not None:
                    lib.plotting.plot_roi_coloc(
                        mov.coloc_blu_red.spots,
                        img_ax=self.canvas.ax_blu_red,
                        color1=gvars.color_blue,
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
            mov = self.currentMovie()

            # if mov.coloc_frac is not None:
            #     self.ui.labelColocFractionVal.setText(
            #         "{:.1f}".format(mov.coloc_frac)
            #     )
            # else:
            #     self.ui.labelColocFractionVal.setText(str("-"))

            channels = (
                # mov.coloc_blu_grn,
                # mov.coloc_blu_red,
                mov.coloc_grn_red,
                # mov.coloc_all,
                # mov.blu,
                mov.grn,
                mov.red,
            )

            for channel, label in zip(channels, self.labels):
                label.setText(str(channel.n_spots))

            # self.ui.spotsBluSpinBox.setDisabled(not mov.blu.exists)
            self.ui.spotsGrnSpinBox.setDisabled(not mov.grn.exists)
            self.ui.spotsRedSpinBox.setDisabled(not mov.red.exists)

            if self.batchLoaded:
                self.disableSpinBoxes(
                    (
                        # "blue",
                        "green",
                        "red",
                    )
                )

        else:
            for label in self.labels:
                label.setText(str("-"))
            # self.ui.labelColocFractionVal.setText(str("-"))

        # Repaint all labels
        for label in self.labels:
            label.repaint()

        # self.ui.labelColocFractionVal.repaint()

    def findTracesAndShow(self):
        """
        Gets the name of currently stored traces and puts them into the
        trace listView.
        """
        if len(self.data.traces) == 0:
            # Load all traces into their respective movies, and generate a
            # list of traces
            self.getTracesAllMovies()
            currently_loaded = self.returnCurrentListviewNames()
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
        Creates an empty movie object to load traces into by transplanting a
        list of loaded TraceContainers
        """
        tracename = self.currName + "_" + str(n)

        trace.currentMovie = self.currName
        trace.name = tracename
        trace.n = n

        if tracename not in self.data.movies[self.currName].traces:
            self.data.movies[self.currName].traces[tracename] = trace

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
        print(self.currName)
        print(self.data.movies.keys())


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

        # Initialize interface
        # Instantiate but do not show
        self.ui = Ui_TraceWindow()
        self.ui.setupUi(self)
        self.setupListView(use_layoutbox=False)
        self.setupFigureCanvas(
            ax_setup=self.getConfig(gvars.key_imgMode),
            ax_window="trace",
            use_layoutbox=False,
        )
        self.setupPlot()
        self.setupSplitter(layout=self.ui.layoutBox)

        self.data = MainWindow_.data

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
            HistogramWindow_.setPooledES()
            HistogramWindow_.gauss_params = None
            if HistogramWindow_.isVisible():
                HistogramWindow_.refreshPlot()

            TransitionDensityWindow_.setPooledLifetimes()
            TransitionDensityWindow_.setClusteredTransitions()

            if TransitionDensityWindow_.isVisible():
                TransitionDensityWindow_.refreshPlot()

    def openFile(self, *args):
        """
        Loads ASCII files directly into the TraceWindow.
        """
        if self.getConfig(gvars.key_lastOpenedDir) == "None":
            directory = os.path.join(
                os.path.join(os.path.expanduser("~")), "Desktop"
            )
        else:
            directory = self.getConfig(gvars.key_lastOpenedDir)

        filenames, selectedFilter = QFileDialog.getOpenFileNames(
            self,
            caption="Open File",
            filter="Trace ASCII files (*.txt)",
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
                newTrace = self.loadTraceFromAscii(full_filename)

                if (n % update_every_n) == 0:
                    progressbar.increment()

                # If file wasn't loaded properly, skip
                if newTrace is None:
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

    def loadTraceFromAscii(self, full_filename) -> [TraceContainer, None]:
        """
        Reads a trace from an ASCII text file. Several checks are included to
        include flexible compatibility with different versions of trace exports.
        Also includes support for all iSMS traces.
        """
        colnames = [
            "D-Dexc-bg.",
            "A-Dexc-bg.",
            "A-Aexc-bg.",
            "D-Dexc-rw.",
            "A-Dexc-rw.",
            "A-Aexc-rw.",
            "S",
            "E",
        ]

        with open(full_filename) as f:
            txt_header = [next(f) for _ in range(5)]

        # This is for iSMS compatibility
        if txt_header[0].split("\n")[0] == "Exported by iSMS":
            df = pd.read_csv(full_filename, skiprows=5, sep="\t", header=None)
            if len(df.columns) == colnames:
                df.columns = colnames
            else:
                try:
                    df.columns = colnames
                except ValueError:
                    colnames = colnames[3:]
                    df.columns = colnames
        # Else DeepFRET trace compatibility
        else:
            df = lib.misc.csv_skip_to(
                path=full_filename, line="D-Dexc", sep="\s+"
            )
        try:
            pair_n = lib.misc.seek_line(
                path=full_filename, line_starts="FRET pair"
            )
            pair_n = int(pair_n.split("#")[-1])

            movie = lib.misc.seek_line(
                path=full_filename, line_starts="Movie filename"
            )
            movie = movie.split(": ")[-1]

            bleaching = lib.misc.seek_line(
                path=full_filename,
                line_starts=("Donor bleaches at", "Bleaches at"),
            )

        except AttributeError:
            return None

        trace = TraceContainer(
            movie=movie, n=pair_n, name=os.path.basename(full_filename)
        )

        if "D-Dexc_F" in df.columns:
            warnings.warn(
                "This trace is created with an older format.",
                DeprecationWarning,
            )
            trace.grn.int = df["D-Dexc_F"].values
            trace.acc.int = df["A-Dexc_I"].values
            trace.red.int = df["A-Aexc_I"].values

            zeros = np.zeros(len(trace.grn.int))
            trace.grn.bg = zeros
            trace.acc.bg = zeros
            trace.red.bg = zeros

        else:
            if "p_blch" in df.columns:
                ml_cols = [
                    "p_blch",
                    "p_aggr",
                    "p_stat",
                    "p_dyna",
                    "p_nois",
                    "p_scrm",
                ]
                colnames += ml_cols
                trace.y_pred = df[ml_cols].values
                trace.y_class, trace.confidence = lib.math.seq_probabilities(
                    trace.y_pred
                )

            # This strips periods if present
            df.columns = [c.strip(".") for c in colnames]

            trace.grn.int = df["D-Dexc-rw"].values
            trace.acc.int = df["A-Dexc-rw"].values
            trace.red.int = df["A-Aexc-rw"].values

            try:
                trace.grn.bg = df["D-Dexc-bg"].values
                trace.acc.bg = df["A-Dexc-bg"].values
                trace.red.bg = df["A-Aexc-bg"].values
            except KeyError:
                zeros = np.zeros(len(trace.grn.int))
                trace.grn.bg = zeros
                trace.acc.bg = zeros
                trace.red.bg = zeros

        trace.fret = lib.math.calc_E(trace.get_intensities())
        trace.stoi = lib.math.calc_S(trace.get_intensities())

        trace.frames = np.arange(1, len(trace.grn.int) + 1, 1)
        trace.frames_max = trace.frames.max()

        # TODO: revert mechanism to first bleaching or separate D/A bleaching
        #  for consistency throughout code (and speedups)
        try:
            bleaching = re.findall(r"\d+", str(bleaching))
            if any(bleaching):  # check if list is not empty
                trace.grn.bleach, trace.red.bleach = (
                    int(bleaching[0]),
                    int(bleaching[-1]),
                )  # works for both 1 or 2 values by indexing both ways
                trace.first_bleach = lib.misc.min_none(
                    (trace.grn.bleach, trace.red.bleach)
                )
        except (ValueError, AttributeError):
            pass

        return trace

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

    def fitSingleTraceHiddenMarkovModel(self, refresh=True):
        """
        Fits the selected trace with a Hidden Markov Model (HMM). Any trace with
        only a single transition will count towards the number of samples
        included, but will not show up in the transition density plot,
        as it's not possible to associate a lifetime of the transition
        without observing the endpoints.
        """
        alpha = self.getConfig(gvars.key_alphaFactor)
        delta = self.getConfig(gvars.key_deltaFactor)

        if self.currName is not None and len(self.data.traces) > 0:
            trace = self.currentTrace()
            F_DA, I_DD, I_DA, I_AA = lib.math.correct_DA(
                trace.get_intensities(), alpha=alpha, delta=delta
            )
            fret = lib.math.calc_E(trace.get_intensities(), alpha, delta)
            X = np.column_stack((I_DD, F_DA))
            X = sklearn.preprocessing.robust_scale(X)

            try:
                idealized, time, transitions = lib.math.fit_hmm(
                    X=X[: trace.first_bleach], y=fret[: trace.first_bleach]
                )
                trace.hmm = idealized
                trace.hmm_idx = time
                trace.transitions = transitions
            except ValueError:
                warnings.warn("Error in HMM. Trace skipped.", RuntimeWarning)

        # Only refresh immediately for single fits
        if refresh:
            self.refreshPlot()
            if TransitionDensityWindow_.isVisible():
                TransitionDensityWindow_.refreshPlot()

    def fitCheckedTracesHiddenMarkovModel(self):
        """
        Fits all selected traces with a Hidden Markov Model (HMM)
        """
        ctxt.app.processEvents()
        traces = [
            trace for trace in self.data.traces.values() if trace.is_checked
        ]

        progressbar = ProgressBar(loop_len=len(traces), parent=TraceWindow_)
        for trace in traces:  # type: TraceContainer
            if progressbar.wasCanceled():
                break
            self.currName = trace.name
            self.fitSingleTraceHiddenMarkovModel(refresh=False)
            progressbar.increment()
        self.resetCurrentName()
        self.refreshPlot()

        if TransitionDensityWindow_.isVisible():
            TransitionDensityWindow_.refreshPlot()

    def setPredictions(self, trace, yi_pred):
        """Assign trace predictions to correspondign trace"""
        trace.y_pred = yi_pred
        trace.y_class, trace.confidence = lib.math.seq_probabilities(
            trace.y_pred
        )
        trace.first_bleach = lib.math.find_bleach(
            p_bleach=trace.y_pred[:, 0], threshold=0.5, window=7
        )
        for c in trace.channels:
            c.bleach = trace.first_bleach

    def predictTraces(self, single=False, checked_only=False):
        """
        Classifies checked traces with deep learning model.
        """
        ctxt.app.processEvents()
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
            progressbar = ProgressBar(loop_len=batches, parent=TraceWindow_)
            all_eq = (
                False
                if single
                else lib.math.all_equal([trace.frames_max for trace in traces])
            )

            model = ctxt.keras_model
            if all_eq:
                # shape is (n_traces) if traces have uneven length
                X = np.array(
                    [
                        lib.math.correct_DA(
                            trace.get_intensities(), alpha=alpha, delta=delta
                        )
                        for trace in traces
                    ]
                )
                X = np.swapaxes(X, 1, 2)
                X = lib.math.sample_max_normalize_3d(X[:, :, 1:])

                # Fix single sample dimension
                if len(X.shape) == 2:
                    X = X[np.newaxis, :, :]

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
                    xi = lib.math.sample_max_normalize_3d(X=xi[:, 1:])
                    yi = lib.math.predict_single(xi=xi, model=model)
                    Y.append(yi)
                    if n % batch_size == 0:
                        progressbar.increment()

            for n, trace in enumerate(traces):
                self.setPredictions(trace=trace, yi_pred=Y[n])

            self.resetCurrentName()
            self.refreshPlot()

    def clearAllPredictions(self):
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
        self.refreshPlot()

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

    def returnTracenamesAllMovies(self, names_only=True):
        """
        Obtains all available tracenames from movies and returns them as a list.
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

                cond6 = False if trace.first_bleach is None else True
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

    @staticmethod
    def generatePrettyTracename(trace: TraceContainer) -> str:
        """
        Generates a pretty, readable generatePrettyTracename.
        """

        if trace.movie is None:
            name = "Trace_pair{}.txt".format(trace.n)
        else:
            name = "Trace_{}_pair{}.txt".format(
                trace.movie.replace(".", "_"), trace.n
            )

        # Scrub mysterious \n if they appear due to filenames
        name = "".join(name.splitlines(keepends=False))
        return name

    def savePlot(self):
        """
        Saves plot with colors suitable for export (e.g. white background).
        """
        self.setSavefigrcParams()

        # Change colors of output to look better for export
        if self.currName is not None:
            self.canvas.defaultImageName = self.generatePrettyTracename(
                self.currentTrace()
            )
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

        if self.getConfig(gvars.key_imgMode) != "bypass":
            # only for FRET red axis (right side)
            self.canvas.ax_red.yaxis.set_label_coords(1.05, 0.5)

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
        if self.currName is not None and len(self.data.traces) > 0:
            trace = self.currentTrace()
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
                ax.set_xlim(1, trace.frames_max)

            # Canvas setup
            if self.canvas.ax_setup in (
                "dual",
                "2-color",
                "2-color-inv",
                "3-color",
            ):
                channels = [trace.grn, trace.acc, trace.red]
                colors = [gvars.color_green, gvars.color_red, gvars.color_red]

            elif self.canvas.ax_setup == "bypass":
                channels = trace.grn, trace.red
                colors = gvars.color_green, gvars.color_red
            else:
                raise ValueError("Setup is not valid. Corrupted config.ini?")

            for (ax, label), channel, color in zip(
                self.canvas.axes_c, channels, colors
            ):
                if label == "A":
                    int_ = F_DA
                    ax = self.canvas.ax_red
                elif label == "A-direct":
                    int_ = I_AA
                    ax = self.canvas.ax_alx
                elif label == "D":
                    int_ = I_DD
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
                    int_ = channel.int
                if int_ is None:
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

                ax.plot(trace.frames, int_, color=color)
                ax.set_ylim(0 - int_.max() * 0.1, int_.max() * 1.1)
                ax.yaxis.label.set_color(gvars.color_gui_text)
                lib.plotting.set_axis_exp_ylabel(
                    ax=ax, label=label, values=int_
                )

            # Continue drawing FRET specifics
            if self.canvas.ax_setup != "bypass":
                fret = lib.math.calc_E(trace.get_intensities(), *factors)
                stoi = lib.math.calc_S(trace.get_intensities(), *factors)

                ax_E = self.canvas.ax_fret
                ax_S = self.canvas.ax_stoi

                for signal, ax, color, label in zip(
                    (fret, stoi),
                    (ax_E, ax_S),
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
                    ax.set_ylabel(label)

                ax_E.set_ylim(-0.1, 1.1)
                if trace.hmm is not None:
                    ax_E.plot(
                        trace.hmm_idx,
                        trace.hmm,
                        color=gvars.color_blue,
                        zorder=3,
                    )

                ax_S.set_ylim(0, 1)
                ax_S.set_yticks([0.5])
                ax_S.axhline(
                    0.5, color="black", alpha=0.3, lw=0.5, ls="--", zorder=2
                )

                # If clicking on the newTrace
                if len(trace.xdata) == 1:
                    self.canvas.ax_grn.axvline(
                        trace.xdata[0], ls="-", alpha=0.2, zorder=10
                    )
                    self.canvas.ax_alx.axvline(
                        trace.xdata[0], ls="-", alpha=0.2, zorder=10
                    )
                elif len(trace.xdata) == 2:
                    xmin, xmax = sorted(trace.xdata)
                    self.canvas.ax_grn.axvspan(xmin, xmax, alpha=0.2, zorder=10)
                    self.canvas.ax_alx.axvspan(xmin, xmax, alpha=0.2, zorder=10)

            if hasattr(self.canvas, "ax_ml") and trace.y_pred is not None:
                lib.plotting.plot_predictions(
                    trace.y_pred, ax=self.canvas.ax_ml
                )

        else:
            for ax in self.canvas.axes:
                ax.clear()
                ax.tick_params(axis="both", colors=gvars.color_gui_bg)

        self.canvas.draw()

    def _debug(self):
        pass


class HistogramWindow(BaseWindow):
    def __init__(self):
        super().__init__()
        self.currDir = None
        self.E = None
        self.E_un = None
        self.S = None
        self.S_un = None
        self.trace_median_len = None
        self.marg_bins = np.arange(-0.3, 1.3, 0.02)
        self.xpts = np.linspace(-0.3, 1.3, 300)
        self.gauss_params = None
        self.bics = None

        self.alpha = None
        self.delta = None
        self.beta = None
        self.gamma = None

        self.ui = Ui_HistogramWindow()
        self.ui.setupUi(self)

        self.setupFigureCanvas(
            ax_setup="plot", ax_window="jointgrid", width=2, height=2
        )
        self.setupPlot()

        self.data = MainWindow_.data

        [
            self.ui.gaussianAutoButton.clicked.connect(x)
            for x in (
                partial(self.fitGaussians, "auto"),
                partial(self.refreshPlot, True),
            )
        ]
        [
            self.ui.gaussianSpinBox.valueChanged.connect(x)
            for x in (self.fitGaussians, self.refreshPlot)
        ]
        [
            self.ui.applyCorrectionsCheckBox.clicked.connect(x)
            for x in (self.fitGaussians, self.refreshPlot)
        ]
        [
            self.ui.framesSpinBox.valueChanged.connect(x)
            for x in (self.fitGaussians, self.refreshPlot)
        ]

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

        for ax in self.canvas.axes_marg:
            ax.tick_params(
                axis="both",
                which="both",
                bottom=False,
                top=False,
                left=False,
                right=False,
                labelbottom=False,
                labeltop=False,
                labelleft=False,
                labelright=False,
            )

    def setPooledES(self, n_first_frames="spinbox"):
        """
        Returns pooled E and S_app data before bleaching, for each trace. The
        loops take approx. 0.1 ms per trace, and it's too much trouble to
        lower it further.
        """
        if n_first_frames == "all":
            n_first_frames = None
        elif n_first_frames == "spinbox":
            n_first_frames = self.ui.framesSpinBox.value()
        else:
            raise ValueError("n_first_frames must be either 'all' or 'spinbox'")

        self.E, self.S, self.E_un, self.S_un = None, None, None, None

        checkedTraces = [
            trace for trace in self.data.traces.values() if trace.is_checked
        ]

        self.n_samples = len(checkedTraces)
        alpha = self.getConfig(gvars.key_alphaFactor)
        delta = self.getConfig(gvars.key_deltaFactor)

        self.trace_median_len = np.median(
            [
                trace.first_bleach
                if trace.first_bleach is not None
                else trace.frames_max
                for trace in checkedTraces
            ]
        )

        E_app, S_app = [], []
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
        self.E_un, self.S_un = lib.math.trim_ES(E_app, S_app)

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

        self.alpha = alpha
        self.delta = delta

    def fitGaussians(self, states):
        """
        Fits multiple gaussians to the E data
        """
        corrected = self.ui.applyCorrectionsCheckBox.isChecked()
        E = self.E if corrected else self.E_un

        if E is not None:
            k_states = (
                range(1, 6)
                if states == "auto"
                else self.ui.gaussianSpinBox.value()
            )
            try:
                fitdict = lib.math.fit_gaussian_mixture(
                    arr=E, k_states=k_states
                )
                self.gauss_params = fitdict["params"]
                self.bics = fitdict["bics"]
                self.best_k = fitdict["best_k"]
            except TypeError:
                pass

            if self.best_k is not None:
                self.ui.gaussianSpinBox.setValue(self.best_k)
                self.ui.gaussianSpinBox.repaint()

    def plotDefaultElements(self):
        """
        Re-plot non-persistent plot settings (otherwise will be overwritten
        by ax.clear())
        """
        self.canvas.ax_ctr.set_xlim(-0.1, 1.1)
        self.canvas.ax_ctr.set_ylim(-0.1, 1.1)
        self.canvas.ax_ctr.set_xlabel(xlabel="E", color=gvars.color_gui_text)
        self.canvas.ax_ctr.set_ylabel(ylabel="S", color=gvars.color_gui_text)

    def plotTop(self, corrected):
        """
        Plots the top histogram (E).
        """
        E = self.E if corrected else self.E_un

        if E is not None:
            self.canvas.ax_top.clear()
            self.canvas.ax_top.hist(
                E,
                bins=self.marg_bins,
                color=gvars.color_orange,
                alpha=0.8,
                density=True,
                histtype="stepfilled",
            )

        if self.gauss_params is not None:
            sum_ = []
            xpts = self.xpts
            for i, gauss_params in enumerate(self.gauss_params):
                m, s, w = gauss_params
                self.canvas.ax_top.plot(
                    xpts, w * scipy.stats.norm.pdf(xpts, m, s)
                )
                sum_.append(np.array(w * scipy.stats.norm.pdf(xpts, m, s)))
            joint = np.sum(sum_, axis=0)
            self.canvas.ax_top.plot(
                xpts, joint, color=gvars.color_grey, alpha=1, zorder=10
            )
        self.canvas.ax_top.set_xlim(-0.1, 1.1)

    def plotRight(self, corrected):
        """
        Plots the right histogram (S).
        """
        S = self.S if corrected else self.S_un

        self.canvas.ax_rgt.clear()
        self.canvas.ax_rgt.hist(
            S,
            bins=self.marg_bins,
            color=gvars.color_purple,
            alpha=0.8,
            density=True,
            orientation="horizontal",
            histtype="stepfilled",
        )
        self.canvas.ax_rgt.set_ylim(-0.1, 1.1)

    def plotCenter(self, corrected):
        """
        Plots the center without histograms.
        """
        S = self.S if corrected else self.S_un
        E = self.E if corrected else self.E_un

        params = self.inspector.returnInspectorValues()
        bandwidth, resolution, n_colors, overlay_pts, pts_alpha = params
        self.inspector.setInspectorConfigs(params)

        self.canvas.ax_ctr.clear()

        c = lib.math.contour_2d(
            xdata=E,
            ydata=S,
            bandwidth=bandwidth / 200,
            resolution=resolution,
            kernel="linear",
            n_colors=n_colors,
        )
        self.canvas.ax_ctr.contourf(*c, cmap="plasma")

        if overlay_pts:
            self.canvas.ax_ctr.scatter(
                E, S, s=20, color="black", zorder=1, alpha=pts_alpha / 20
            )  # Conversion factor, because sliders can't do [0,1]
        self.canvas.ax_ctr.axhline(
            0.5, color="black", alpha=0.3, lw=0.5, ls="--", zorder=2
        )

        n_equals_txt = "N = {}\n".format(self.n_samples)
        if not np.isnan(self.trace_median_len):
            n_equals_txt += "(median length {:.0f})".format(
                self.trace_median_len
            )

        self.canvas.ax_ctr.text(
            x=0, y=0.9, s=n_equals_txt, color=gvars.color_gui_text
        )

        if self.gauss_params is not None:
            for n, gauss_params in enumerate(self.gauss_params):
                m, s, w = gauss_params
                self.canvas.ax_ctr.text(
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
            self.canvas.ax_ctr.text(
                x=0.0,
                y=0.15 - 0.05 * n,
                s=r"$\{}$ = {:.2f}".format(name, factor),
                color=gvars.color_gui_text,
                zorder=10,
            )

    def refreshPlot(self, autofit=False):

        """
        Refreshes plot with currently selected traces. Plot to refresh can be
        top, right (histograms) or center (scatterplot), or all.
        """
        corrected = self.ui.applyCorrectionsCheckBox.isChecked()
        try:
            self.setPooledES()
            if self.E is not None:
                self.plotTop(corrected)
                self.plotRight(corrected)
                self.plotCenter(corrected)
            else:
                for ax in self.canvas.axes:
                    ax.clear()
                self.plotDefaultElements()
        except (AttributeError, ValueError):
            pass

        self.plotDefaultElements()
        self.canvas.draw()

    def _debug(self):
        pass


class TransitionDensityWindow(BaseWindow):
    """
    Window for displaying all transitions available as fitted by the the
    Hidden Markov Model.
    """

    def __init__(self):
        super().__init__()
        self.currDir = None
        self.fret_before = None
        self.fret_after = None
        self.fret_lifetime = None
        self.n_samples = None
        self.selected_data = None
        self.tdp_df = None
        self.colors = (
            "#66c2a5",
            "#fc8d62",
            "#8da0cb",
            "#e78ac3",
            "#a6d854",
            "#ffd92f",
            "#e5c494",
            "#b3b3b3",
        )
        self.data = MainWindow_.data

        # dynamically created once plot is refreshed
        self.tdp_ax = None
        self.hist_axes = None

        self.ui = Ui_TransitionDensityWindow()
        self.ui.setupUi(self)
        self.ui.nClustersSpinBox.valueChanged.connect(self.refreshPlot)

        self.setupFigureCanvas(
            ax_setup="plot", ax_window="dynamic", width=2, height=2
        )
        self.setupPlot()

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
        self.tdp_ax.set_xlim(-0.15, 1.15)
        self.tdp_ax.set_ylim(-0.15, 1.15)
        self.tdp_ax.set_xlabel(xlabel="E", color=gvars.color_gui_text)
        self.tdp_ax.set_ylabel(ylabel="E + 1", color=gvars.color_gui_text)
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

    def setPooledLifetimes(self):
        """
        Return pooled lifetimes from Hidden Markov Model fits.
        """
        checkedTraces = [
            trace for trace in self.data.traces.values() if trace.is_checked
        ]
        self.n_samples = len(checkedTraces)

        try:
            transitions = pd.concat(
                [trace.transitions for trace in checkedTraces]
            )
            transitions.reset_index(inplace=True)

            self.fret_lifetime = transitions["lifetime"]
            self.fret_before = transitions["y_before"]
            self.fret_after = transitions["y_after"]

        except ValueError:
            self.fret_lifetime = None
            self.fret_before = None
            self.fret_after = None
            self.tdp_df = None

    def setClusteredTransitions(self):
        if self.fret_before is not None:
            n_clusters = self.ui.nClustersSpinBox.value()

            tdp_df = pd.DataFrame(
                {
                    "E_bf": self.fret_before,
                    "E_af": self.fret_after,
                    "lifetime": self.fret_lifetime,
                }
            )

            tdp_df.dropna(inplace=True)

            up_diag = tdp_df[tdp_df["E_bf"] < tdp_df["E_af"]]  # 0
            lw_diag = tdp_df[tdp_df["E_bf"] > tdp_df["E_af"]]  # 1

            halves = up_diag, lw_diag
            for n, half in enumerate(halves):
                m = sklearn.cluster.KMeans(n_clusters=n_clusters)
                m.fit(half[["E_bf", "E_af"]])
                half["label"] = m.labels_ + n_clusters * n

            diags = pd.concat(halves)
            tdp_df["label"] = diags["label"]

            self.tdp_df = tdp_df

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

        self.tdp_ax.plot([-1, 2], [-1, 2], color="lightgrey", ls="--")
        if self.fret_before is not None and len(self.fret_before) > 0:
            cont = lib.math.contour_2d(
                xdata=self.fret_before,
                ydata=self.fret_after,
                bandwidth=bandwidth / 200,
                resolution=resolution,
                kernel="linear",
                n_colors=n_colors,
            )
            self.tdp_ax.contourf(*cont, cmap="viridis")
            self.tdp_ax.text(
                x=0,
                y=0.9,
                s="N = {}\n"
                "{} transitions\n".format(
                    self.n_samples, len(self.fret_lifetime)
                ),
                color=gvars.color_gui_text,
            )

            for i, cluster in self.tdp_df.groupby("label"):
                xi = self.tdp_df["E_bf"][self.tdp_df["label"] == i]
                yi = self.tdp_df["E_af"][self.tdp_df["label"] == i]

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

        def estimate_binwidth(x):
            """Estimate optimal binwidth by the Freedman-Diaconis rule."""
            return 2 * scipy.stats.iqr(x) / np.size(x) ** (1 / 3)

        def exp_(x, N, l):
            e = scipy.special.expit
            return N * (l * e(-l * x))

        if self.tdp_df is not None:
            fret_lifetimes = self.tdp_df["lifetime"]
            max_lifetime = np.max(fret_lifetimes)
            bw = estimate_binwidth(fret_lifetimes)
            bins = np.arange(0, max_lifetime, bw)

            for k, cluster in self.tdp_df.groupby("label"):
                try:
                    hx, hy, *_ = lib.math.histpoints_w_err(
                        data=cluster["lifetime"],
                        bins=bins,
                        density=False,
                        least_count=1,
                    )
                    popt, pcov = scipy.optimize.curve_fit(
                        exp_, xdata=hx, ydata=hy
                    )
                    perr = np.sqrt(np.diag(pcov))

                    rate = popt[1]
                    rate_err = perr[1]

                    if rate_err > 3 * rate:
                        rate_err = np.inf

                    self.hist_axes[k].plot(
                        bins,
                        exp_(bins, *popt),
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
            HistogramWindow_.inspector = self
        elif isinstance(parent, TransitionDensityWindow):
            self.keys = gvars.keys_tdp
            TransitionDensityWindow_.inspector = self
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
                if isinstance(parent, HistogramWindow):
                    slider.valueChanged.connect(HistogramWindow_.refreshPlot)
                elif isinstance(parent, TransitionDensityWindow):
                    slider.valueChanged.connect(
                        TransitionDensityWindow_.refreshPlot
                    )
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

        TraceWindow_.inspector = self

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
        self.keras_model = None
        self.config = None

    def load_resources(self):
        """
        Loads initial resources from disk to application
        """
        # model_experimental is better but undocumented
        self.keras_model = load_model(
            self.get_resource("model_published.h5")
        )
        self.config = ConfigObj(self.get_resource("config.ini"))

    def run(self):
        """
        Returns main loop exit code to be put into sys.exit()
        """
        return self.app.exec_()


if __name__ == "__main__":
    # Fixes https://github.com/mherrmann/fbs/issues/87
    multiprocessing.freeze_support()

    # Load app
    ctxt = AppContext()
    ctxt.load_resources()

    # Windows
    MainWindow_ = MainWindow()
    TraceWindow_ = TraceWindow()
    HistogramWindow_ = HistogramWindow()
    TransitionDensityWindow_ = TransitionDensityWindow()

    # Inspector sheets
    HistogramInspector_ = DensityWindowInspector(HistogramWindow_)
    TransitionDensityInspector_ = DensityWindowInspector(
        TransitionDensityWindow_
    )
    CorrectionFactorInspector_ = CorrectionFactorInspector(TraceWindow_)
    TraceWindowInspector_ = TraceWindowInspector(TraceWindow_)

    exit_code = ctxt.run()
    sys.exit(exit_code)
