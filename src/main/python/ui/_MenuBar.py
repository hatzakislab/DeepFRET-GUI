# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/MenuBar.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MenuBar(object):
    def setupUi(self, MenuBar):
        MenuBar.setObjectName("MenuBar")
        MenuBar.resize(500, 500)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MenuBar.sizePolicy().hasHeightForWidth())
        MenuBar.setSizePolicy(sizePolicy)
        MenuBar.setMinimumSize(QtCore.QSize(500, 500))
        MenuBar.setMaximumSize(QtCore.QSize(5000, 5000))
        MenuBar.setFocusPolicy(QtCore.Qt.ClickFocus)
        MenuBar.setWindowTitle("")
        MenuBar.setStatusTip("")
        MenuBar.setWhatsThis("")
        MenuBar.setAccessibleName("")
        MenuBar.setAutoFillBackground(False)
        MenuBar.setStyleSheet("")
        MenuBar.setDocumentMode(False)
        MenuBar.setDockNestingEnabled(False)
        MenuBar.setUnifiedTitleAndToolBarOnMac(False)
        self.centralWidget = QtWidgets.QWidget(MenuBar)
        self.centralWidget.setStyleSheet("background rgb(0,0,0)")
        self.centralWidget.setObjectName("centralWidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralWidget)
        self.gridLayout_2.setContentsMargins(11, 11, 11, 11)
        self.gridLayout_2.setSpacing(6)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.list_LayoutBox = QtWidgets.QHBoxLayout()
        self.list_LayoutBox.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.list_LayoutBox.setSpacing(6)
        self.list_LayoutBox.setObjectName("list_LayoutBox")
        self.gridLayout_2.addLayout(self.list_LayoutBox, 0, 0, 1, 1)
        self.mpl_LayoutBox = QtWidgets.QHBoxLayout()
        self.mpl_LayoutBox.setSpacing(6)
        self.mpl_LayoutBox.setObjectName("mpl_LayoutBox")
        self.gridLayout_2.addLayout(self.mpl_LayoutBox, 0, 1, 1, 1)
        MenuBar.setCentralWidget(self.centralWidget)
        self.menuBar = QtWidgets.QMenuBar(MenuBar)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 500, 22))
        self.menuBar.setNativeMenuBar(True)
        self.menuBar.setObjectName("menuBar")
        self.menuFile = QtWidgets.QMenu(self.menuBar)
        self.menuFile.setObjectName("menuFile")
        self.menuExport = QtWidgets.QMenu(self.menuFile)
        self.menuExport.setObjectName("menuExport")
        self.menuEdit = QtWidgets.QMenu(self.menuBar)
        self.menuEdit.setObjectName("menuEdit")
        self.menuAnalyze = QtWidgets.QMenu(self.menuBar)
        self.menuAnalyze.setObjectName("menuAnalyze")
        self.menuManually_Select_Bleaching = QtWidgets.QMenu(self.menuAnalyze)
        self.menuManually_Select_Bleaching.setObjectName(
            "menuManually_Select_Bleaching"
        )
        self.menuPredict_Trace_Type = QtWidgets.QMenu(self.menuAnalyze)
        self.menuPredict_Trace_Type.setObjectName("menuPredict_Trace_Type")
        self.menuWindow = QtWidgets.QMenu(self.menuBar)
        self.menuWindow.setObjectName("menuWindow")
        self.menuHelp = QtWidgets.QMenu(self.menuBar)
        self.menuHelp.setObjectName("menuHelp")
        self.menuView = QtWidgets.QMenu(self.menuBar)
        self.menuView.setObjectName("menuView")
        MenuBar.setMenuBar(self.menuBar)
        self.actionOpen = QtWidgets.QAction(MenuBar)
        self.actionOpen.setObjectName("actionOpen")
        self.actionSave = QtWidgets.QAction(MenuBar)
        self.actionSave.setObjectName("actionSave")
        self.actionRemove_File = QtWidgets.QAction(MenuBar)
        self.actionRemove_File.setEnabled(False)
        self.actionRemove_File.setObjectName("actionRemove_File")
        self.actionMinimize = QtWidgets.QAction(MenuBar)
        self.actionMinimize.setObjectName("actionMinimize")
        self.actionFind_Show_Traces = QtWidgets.QAction(MenuBar)
        self.actionFind_Show_Traces.setEnabled(False)
        self.actionFind_Show_Traces.setObjectName("actionFind_Show_Traces")
        self.actionClose = QtWidgets.QAction(MenuBar)
        self.actionClose.setEnabled(True)
        self.actionClose.setObjectName("actionClose")
        self.actionGet_Help_Online = QtWidgets.QAction(MenuBar)
        self.actionGet_Help_Online.setObjectName("actionGet_Help_Online")
        self.actionDebug = QtWidgets.QAction(MenuBar)
        self.actionDebug.setObjectName("actionDebug")
        self.actionPreferences = QtWidgets.QAction(MenuBar)
        self.actionPreferences.setObjectName("actionPreferences")
        self.actionAbout = QtWidgets.QAction(MenuBar)
        self.actionAbout.setObjectName("actionAbout")
        self.actionColocalize_All = QtWidgets.QAction(MenuBar)
        self.actionColocalize_All.setEnabled(False)
        self.actionColocalize_All.setObjectName("actionColocalize_All")
        self.actionColor_Red = QtWidgets.QAction(MenuBar)
        self.actionColor_Red.setEnabled(False)
        self.actionColor_Red.setObjectName("actionColor_Red")
        self.actionSort_by_Red_Bleach = QtWidgets.QAction(MenuBar)
        self.actionSort_by_Red_Bleach.setEnabled(False)
        self.actionSort_by_Red_Bleach.setObjectName("actionSort_by_Red_Bleach")
        self.actionClear_Color = QtWidgets.QAction(MenuBar)
        self.actionClear_Color.setEnabled(False)
        self.actionClear_Color.setObjectName("actionClear_Color")
        self.actionColor_Yellow = QtWidgets.QAction(MenuBar)
        self.actionColor_Yellow.setEnabled(False)
        self.actionColor_Yellow.setObjectName("actionColor_Yellow")
        self.actionColor_Green = QtWidgets.QAction(MenuBar)
        self.actionColor_Green.setEnabled(False)
        self.actionColor_Green.setObjectName("actionColor_Green")
        self.actionClear_All_Colors = QtWidgets.QAction(MenuBar)
        self.actionClear_All_Colors.setEnabled(False)
        self.actionClear_All_Colors.setObjectName("actionClear_All_Colors")
        self.actionSort_by_Ascending = QtWidgets.QAction(MenuBar)
        self.actionSort_by_Ascending.setEnabled(False)
        self.actionSort_by_Ascending.setObjectName("actionSort_by_Ascending")
        self.actionSave_Session = QtWidgets.QAction(MenuBar)
        self.actionSave_Session.setObjectName("actionSave_Session")
        self.actionLoad_Session = QtWidgets.QAction(MenuBar)
        self.actionLoad_Session.setObjectName("actionLoad_Session")
        self.actionRemove_All_Files = QtWidgets.QAction(MenuBar)
        self.actionRemove_All_Files.setEnabled(False)
        self.actionRemove_All_Files.setObjectName("actionRemove_All_Files")
        self.actionGet_alphaFactor = QtWidgets.QAction(MenuBar)
        self.actionGet_alphaFactor.setEnabled(False)
        self.actionGet_alphaFactor.setObjectName("actionGet_alphaFactor")
        self.actionGet_deltaFactor = QtWidgets.QAction(MenuBar)
        self.actionGet_deltaFactor.setEnabled(False)
        self.actionGet_deltaFactor.setObjectName("actionGet_deltaFactor")
        self.actionClear_Correction_Factors = QtWidgets.QAction(MenuBar)
        self.actionClear_Correction_Factors.setEnabled(False)
        self.actionClear_Correction_Factors.setObjectName(
            "actionClear_Correction_Factors"
        )
        self.actionTraceWindow = QtWidgets.QAction(MenuBar)
        self.actionTraceWindow.setObjectName("actionTraceWindow")
        self.actionMainWindow = QtWidgets.QAction(MenuBar)
        self.actionMainWindow.setObjectName("actionMainWindow")
        self.actionHistogramWindow = QtWidgets.QAction(MenuBar)
        self.actionHistogramWindow.setObjectName("actionHistogramWindow")
        self.actionExport_Correction_Factors = QtWidgets.QAction(MenuBar)
        self.actionExport_Correction_Factors.setObjectName(
            "actionExport_Correction_Factors"
        )
        self.actionExport_Colocalization = QtWidgets.QAction(MenuBar)
        self.actionExport_Colocalization.setObjectName(
            "actionExport_Colocalization"
        )
        self.actionExport_Selected_Traces = QtWidgets.QAction(MenuBar)
        self.actionExport_Selected_Traces.setObjectName(
            "actionExport_Selected_Traces"
        )
        self.actionExport_All_Traces = QtWidgets.QAction(MenuBar)
        self.actionExport_All_Traces.setObjectName("actionExport_All_Traces")
        self.actionSort_by_Green_Bleach = QtWidgets.QAction(MenuBar)
        self.actionSort_by_Green_Bleach.setEnabled(False)
        self.actionSort_by_Green_Bleach.setObjectName(
            "actionSort_by_Green_Bleach"
        )
        self.actionClear_Traces = QtWidgets.QAction(MenuBar)
        self.actionClear_Traces.setEnabled(False)
        self.actionClear_Traces.setObjectName("actionClear_Traces")
        self.actionExport_ES_Histogram_Data = QtWidgets.QAction(MenuBar)
        self.actionExport_ES_Histogram_Data.setObjectName(
            "actionExport_ES_Histogram_Data"
        )
        self.actionTransitionDensityWindow = QtWidgets.QAction(MenuBar)
        self.actionTransitionDensityWindow.setObjectName(
            "actionTransitionDensityWindow"
        )
        self.actionCorrectionFactorsWindow = QtWidgets.QAction(MenuBar)
        self.actionCorrectionFactorsWindow.setEnabled(False)
        self.actionCorrectionFactorsWindow.setObjectName(
            "actionCorrectionFactorsWindow"
        )
        self.actionFit_Hmm_Current = QtWidgets.QAction(MenuBar)
        self.actionFit_Hmm_Current.setEnabled(False)
        self.actionFit_Hmm_Current.setObjectName("actionFit_Hmm_Current")
        self.actionFit_Hmm_Selected = QtWidgets.QAction(MenuBar)
        self.actionFit_Hmm_Selected.setEnabled(False)
        self.actionFit_Hmm_Selected.setObjectName("actionFit_Hmm_Selected")
        self.actionBatch_analyze = QtWidgets.QAction(MenuBar)
        self.actionBatch_analyze.setObjectName("actionBatch_analyze")
        self.actionSort_by_Equal_Stoichiometry = QtWidgets.QAction(MenuBar)
        self.actionSort_by_Equal_Stoichiometry.setEnabled(False)
        self.actionSort_by_Equal_Stoichiometry.setObjectName(
            "actionSort_by_Equal_Stoichiometry"
        )
        self.actionEdit_Plot = QtWidgets.QAction(MenuBar)
        self.actionEdit_Plot.setObjectName("actionEdit_Plot")
        self.actionFormat_Plot = QtWidgets.QAction(MenuBar)
        self.actionFormat_Plot.setEnabled(False)
        self.actionFormat_Plot.setObjectName("actionFormat_Plot")
        self.actionUncheck_All = QtWidgets.QAction(MenuBar)
        self.actionUncheck_All.setEnabled(False)
        self.actionUncheck_All.setObjectName("actionUncheck_All")
        self.actionClear_and_Rerun = QtWidgets.QAction(MenuBar)
        self.actionClear_and_Rerun.setEnabled(False)
        self.actionClear_and_Rerun.setObjectName("actionClear_and_Rerun")
        self.actionSelect_Bleach_Red_Channel = QtWidgets.QAction(MenuBar)
        self.actionSelect_Bleach_Red_Channel.setEnabled(False)
        self.actionSelect_Bleach_Red_Channel.setObjectName(
            "actionSelect_Bleach_Red_Channel"
        )
        self.actionSelect_Bleach_Green_Channel = QtWidgets.QAction(MenuBar)
        self.actionSelect_Bleach_Green_Channel.setEnabled(False)
        self.actionSelect_Bleach_Green_Channel.setObjectName(
            "actionSelect_Bleach_Green_Channel"
        )
        self.actionAdvanced_Sort = QtWidgets.QAction(MenuBar)
        self.actionAdvanced_Sort.setEnabled(False)
        self.actionAdvanced_Sort.setObjectName("actionAdvanced_Sort")
        self.actionCheck_All = QtWidgets.QAction(MenuBar)
        self.actionCheck_All.setEnabled(False)
        self.actionCheck_All.setObjectName("actionCheck_All")
        self.actionExport_Transition_Density_Data = QtWidgets.QAction(MenuBar)
        self.actionExport_Transition_Density_Data.setObjectName(
            "actionExport_Transition_Density_Data"
        )
        self.actionSelect_All = QtWidgets.QAction(MenuBar)
        self.actionSelect_All.setEnabled(True)
        self.actionSelect_All.setObjectName("actionSelect_All")
        self.actionDeselect_All = QtWidgets.QAction(MenuBar)
        self.actionDeselect_All.setObjectName("actionDeselect_All")
        self.actionLifetimes = QtWidgets.QAction(MenuBar)
        self.actionLifetimes.setObjectName("actionLifetimes")
        self.actionPredict_Selected_Traces = QtWidgets.QAction(MenuBar)
        self.actionPredict_Selected_Traces.setObjectName(
            "actionPredict_Selected_Traces"
        )
        self.actionPredict_All_traces = QtWidgets.QAction(MenuBar)
        self.actionPredict_All_traces.setObjectName("actionPredict_All_traces")
        self.actionClear_All_Predictions = QtWidgets.QAction(MenuBar)
        self.actionClear_All_Predictions.setEnabled(False)
        self.actionClear_All_Predictions.setObjectName(
            "actionClear_All_Predictions"
        )
        self.actionTraceSimulatorWindow = QtWidgets.QAction(MenuBar)
        self.actionTraceSimulatorWindow.setObjectName(
            "actionTraceSimulatorWindow"
        )
        self.menuExport.addAction(self.actionExport_Correction_Factors)
        self.menuExport.addAction(self.actionExport_Colocalization)
        self.menuExport.addAction(self.actionExport_Selected_Traces)
        self.menuExport.addAction(self.actionExport_All_Traces)
        self.menuExport.addAction(self.actionExport_ES_Histogram_Data)
        self.menuExport.addAction(self.actionExport_Transition_Density_Data)
        self.menuFile.addAction(self.actionOpen)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionClose)
        self.menuFile.addAction(self.actionSave)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.menuExport.menuAction())
        self.menuEdit.addAction(self.actionRemove_File)
        self.menuEdit.addAction(self.actionRemove_All_Files)
        self.menuEdit.addAction(self.actionSelect_All)
        self.menuEdit.addAction(self.actionDeselect_All)
        self.menuManually_Select_Bleaching.addAction(
            self.actionSelect_Bleach_Red_Channel
        )
        self.menuManually_Select_Bleaching.addAction(
            self.actionSelect_Bleach_Green_Channel
        )
        self.menuPredict_Trace_Type.addAction(
            self.actionPredict_Selected_Traces
        )
        self.menuPredict_Trace_Type.addAction(self.actionPredict_All_traces)
        self.menuAnalyze.addAction(self.actionColocalize_All)
        self.menuAnalyze.addAction(self.actionClear_Traces)
        self.menuAnalyze.addAction(self.actionFind_Show_Traces)
        self.menuAnalyze.addAction(self.actionClear_and_Rerun)
        self.menuAnalyze.addSeparator()
        self.menuAnalyze.addAction(self.actionColor_Red)
        self.menuAnalyze.addAction(self.actionColor_Yellow)
        self.menuAnalyze.addAction(self.actionColor_Green)
        self.menuAnalyze.addAction(self.actionClear_Color)
        self.menuAnalyze.addAction(self.actionClear_All_Colors)
        self.menuAnalyze.addSeparator()
        self.menuAnalyze.addAction(self.actionCorrectionFactorsWindow)
        self.menuAnalyze.addAction(self.actionGet_alphaFactor)
        self.menuAnalyze.addAction(self.actionGet_deltaFactor)
        self.menuAnalyze.addAction(self.actionClear_Correction_Factors)
        self.menuAnalyze.addSeparator()
        self.menuAnalyze.addAction(
            self.menuManually_Select_Bleaching.menuAction()
        )
        self.menuAnalyze.addSeparator()
        self.menuAnalyze.addAction(self.actionFit_Hmm_Selected)
        self.menuAnalyze.addAction(self.menuPredict_Trace_Type.menuAction())
        self.menuAnalyze.addAction(self.actionClear_All_Predictions)
        self.menuWindow.addAction(self.actionMinimize)
        self.menuWindow.addSeparator()
        self.menuWindow.addAction(self.actionMainWindow)
        self.menuWindow.addAction(self.actionTraceWindow)
        self.menuWindow.addAction(self.actionHistogramWindow)
        self.menuWindow.addAction(self.actionTransitionDensityWindow)
        self.menuWindow.addAction(self.actionTraceSimulatorWindow)
        self.menuHelp.addAction(self.actionGet_Help_Online)
        self.menuHelp.addSeparator()
        self.menuHelp.addAction(self.actionPreferences)
        self.menuHelp.addAction(self.actionAbout)
        self.menuHelp.addAction(self.actionDebug)
        self.menuView.addAction(self.actionFormat_Plot)
        self.menuView.addSeparator()
        self.menuView.addAction(self.actionAdvanced_Sort)
        self.menuView.addAction(self.actionSort_by_Ascending)
        self.menuView.addAction(self.actionSort_by_Red_Bleach)
        self.menuView.addAction(self.actionSort_by_Green_Bleach)
        self.menuView.addAction(self.actionSort_by_Equal_Stoichiometry)
        self.menuBar.addAction(self.menuFile.menuAction())
        self.menuBar.addAction(self.menuEdit.menuAction())
        self.menuBar.addAction(self.menuView.menuAction())
        self.menuBar.addAction(self.menuAnalyze.menuAction())
        self.menuBar.addAction(self.menuWindow.menuAction())
        self.menuBar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(MenuBar)
        QtCore.QMetaObject.connectSlotsByName(MenuBar)

    def retranslateUi(self, MenuBar):
        _translate = QtCore.QCoreApplication.translate
        self.menuFile.setTitle(_translate("MenuBar", "File"))
        self.menuExport.setTitle(_translate("MenuBar", "Export"))
        self.menuEdit.setTitle(_translate("MenuBar", "Edit"))
        self.menuAnalyze.setTitle(_translate("MenuBar", "Analyze"))
        self.menuManually_Select_Bleaching.setTitle(
            _translate("MenuBar", "Manually Select Bleaching")
        )
        self.menuPredict_Trace_Type.setTitle(
            _translate("MenuBar", "Predict Trace Type")
        )
        self.menuWindow.setTitle(_translate("MenuBar", "Window"))
        self.menuHelp.setTitle(_translate("MenuBar", "Help"))
        self.menuView.setTitle(_translate("MenuBar", "View"))
        self.actionOpen.setText(_translate("MenuBar", "Open Files"))
        self.actionOpen.setShortcut(_translate("MenuBar", "Ctrl+O"))
        self.actionSave.setText(_translate("MenuBar", "Save Plot"))
        self.actionSave.setShortcut(_translate("MenuBar", "Ctrl+S"))
        self.actionRemove_File.setText(
            _translate("MenuBar", "Remove Current From List")
        )
        self.actionRemove_File.setShortcut(
            _translate("MenuBar", "Ctrl+Backspace")
        )
        self.actionMinimize.setText(_translate("MenuBar", "Minimize"))
        self.actionMinimize.setShortcut(_translate("MenuBar", "Ctrl+M"))
        self.actionFind_Show_Traces.setText(
            _translate("MenuBar", "Show Traces")
        )
        self.actionFind_Show_Traces.setShortcut(
            _translate("MenuBar", "Ctrl+Shift+T")
        )
        self.actionClose.setText(_translate("MenuBar", "Close Window"))
        self.actionClose.setShortcut(_translate("MenuBar", "Ctrl+W"))
        self.actionGet_Help_Online.setText(
            _translate("MenuBar", "Get Help Online")
        )
        self.actionDebug.setText(_translate("MenuBar", "Debug..."))
        self.actionPreferences.setText(_translate("MenuBar", "Preferences"))
        self.actionPreferences.setShortcut(_translate("MenuBar", "Ctrl+,"))
        self.actionAbout.setText(_translate("MenuBar", "About"))
        self.actionColocalize_All.setText(
            _translate("MenuBar", "Colocalize All")
        )
        self.actionColocalize_All.setShortcut(
            _translate("MenuBar", "Ctrl+Alt+L")
        )
        self.actionColor_Red.setText(_translate("MenuBar", "Color Red"))
        self.actionColor_Red.setShortcut(_translate("MenuBar", "Meta+Shift+R"))
        self.actionSort_by_Red_Bleach.setText(
            _translate("MenuBar", "Sort by Red Bleach")
        )
        self.actionClear_Color.setText(_translate("MenuBar", "Clear Color"))
        self.actionClear_Color.setShortcut(
            _translate("MenuBar", "Meta+Shift+C")
        )
        self.actionColor_Yellow.setText(_translate("MenuBar", "Color Yellow"))
        self.actionColor_Yellow.setShortcut(
            _translate("MenuBar", "Meta+Shift+Y")
        )
        self.actionColor_Green.setText(_translate("MenuBar", "Color Green"))
        self.actionColor_Green.setToolTip(_translate("MenuBar", "Color Green"))
        self.actionColor_Green.setShortcut(
            _translate("MenuBar", "Meta+Shift+G")
        )
        self.actionClear_All_Colors.setText(
            _translate("MenuBar", "Clear All Colors")
        )
        self.actionClear_All_Colors.setShortcut(
            _translate("MenuBar", "Meta+Alt+Shift+C")
        )
        self.actionSort_by_Ascending.setText(
            _translate("MenuBar", "Sort by Ascending")
        )
        self.actionSave_Session.setText(
            _translate("MenuBar", "Save Session (Beta)")
        )
        self.actionLoad_Session.setText(
            _translate("MenuBar", "Load Session (Beta)")
        )
        self.actionRemove_All_Files.setText(
            _translate("MenuBar", "Remove All From List")
        )
        self.actionGet_alphaFactor.setText(
            _translate("MenuBar", "Find ɑ-Factor (for D-only)")
        )
        self.actionGet_alphaFactor.setShortcut(
            _translate("MenuBar", "Meta+Shift+A")
        )
        self.actionGet_deltaFactor.setText(
            _translate("MenuBar", "Find δ-Factor (for A-only)")
        )
        self.actionGet_deltaFactor.setShortcut(
            _translate("MenuBar", "Meta+Shift+D")
        )
        self.actionClear_Correction_Factors.setText(
            _translate("MenuBar", "Clear Correction Factors")
        )
        self.actionClear_Correction_Factors.setShortcut(
            _translate("MenuBar", "Meta+Shift+C")
        )
        self.actionTraceWindow.setText(_translate("MenuBar", "Traces"))
        self.actionTraceWindow.setShortcut(_translate("MenuBar", "Ctrl+2"))
        self.actionMainWindow.setText(_translate("MenuBar", "Images"))
        self.actionMainWindow.setShortcut(_translate("MenuBar", "Ctrl+1"))
        self.actionHistogramWindow.setText(_translate("MenuBar", "Histogram"))
        self.actionHistogramWindow.setShortcut(_translate("MenuBar", "Ctrl+3"))
        self.actionExport_Correction_Factors.setText(
            _translate("MenuBar", "Correction Factors")
        )
        self.actionExport_Colocalization.setText(
            _translate("MenuBar", "Colocalization")
        )
        self.actionExport_Selected_Traces.setText(
            _translate("MenuBar", "Selected Traces")
        )
        self.actionExport_All_Traces.setText(
            _translate("MenuBar", "All Traces")
        )
        self.actionSort_by_Green_Bleach.setText(
            _translate("MenuBar", "Sort by Green Bleach")
        )
        self.actionClear_Traces.setText(_translate("MenuBar", "Clear Traces"))
        self.actionExport_ES_Histogram_Data.setText(
            _translate("MenuBar", "E-S Histogram Data")
        )
        self.actionTransitionDensityWindow.setText(
            _translate("MenuBar", "Transition Density")
        )
        self.actionTransitionDensityWindow.setShortcut(
            _translate("MenuBar", "Ctrl+4")
        )
        self.actionCorrectionFactorsWindow.setText(
            _translate("MenuBar", "Set Correction Factors")
        )
        self.actionCorrectionFactorsWindow.setShortcut(
            _translate("MenuBar", "Ctrl+Shift+K")
        )
        self.actionFit_Hmm_Current.setText(
            _translate("MenuBar", "Current Trace")
        )
        self.actionFit_Hmm_Selected.setText(
            _translate("MenuBar", "Fit Hidden Markov Model to Selected")
        )
        self.actionFit_Hmm_Selected.setShortcut(
            _translate("MenuBar", "Ctrl+Shift+M")
        )
        self.actionBatch_analyze.setText(_translate("MenuBar", "Batch analyze"))
        self.actionBatch_analyze.setShortcut(_translate("MenuBar", "Ctrl+B"))
        self.actionSort_by_Equal_Stoichiometry.setText(
            _translate("MenuBar", "Sort by Equal Stoichiometry")
        )
        self.actionEdit_Plot.setText(_translate("MenuBar", "Edit Plot"))
        self.actionFormat_Plot.setText(_translate("MenuBar", "Format Plot"))
        self.actionUncheck_All.setText(_translate("MenuBar", "Uncheck All"))
        self.actionUncheck_All.setShortcut(
            _translate("MenuBar", "Ctrl+Alt+Shift+A")
        )
        self.actionClear_and_Rerun.setText(
            _translate("MenuBar", "Clear Traces and Rerun")
        )
        self.actionSelect_Bleach_Red_Channel.setText(
            _translate("MenuBar", "Red Channel")
        )
        self.actionSelect_Bleach_Green_Channel.setText(
            _translate("MenuBar", "Green Channel")
        )
        self.actionAdvanced_Sort.setText(_translate("MenuBar", "Advanced Sort"))
        self.actionCheck_All.setText(_translate("MenuBar", "Check All"))
        self.actionCheck_All.setShortcut(_translate("MenuBar", "Ctrl+Alt+A"))
        self.actionExport_Transition_Density_Data.setText(
            _translate("MenuBar", "Transition Density Data")
        )
        self.actionSelect_All.setText(_translate("MenuBar", "Select All"))
        self.actionSelect_All.setShortcut(_translate("MenuBar", "Ctrl+A"))
        self.actionDeselect_All.setText(_translate("MenuBar", "Deselect All"))
        self.actionDeselect_All.setShortcut(_translate("MenuBar", "Ctrl+Alt+A"))
        self.actionLifetimes.setText(
            _translate("MenuBar", "Transition Lifetimes")
        )
        self.actionLifetimes.setShortcut(_translate("MenuBar", "Ctrl+5"))
        self.actionPredict_Selected_Traces.setText(
            _translate("MenuBar", "Selected Traces")
        )
        self.actionPredict_Selected_Traces.setShortcut(
            _translate("MenuBar", "Ctrl+P")
        )
        self.actionPredict_All_traces.setText(
            _translate("MenuBar", "All traces")
        )
        self.actionPredict_All_traces.setShortcut(
            _translate("MenuBar", "Ctrl+Alt+P")
        )
        self.actionClear_All_Predictions.setText(
            _translate("MenuBar", "Clear All Predictions")
        )
        self.actionClear_All_Predictions.setShortcut(
            _translate("MenuBar", "Meta+Ctrl+P")
        )
        self.actionTraceSimulatorWindow.setText(
            _translate("MenuBar", "Trace Simulator")
        )
        self.actionTraceSimulatorWindow.setShortcut(
            _translate("MenuBar", "Ctrl+5")
        )


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MenuBar = QtWidgets.QMainWindow()
    ui = Ui_MenuBar()
    ui.setupUi(MenuBar)
    MenuBar.show()
    sys.exit(app.exec_())
