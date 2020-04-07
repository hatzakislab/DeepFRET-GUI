# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/TraceWindowInspector.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_TraceWindowInspector(object):
    def setupUi(self, TraceWindowInspector):
        TraceWindowInspector.setObjectName("TraceWindowInspector")
        TraceWindowInspector.resize(347, 238)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            TraceWindowInspector.sizePolicy().hasHeightForWidth()
        )
        TraceWindowInspector.setSizePolicy(sizePolicy)
        TraceWindowInspector.setMinimumSize(QtCore.QSize(0, 0))
        TraceWindowInspector.setMaximumSize(QtCore.QSize(1000, 1000))
        self.gridLayout = QtWidgets.QGridLayout(TraceWindowInspector)
        self.gridLayout.setContentsMargins(11, 11, 11, 11)
        self.gridLayout.setSpacing(6)
        self.gridLayout.setObjectName("gridLayout")
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setSpacing(6)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.FretLabel = QtWidgets.QLabel(TraceWindowInspector)
        self.FretLabel.setObjectName("FretLabel")
        self.gridLayout_2.addWidget(self.FretLabel, 1, 0, 1, 1)
        self.spinBoxStoiLo = QtWidgets.QDoubleSpinBox(TraceWindowInspector)
        self.spinBoxStoiLo.setFrame(True)
        self.spinBoxStoiLo.setButtonSymbols(
            QtWidgets.QAbstractSpinBox.NoButtons
        )
        self.spinBoxStoiLo.setMaximum(1.0)
        self.spinBoxStoiLo.setSingleStep(0.1)
        self.spinBoxStoiLo.setObjectName("spinBoxStoiLo")
        self.gridLayout_2.addWidget(self.spinBoxStoiLo, 0, 1, 1, 1)
        self.FramesLabel = QtWidgets.QLabel(TraceWindowInspector)
        self.FramesLabel.setObjectName("FramesLabel")
        self.gridLayout_2.addWidget(self.FramesLabel, 4, 0, 1, 1)
        self.spinBoxFretLo = QtWidgets.QDoubleSpinBox(TraceWindowInspector)
        self.spinBoxFretLo.setFrame(True)
        self.spinBoxFretLo.setButtonSymbols(
            QtWidgets.QAbstractSpinBox.NoButtons
        )
        self.spinBoxFretLo.setMaximum(1.0)
        self.spinBoxFretLo.setSingleStep(0.1)
        self.spinBoxFretLo.setObjectName("spinBoxFretLo")
        self.gridLayout_2.addWidget(self.spinBoxFretLo, 1, 1, 1, 1)
        self.FramesLabel_3 = QtWidgets.QLabel(TraceWindowInspector)
        self.FramesLabel_3.setObjectName("FramesLabel_3")
        self.gridLayout_2.addWidget(self.FramesLabel_3, 3, 0, 1, 1)
        self.spinBoxDynamics = QtWidgets.QDoubleSpinBox(TraceWindowInspector)
        self.spinBoxDynamics.setFrame(True)
        self.spinBoxDynamics.setButtonSymbols(
            QtWidgets.QAbstractSpinBox.NoButtons
        )
        self.spinBoxDynamics.setMaximum(1.0)
        self.spinBoxDynamics.setSingleStep(0.05)
        self.spinBoxDynamics.setObjectName("spinBoxDynamics")
        self.gridLayout_2.addWidget(self.spinBoxDynamics, 3, 1, 1, 2)
        self.spinBoxFretHi = QtWidgets.QDoubleSpinBox(TraceWindowInspector)
        self.spinBoxFretHi.setFrame(True)
        self.spinBoxFretHi.setButtonSymbols(
            QtWidgets.QAbstractSpinBox.NoButtons
        )
        self.spinBoxFretHi.setMaximum(1.0)
        self.spinBoxFretHi.setSingleStep(0.1)
        self.spinBoxFretHi.setProperty("value", 1.0)
        self.spinBoxFretHi.setObjectName("spinBoxFretHi")
        self.gridLayout_2.addWidget(self.spinBoxFretHi, 1, 2, 1, 1)
        self.spinBoxStoiHi = QtWidgets.QDoubleSpinBox(TraceWindowInspector)
        self.spinBoxStoiHi.setFrame(True)
        self.spinBoxStoiHi.setButtonSymbols(
            QtWidgets.QAbstractSpinBox.NoButtons
        )
        self.spinBoxStoiHi.setMaximum(1.0)
        self.spinBoxStoiHi.setSingleStep(0.1)
        self.spinBoxStoiHi.setProperty("value", 1.0)
        self.spinBoxStoiHi.setObjectName("spinBoxStoiHi")
        self.gridLayout_2.addWidget(self.spinBoxStoiHi, 0, 2, 1, 1)
        self.FramesLabel_2 = QtWidgets.QLabel(TraceWindowInspector)
        self.FramesLabel_2.setObjectName("FramesLabel_2")
        self.gridLayout_2.addWidget(self.FramesLabel_2, 2, 0, 1, 1)
        self.StoiLabel = QtWidgets.QLabel(TraceWindowInspector)
        self.StoiLabel.setObjectName("StoiLabel")
        self.gridLayout_2.addWidget(self.StoiLabel, 0, 0, 1, 1)
        self.spinBoxMinFrames = QtWidgets.QSpinBox(TraceWindowInspector)
        self.spinBoxMinFrames.setButtonSymbols(
            QtWidgets.QAbstractSpinBox.NoButtons
        )
        self.spinBoxMinFrames.setMaximum(1000)
        self.spinBoxMinFrames.setProperty("value", 15)
        self.spinBoxMinFrames.setObjectName("spinBoxMinFrames")
        self.gridLayout_2.addWidget(self.spinBoxMinFrames, 4, 1, 1, 2)
        self.pushButtonFind = QtWidgets.QPushButton(TraceWindowInspector)
        self.pushButtonFind.setObjectName("pushButtonFind")
        self.gridLayout_2.addWidget(self.pushButtonFind, 6, 2, 1, 1)
        self.spinBoxConfidence = QtWidgets.QDoubleSpinBox(TraceWindowInspector)
        self.spinBoxConfidence.setFrame(True)
        self.spinBoxConfidence.setButtonSymbols(
            QtWidgets.QAbstractSpinBox.NoButtons
        )
        self.spinBoxConfidence.setMaximum(1.0)
        self.spinBoxConfidence.setSingleStep(0.05)
        self.spinBoxConfidence.setProperty("value", 0.75)
        self.spinBoxConfidence.setObjectName("spinBoxConfidence")
        self.gridLayout_2.addWidget(self.spinBoxConfidence, 2, 1, 1, 2)
        self.checkBoxBleach = QtWidgets.QCheckBox(TraceWindowInspector)
        self.checkBoxBleach.setText("")
        self.checkBoxBleach.setObjectName("checkBoxBleach")
        self.gridLayout_2.addWidget(self.checkBoxBleach, 5, 1, 1, 1)
        self.FramesLabel_4 = QtWidgets.QLabel(TraceWindowInspector)
        self.FramesLabel_4.setObjectName("FramesLabel_4")
        self.gridLayout_2.addWidget(self.FramesLabel_4, 5, 0, 1, 1)
        self.gridLayout.addLayout(self.gridLayout_2, 4, 0, 1, 1)

        self.retranslateUi(TraceWindowInspector)
        QtCore.QMetaObject.connectSlotsByName(TraceWindowInspector)
        TraceWindowInspector.setTabOrder(self.spinBoxStoiLo, self.spinBoxStoiHi)
        TraceWindowInspector.setTabOrder(self.spinBoxStoiHi, self.spinBoxFretLo)
        TraceWindowInspector.setTabOrder(self.spinBoxFretLo, self.spinBoxFretHi)
        TraceWindowInspector.setTabOrder(
            self.spinBoxFretHi, self.spinBoxConfidence
        )
        TraceWindowInspector.setTabOrder(
            self.spinBoxConfidence, self.spinBoxDynamics
        )
        TraceWindowInspector.setTabOrder(
            self.spinBoxDynamics, self.spinBoxMinFrames
        )
        TraceWindowInspector.setTabOrder(
            self.spinBoxMinFrames, self.pushButtonFind
        )

    def retranslateUi(self, TraceWindowInspector):
        _translate = QtCore.QCoreApplication.translate
        TraceWindowInspector.setWindowTitle(
            _translate("TraceWindowInspector", "Adjust")
        )
        self.FretLabel.setText(
            _translate("TraceWindowInspector", "FRET Median Range:")
        )
        self.FramesLabel.setText(
            _translate("TraceWindowInspector", "Minimum Number of Frames:")
        )
        self.FramesLabel_3.setText(
            _translate("TraceWindowInspector", "Minimum Dynamics Confidence")
        )
        self.FramesLabel_2.setText(
            _translate("TraceWindowInspector", "Minimum Trace Confidence:")
        )
        self.StoiLabel.setText(
            _translate("TraceWindowInspector", "Stoichiometry Median Range:")
        )
        self.pushButtonFind.setText(_translate("TraceWindowInspector", "Find"))
        self.FramesLabel_4.setText(
            _translate("TraceWindowInspector", "Trace Must Bleach:")
        )


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    TraceWindowInspector = QtWidgets.QDialog()
    ui = Ui_TraceWindowInspector()
    ui.setupUi(TraceWindowInspector)
    TraceWindowInspector.show()
    sys.exit(app.exec_())
