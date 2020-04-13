# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/HistogramWindow.ui'
#
# Created by: PyQt5 UI code generator 5.14.2
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_HistogramWindow(object):
    def setupUi(self, HistogramWindow):
        HistogramWindow.setObjectName("HistogramWindow")
        HistogramWindow.resize(1200, 1200)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            HistogramWindow.sizePolicy().hasHeightForWidth()
        )
        HistogramWindow.setSizePolicy(sizePolicy)
        HistogramWindow.setMinimumSize(QtCore.QSize(1200, 1200))
        HistogramWindow.setMaximumSize(QtCore.QSize(5000, 2000))
        HistogramWindow.setBaseSize(QtCore.QSize(1200, 1200))
        HistogramWindow.setWindowOpacity(1.0)
        HistogramWindow.setAutoFillBackground(False)
        HistogramWindow.setDocumentMode(False)
        HistogramWindow.setDockNestingEnabled(False)
        HistogramWindow.setUnifiedTitleAndToolBarOnMac(False)
        self.centralWidget = QtWidgets.QWidget(HistogramWindow)
        self.centralWidget.setObjectName("centralWidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralWidget)
        self.gridLayout_2.setContentsMargins(11, 11, 11, 11)
        self.gridLayout_2.setSpacing(6)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.mpl_LayoutBox = QtWidgets.QVBoxLayout()
        self.mpl_LayoutBox.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        self.mpl_LayoutBox.setContentsMargins(0, 0, 0, 0)
        self.mpl_LayoutBox.setSpacing(6)
        self.mpl_LayoutBox.setObjectName("mpl_LayoutBox")
        self.gridLayout_2.addLayout(self.mpl_LayoutBox, 2, 0, 1, 1)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setSpacing(6)
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(self.centralWidget)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.gaussianSpinBox = QtWidgets.QSpinBox(self.centralWidget)
        self.gaussianSpinBox.setMinimumSize(QtCore.QSize(0, 21))
        self.gaussianSpinBox.setFrame(True)
        self.gaussianSpinBox.setButtonSymbols(
            QtWidgets.QAbstractSpinBox.NoButtons
        )
        self.gaussianSpinBox.setKeyboardTracking(True)
        self.gaussianSpinBox.setMinimum(1)
        self.gaussianSpinBox.setMaximum(5)
        self.gaussianSpinBox.setObjectName("gaussianSpinBox")
        self.gridLayout.addWidget(self.gaussianSpinBox, 0, 1, 1, 1)
        self.gaussianAutoButton = QtWidgets.QPushButton(self.centralWidget)
        self.gaussianAutoButton.setObjectName("gaussianAutoButton")
        self.gridLayout.addWidget(self.gaussianAutoButton, 0, 2, 1, 1)
        self.framesSpinBox = QtWidgets.QSpinBox(self.centralWidget)
        self.framesSpinBox.setButtonSymbols(
            QtWidgets.QAbstractSpinBox.NoButtons
        )
        self.framesSpinBox.setKeyboardTracking(False)
        self.framesSpinBox.setMinimum(1)
        self.framesSpinBox.setMaximum(100)
        self.framesSpinBox.setProperty("value", 10)
        self.framesSpinBox.setObjectName("framesSpinBox")
        self.gridLayout.addWidget(self.framesSpinBox, 1, 1, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.centralWidget)
        self.label_3.setText("")
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 2, 2, 1, 1)
        self.applyCorrectionsCheckBox = QtWidgets.QCheckBox(self.centralWidget)
        self.applyCorrectionsCheckBox.setObjectName("applyCorrectionsCheckBox")
        self.gridLayout.addWidget(self.applyCorrectionsCheckBox, 2, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.centralWidget)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(
            40,
            20,
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Minimum,
        )
        self.gridLayout.addItem(spacerItem, 0, 3, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)
        HistogramWindow.setCentralWidget(self.centralWidget)

        self.retranslateUi(HistogramWindow)
        QtCore.QMetaObject.connectSlotsByName(HistogramWindow)

    def retranslateUi(self, HistogramWindow):
        _translate = QtCore.QCoreApplication.translate
        HistogramWindow.setWindowTitle(
            _translate("HistogramWindow", "Histogram")
        )
        self.label.setText(
            _translate("HistogramWindow", "Number of Gaussians:")
        )
        self.gaussianAutoButton.setText(_translate("HistogramWindow", "Auto"))
        self.applyCorrectionsCheckBox.setText(
            _translate("HistogramWindow", "Apply β / ɣ Corrections")
        )
        self.label_2.setText(
            _translate("HistogramWindow", "Max number of frames:")
        )


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    HistogramWindow = QtWidgets.QMainWindow()
    ui = Ui_HistogramWindow()
    ui.setupUi(HistogramWindow)
    HistogramWindow.show()
    sys.exit(app.exec_())
