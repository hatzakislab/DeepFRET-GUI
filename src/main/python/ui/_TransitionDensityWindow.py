# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'src/main/python/ui/TransitionDensityWindow.ui'
#
# Created by: PyQt5 UI code generator 5.14.2
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_TransitionDensityWindow(object):
    def setupUi(self, TransitionDensityWindow):
        TransitionDensityWindow.setObjectName("TransitionDensityWindow")
        TransitionDensityWindow.resize(1200, 600)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            TransitionDensityWindow.sizePolicy().hasHeightForWidth()
        )
        TransitionDensityWindow.setSizePolicy(sizePolicy)
        TransitionDensityWindow.setMinimumSize(QtCore.QSize(1200, 600))
        TransitionDensityWindow.setMaximumSize(QtCore.QSize(5000, 2000))
        TransitionDensityWindow.setBaseSize(QtCore.QSize(1400, 700))
        TransitionDensityWindow.setWindowOpacity(1.0)
        TransitionDensityWindow.setAutoFillBackground(False)
        TransitionDensityWindow.setDocumentMode(False)
        TransitionDensityWindow.setDockNestingEnabled(False)
        TransitionDensityWindow.setUnifiedTitleAndToolBarOnMac(False)
        self.centralWidget = QtWidgets.QWidget(TransitionDensityWindow)
        self.centralWidget.setObjectName("centralWidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralWidget)
        self.gridLayout_2.setContentsMargins(11, 11, 11, 11)
        self.gridLayout_2.setSpacing(6)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.mpl_LayoutBox = QtWidgets.QVBoxLayout()
        self.mpl_LayoutBox.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        self.mpl_LayoutBox.setContentsMargins(0, 0, 0, 0)
        self.mpl_LayoutBox.setObjectName("mpl_LayoutBox")
        self.gridLayout_2.addLayout(self.mpl_LayoutBox, 1, 0, 1, 1)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setSpacing(6)
        self.gridLayout.setObjectName("gridLayout")
        self.nClustersSpinBox = QtWidgets.QSpinBox(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.nClustersSpinBox.sizePolicy().hasHeightForWidth()
        )
        self.nClustersSpinBox.setSizePolicy(sizePolicy)
        self.nClustersSpinBox.setMinimumSize(QtCore.QSize(30, 0))
        self.nClustersSpinBox.setFrame(True)
        self.nClustersSpinBox.setButtonSymbols(
            QtWidgets.QAbstractSpinBox.NoButtons
        )
        self.nClustersSpinBox.setMinimum(1)
        self.nClustersSpinBox.setMaximum(9)
        self.nClustersSpinBox.setObjectName("nClustersSpinBox")
        self.gridLayout.addWidget(self.nClustersSpinBox, 0, 1, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(
            40,
            20,
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Minimum,
        )
        self.gridLayout.addItem(spacerItem, 0, 2, 1, 1)
        self.label = QtWidgets.QLabel(self.centralWidget)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)
        TransitionDensityWindow.setCentralWidget(self.centralWidget)

        self.retranslateUi(TransitionDensityWindow)
        QtCore.QMetaObject.connectSlotsByName(TransitionDensityWindow)

    def retranslateUi(self, TransitionDensityWindow):
        _translate = QtCore.QCoreApplication.translate
        TransitionDensityWindow.setWindowTitle(
            _translate("TransitionDensityWindow", "Transition Density Plot")
        )
        self.label.setText(
            _translate("TransitionDensityWindow", "Number of Clusters Per Half")
        )


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    TransitionDensityWindow = QtWidgets.QMainWindow()
    ui = Ui_TransitionDensityWindow()
    ui.setupUi(TransitionDensityWindow)
    TransitionDensityWindow.show()
    sys.exit(app.exec_())
