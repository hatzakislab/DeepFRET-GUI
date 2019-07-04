# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/_LifetimeWindow.ui'
#
# Created by: PyQt5 UI code generator 5.12.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_LifetimeWindow(object):
    def setupUi(self, LifetimeWindow):
        LifetimeWindow.setObjectName("LifetimeWindow")
        LifetimeWindow.resize(700, 700)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(LifetimeWindow.sizePolicy().hasHeightForWidth())
        LifetimeWindow.setSizePolicy(sizePolicy)
        LifetimeWindow.setMinimumSize(QtCore.QSize(700, 700))
        LifetimeWindow.setMaximumSize(QtCore.QSize(5000, 2000))
        LifetimeWindow.setBaseSize(QtCore.QSize(700, 700))
        LifetimeWindow.setWindowOpacity(1.0)
        LifetimeWindow.setAutoFillBackground(False)
        LifetimeWindow.setDocumentMode(False)
        LifetimeWindow.setDockNestingEnabled(False)
        LifetimeWindow.setUnifiedTitleAndToolBarOnMac(False)
        self.centralWidget = QtWidgets.QWidget(LifetimeWindow)
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
        LifetimeWindow.setCentralWidget(self.centralWidget)

        self.retranslateUi(LifetimeWindow)
        QtCore.QMetaObject.connectSlotsByName(LifetimeWindow)

    def retranslateUi(self, LifetimeWindow):
        _translate = QtCore.QCoreApplication.translate
        LifetimeWindow.setWindowTitle(_translate("LifetimeWindow", "Histogram"))




if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    LifetimeWindow = QtWidgets.QMainWindow()
    ui = Ui_LifetimeWindow()
    ui.setupUi(LifetimeWindow)
    LifetimeWindow.show()
    sys.exit(app.exec_())
