# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/_TraceWindow.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_TraceWindow(object):
    def setupUi(self, TraceWindow):
        TraceWindow.setObjectName("TraceWindow")
        TraceWindow.resize(1200, 500)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(TraceWindow.sizePolicy().hasHeightForWidth())
        TraceWindow.setSizePolicy(sizePolicy)
        TraceWindow.setMinimumSize(QtCore.QSize(1200, 500))
        TraceWindow.setMaximumSize(QtCore.QSize(5000, 2000))
        TraceWindow.setBaseSize(QtCore.QSize(1400, 700))
        TraceWindow.setAutoFillBackground(False)
        TraceWindow.setDocumentMode(False)
        TraceWindow.setDockNestingEnabled(False)
        TraceWindow.setUnifiedTitleAndToolBarOnMac(False)
        self.centralWidget = QtWidgets.QWidget(TraceWindow)
        self.centralWidget.setObjectName("centralWidget")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.centralWidget)
        self.horizontalLayout_3.setContentsMargins(11, 11, 11, 11)
        self.horizontalLayout_3.setSpacing(6)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.layoutBox = QtWidgets.QHBoxLayout()
        self.layoutBox.setSpacing(6)
        self.layoutBox.setObjectName("layoutBox")
        self.horizontalLayout_3.addLayout(self.layoutBox)
        TraceWindow.setCentralWidget(self.centralWidget)

        self.retranslateUi(TraceWindow)
        QtCore.QMetaObject.connectSlotsByName(TraceWindow)

    def retranslateUi(self, TraceWindow):
        _translate = QtCore.QCoreApplication.translate
        TraceWindow.setWindowTitle(_translate("TraceWindow", "Traces"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    TraceWindow = QtWidgets.QMainWindow()
    ui = Ui_TraceWindow()
    ui.setupUi(TraceWindow)
    TraceWindow.show()
    sys.exit(app.exec_())
