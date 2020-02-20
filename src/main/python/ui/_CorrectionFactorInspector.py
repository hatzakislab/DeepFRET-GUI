# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/_CorrectionFactorInspector.ui'
#
# Created by: PyQt5 UI code generator 5.13.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_CorrectionFactorInspector(object):
    def setupUi(self, CorrectionFactorInspector):
        CorrectionFactorInspector.setObjectName("CorrectionFactorInspector")
        CorrectionFactorInspector.resize(419, 47)
        self.gridLayout_2 = QtWidgets.QGridLayout(CorrectionFactorInspector)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setObjectName("gridLayout")
        spacerItem = QtWidgets.QSpacerItem(5, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 0, 6, 1, 1)
        self.alphaFactorBox = QtWidgets.QDoubleSpinBox(CorrectionFactorInspector)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.alphaFactorBox.sizePolicy().hasHeightForWidth())
        self.alphaFactorBox.setSizePolicy(sizePolicy)
        self.alphaFactorBox.setMaximumSize(QtCore.QSize(40, 16777215))
        self.alphaFactorBox.setFrame(True)
        self.alphaFactorBox.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.alphaFactorBox.setKeyboardTracking(False)
        self.alphaFactorBox.setObjectName("alphaFactorBox")
        self.gridLayout.addWidget(self.alphaFactorBox, 0, 3, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(15, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem1, 0, 4, 1, 1)
        self.label_2 = QtWidgets.QLabel(CorrectionFactorInspector)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 0, 5, 1, 1)
        self.label = QtWidgets.QLabel(CorrectionFactorInspector)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 1, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(31, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem2, 0, 0, 1, 1)
        self.deltaFactorBox = QtWidgets.QDoubleSpinBox(CorrectionFactorInspector)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.deltaFactorBox.sizePolicy().hasHeightForWidth())
        self.deltaFactorBox.setSizePolicy(sizePolicy)
        self.deltaFactorBox.setMaximumSize(QtCore.QSize(45, 16777215))
        self.deltaFactorBox.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.deltaFactorBox.setKeyboardTracking(False)
        self.deltaFactorBox.setObjectName("deltaFactorBox")
        self.gridLayout.addWidget(self.deltaFactorBox, 0, 7, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(5, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem3, 0, 2, 1, 1)
        spacerItem4 = QtWidgets.QSpacerItem(35, 0, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem4, 0, 8, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)

        self.retranslateUi(CorrectionFactorInspector)
        QtCore.QMetaObject.connectSlotsByName(CorrectionFactorInspector)

    def retranslateUi(self, CorrectionFactorInspector):
        _translate = QtCore.QCoreApplication.translate
        CorrectionFactorInspector.setWindowTitle(_translate("CorrectionFactorInspector", "Dialog"))
        self.label_2.setText(_translate("CorrectionFactorInspector", "global δ-Factor:"))
        self.label.setText(_translate("CorrectionFactorInspector", "global ɑ-Factor:"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    CorrectionFactorInspector = QtWidgets.QDialog()
    ui = Ui_CorrectionFactorInspector()
    ui.setupUi(CorrectionFactorInspector)
    CorrectionFactorInspector.show()
    sys.exit(app.exec_())
