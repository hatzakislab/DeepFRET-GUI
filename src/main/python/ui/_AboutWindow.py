# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/_AboutWindow.ui'
#
# Created by: PyQt5 UI code generator 5.13.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_About(object):
    def setupUi(self, About):
        About.setObjectName("About")
        About.resize(300, 200)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(About.sizePolicy().hasHeightForWidth())
        About.setSizePolicy(sizePolicy)
        About.setMinimumSize(QtCore.QSize(280, 150))
        About.setMaximumSize(QtCore.QSize(300, 206))
        About.setBaseSize(QtCore.QSize(0, 0))
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(About)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_APPNAME = QtWidgets.QLabel(About)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_APPNAME.setFont(font)
        self.label_APPNAME.setTextFormat(QtCore.Qt.PlainText)
        self.label_APPNAME.setAlignment(QtCore.Qt.AlignCenter)
        self.label_APPNAME.setWordWrap(True)
        self.label_APPNAME.setObjectName("label_APPNAME")
        self.verticalLayout.addWidget(self.label_APPNAME)
        self.label_APPVER = QtWidgets.QLabel(About)
        self.label_APPVER.setAlignment(QtCore.Qt.AlignCenter)
        self.label_APPVER.setWordWrap(True)
        self.label_APPVER.setObjectName("label_APPVER")
        self.verticalLayout.addWidget(self.label_APPVER)
        self.label_AUTHORS = QtWidgets.QLabel(About)
        self.label_AUTHORS.setAlignment(QtCore.Qt.AlignCenter)
        self.label_AUTHORS.setWordWrap(True)
        self.label_AUTHORS.setObjectName("label_AUTHORS")
        self.verticalLayout.addWidget(self.label_AUTHORS)
        self.label_LICENSE = QtWidgets.QLabel(About)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_LICENSE.sizePolicy().hasHeightForWidth())
        self.label_LICENSE.setSizePolicy(sizePolicy)
        self.label_LICENSE.setMinimumSize(QtCore.QSize(0, 100))
        self.label_LICENSE.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
        self.label_LICENSE.setWordWrap(True)
        self.label_LICENSE.setObjectName("label_LICENSE")
        self.verticalLayout.addWidget(self.label_LICENSE)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.verticalLayout_2.addLayout(self.verticalLayout)

        self.retranslateUi(About)
        QtCore.QMetaObject.connectSlotsByName(About)

    def retranslateUi(self, About):
        _translate = QtCore.QCoreApplication.translate
        About.setWindowTitle(_translate("About", "Form"))
        self.label_APPNAME.setText(_translate("About", "APPNAME"))
        self.label_APPVER.setText(_translate("About", "APPVERSION"))
        self.label_AUTHORS.setText(_translate("About", "AUTHORS"))
        self.label_LICENSE.setText(_translate("About", "LICENSE INFO"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    About = QtWidgets.QWidget()
    ui = Ui_About()
    ui.setupUi(About)
    About.show()
    sys.exit(app.exec_())
