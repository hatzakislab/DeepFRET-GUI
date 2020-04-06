# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/PreferencesWindow.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Preferences(object):
    def setupUi(self, Preferences):
        Preferences.setObjectName("Preferences")
        Preferences.resize(491, 748)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            Preferences.sizePolicy().hasHeightForWidth()
        )
        Preferences.setSizePolicy(sizePolicy)
        Preferences.setMinimumSize(QtCore.QSize(0, 0))
        Preferences.setWindowTitle("Preferences")
        Preferences.setStatusTip("")
        Preferences.setWhatsThis("")
        Preferences.setAccessibleName("")
        self.verticalLayout = QtWidgets.QVBoxLayout(Preferences)
        self.verticalLayout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.verticalLayout.setObjectName("verticalLayout")
        self.generalGroup = QtWidgets.QGroupBox(Preferences)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.generalGroup.setFont(font)
        self.generalGroup.setFlat(False)
        self.generalGroup.setCheckable(False)
        self.generalGroup.setObjectName("generalGroup")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.generalGroup)
        self.gridLayout_2.setContentsMargins(-1, 12, -1, -1)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.checkBox_unColocRed = QtWidgets.QCheckBox(self.generalGroup)
        self.checkBox_unColocRed.setObjectName("checkBox_unColocRed")
        self.gridLayout_2.addWidget(self.checkBox_unColocRed, 7, 0, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.generalGroup)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.gridLayout_2.addWidget(self.label_5, 12, 0, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.generalGroup)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.gridLayout_2.addWidget(self.label_4, 3, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.generalGroup)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.gridLayout_2.addWidget(self.label_2, 10, 0, 1, 1)
        self.checkBox_batchLoadingMode = QtWidgets.QCheckBox(self.generalGroup)
        self.checkBox_batchLoadingMode.setObjectName(
            "checkBox_batchLoadingMode"
        )
        self.gridLayout_2.addWidget(self.checkBox_batchLoadingMode, 1, 0, 1, 1)
        self.label = QtWidgets.QLabel(self.generalGroup)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.gridLayout_2.addWidget(self.label, 8, 0, 1, 1)
        self.checkBox_fitSpots = QtWidgets.QCheckBox(self.generalGroup)
        self.checkBox_fitSpots.setObjectName("checkBox_fitSpots")
        self.gridLayout_2.addWidget(self.checkBox_fitSpots, 11, 0, 1, 1)
        self.checkBox_illuCorrect = QtWidgets.QCheckBox(self.generalGroup)
        self.checkBox_illuCorrect.setObjectName("checkBox_illuCorrect")
        self.gridLayout_2.addWidget(self.checkBox_illuCorrect, 9, 0, 1, 1)
        self.verticalLayout.addWidget(self.generalGroup)
        self.imagingGroup = QtWidgets.QGroupBox(Preferences)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.imagingGroup.setFont(font)
        self.imagingGroup.setAutoFillBackground(False)
        self.imagingGroup.setFlat(False)
        self.imagingGroup.setObjectName("imagingGroup")
        self.gridLayout = QtWidgets.QGridLayout(self.imagingGroup)
        self.gridLayout.setSizeConstraint(
            QtWidgets.QLayout.SetDefaultConstraint
        )
        self.gridLayout.setContentsMargins(-1, 12, -1, -1)
        self.gridLayout.setObjectName("gridLayout")
        self.radioButton_dual = QtWidgets.QRadioButton(self.imagingGroup)
        self.radioButton_dual.setObjectName("radioButton_dual")
        self.gridLayout.addWidget(self.radioButton_dual, 3, 0, 1, 1)
        self.radioButton_2_col = QtWidgets.QRadioButton(self.imagingGroup)
        self.radioButton_2_col.setObjectName("radioButton_2_col")
        self.gridLayout.addWidget(self.radioButton_2_col, 4, 0, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.imagingGroup)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.gridLayout.addWidget(self.label_7, 2, 0, 1, 1)
        self.radioButton_2_col_inv = QtWidgets.QRadioButton(self.imagingGroup)
        self.radioButton_2_col_inv.setObjectName("radioButton_2_col_inv")
        self.gridLayout.addWidget(self.radioButton_2_col_inv, 5, 0, 1, 1)
        self.checkBox_donorLeft = QtWidgets.QCheckBox(self.imagingGroup)
        self.checkBox_donorLeft.setObjectName("checkBox_donorLeft")
        self.gridLayout.addWidget(self.checkBox_donorLeft, 1, 0, 1, 1)
        self.checkBox_firstFrameIsDonor = QtWidgets.QCheckBox(self.imagingGroup)
        self.checkBox_firstFrameIsDonor.setObjectName(
            "checkBox_firstFrameIsDonor"
        )
        self.gridLayout.addWidget(self.checkBox_firstFrameIsDonor, 0, 0, 1, 1)
        self.verticalLayout.addWidget(self.imagingGroup)
        self.hmmGroup = QtWidgets.QGroupBox(Preferences)
        self.hmmGroup.setObjectName("hmmGroup")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.hmmGroup)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.radioButton_hmm_fitE = QtWidgets.QRadioButton(self.hmmGroup)
        self.radioButton_hmm_fitE.setChecked(True)
        self.radioButton_hmm_fitE.setObjectName("radioButton_hmm_fitE")
        self.gridLayout_4.addWidget(self.radioButton_hmm_fitE, 0, 0, 1, 1)
        self.radioButton_hmm_fitDD = QtWidgets.QRadioButton(self.hmmGroup)
        self.radioButton_hmm_fitDD.setObjectName("radioButton_hmm_fitDD")
        self.gridLayout_4.addWidget(self.radioButton_hmm_fitDD, 1, 0, 1, 1)
        self.doubleSpinBox_hmm_BIC = QtWidgets.QDoubleSpinBox(self.hmmGroup)
        self.doubleSpinBox_hmm_BIC.setMinimum(0.5)
        self.doubleSpinBox_hmm_BIC.setMaximum(5.0)
        self.doubleSpinBox_hmm_BIC.setSingleStep(0.5)
        self.doubleSpinBox_hmm_BIC.setProperty("value", 2.0)
        self.doubleSpinBox_hmm_BIC.setObjectName("doubleSpinBox_hmm_BIC")
        self.gridLayout_4.addWidget(self.doubleSpinBox_hmm_BIC, 5, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(
            40,
            20,
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Minimum,
        )
        self.gridLayout_4.addItem(spacerItem, 5, 1, 1, 1)
        self.checkBox_hmm_local = QtWidgets.QCheckBox(self.hmmGroup)
        self.checkBox_hmm_local.setObjectName("checkBox_hmm_local")
        self.gridLayout_4.addWidget(self.checkBox_hmm_local, 6, 0, 1, 1)
        self.label_hmm_bic = QtWidgets.QLabel(self.hmmGroup)
        self.label_hmm_bic.setObjectName("label_hmm_bic")
        self.gridLayout_4.addWidget(self.label_hmm_bic, 4, 0, 1, 1)
        self.label_hmm_checkbox = QtWidgets.QLabel(self.hmmGroup)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_hmm_checkbox.setFont(font)
        self.label_hmm_checkbox.setObjectName("label_hmm_checkbox")
        self.gridLayout_4.addWidget(self.label_hmm_checkbox, 7, 0, 1, 1)
        self.verticalLayout.addWidget(self.hmmGroup)
        self.groupBox = QtWidgets.QGroupBox(Preferences)
        self.groupBox.setFlat(True)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.toleranceComboBox = QtWidgets.QComboBox(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.toleranceComboBox.sizePolicy().hasHeightForWidth()
        )
        self.toleranceComboBox.setSizePolicy(sizePolicy)
        self.toleranceComboBox.setMaximumSize(QtCore.QSize(125, 16777215))
        self.toleranceComboBox.setSizeAdjustPolicy(
            QtWidgets.QComboBox.AdjustToContents
        )
        self.toleranceComboBox.setObjectName("toleranceComboBox")
        self.toleranceComboBox.addItem("")
        self.toleranceComboBox.addItem("")
        self.toleranceComboBox.addItem("")
        self.gridLayout_3.addWidget(self.toleranceComboBox, 0, 0, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(
            40,
            20,
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Minimum,
        )
        self.gridLayout_3.addItem(spacerItem1, 0, 1, 1, 1)
        self.verticalLayout.addWidget(self.groupBox)
        self.groupBox_2 = QtWidgets.QGroupBox(Preferences)
        self.groupBox_2.setFlat(True)
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.spinBox_autoDetect = QtWidgets.QSpinBox(self.groupBox_2)
        self.spinBox_autoDetect.setMaximum(999)
        self.spinBox_autoDetect.setObjectName("spinBox_autoDetect")
        self.gridLayout_5.addWidget(self.spinBox_autoDetect, 0, 0, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(
            40,
            20,
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Minimum,
        )
        self.gridLayout_5.addItem(spacerItem2, 0, 2, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.groupBox_2)
        self.label_6.setObjectName("label_6")
        self.gridLayout_5.addWidget(self.label_6, 0, 1, 1, 1)
        self.verticalLayout.addWidget(self.groupBox_2)

        self.retranslateUi(Preferences)
        QtCore.QMetaObject.connectSlotsByName(Preferences)
        Preferences.setTabOrder(
            self.checkBox_batchLoadingMode, self.checkBox_unColocRed
        )
        Preferences.setTabOrder(
            self.checkBox_unColocRed, self.checkBox_illuCorrect
        )
        Preferences.setTabOrder(
            self.checkBox_illuCorrect, self.checkBox_fitSpots
        )
        Preferences.setTabOrder(
            self.checkBox_fitSpots, self.checkBox_firstFrameIsDonor
        )
        Preferences.setTabOrder(
            self.checkBox_firstFrameIsDonor, self.checkBox_donorLeft
        )
        Preferences.setTabOrder(self.checkBox_donorLeft, self.radioButton_dual)
        Preferences.setTabOrder(self.radioButton_dual, self.radioButton_2_col)
        Preferences.setTabOrder(
            self.radioButton_2_col, self.radioButton_2_col_inv
        )
        Preferences.setTabOrder(
            self.radioButton_2_col_inv, self.radioButton_hmm_fitE
        )
        Preferences.setTabOrder(
            self.radioButton_hmm_fitE, self.radioButton_hmm_fitDD
        )
        Preferences.setTabOrder(
            self.radioButton_hmm_fitDD, self.doubleSpinBox_hmm_BIC
        )
        Preferences.setTabOrder(
            self.doubleSpinBox_hmm_BIC, self.checkBox_hmm_local
        )
        Preferences.setTabOrder(self.checkBox_hmm_local, self.toleranceComboBox)
        Preferences.setTabOrder(self.toleranceComboBox, self.spinBox_autoDetect)

    def retranslateUi(self, Preferences):
        _translate = QtCore.QCoreApplication.translate
        self.generalGroup.setTitle(_translate("Preferences", "General"))
        self.checkBox_unColocRed.setText(
            _translate("Preferences", "Detect uncolocalized red")
        )
        self.label_5.setText(
            _translate(
                "Preferences", "      Leave off for local maxima detection"
            )
        )
        self.label_4.setText(
            _translate(
                "Preferences",
                "      Disables interactivity to avoid consuming memory",
            )
        )
        self.label_2.setText(
            _translate(
                "Preferences",
                "      Improves spot detection but may increase video loading time",
            )
        )
        self.checkBox_batchLoadingMode.setText(
            _translate("Preferences", "Batch loading mode")
        )
        self.label.setText(
            _translate(
                "Preferences", "      For determining acceptor bleedthrough"
            )
        )
        self.checkBox_fitSpots.setText(
            _translate(
                "Preferences",
                "Detect spots using Laplacian of Gaussian fitting",
            )
        )
        self.checkBox_illuCorrect.setText(
            _translate("Preferences", "Correct for illumination background")
        )
        self.imagingGroup.setTitle(
            _translate("Preferences", "Imaging Setup (restart required!)")
        )
        self.radioButton_dual.setText(
            _translate("Preferences", "Interleaved Video")
        )
        self.radioButton_2_col.setText(
            _translate("Preferences", "Quad View (2-channel)")
        )
        self.label_7.setText(
            _translate(
                "Preferences", "      Assuming left/right chip allocation"
            )
        )
        self.radioButton_2_col_inv.setText(
            _translate("Preferences", "Quad View (inverted 2-channel)")
        )
        self.checkBox_donorLeft.setText(
            _translate("Preferences", "Donor is left side")
        )
        self.checkBox_firstFrameIsDonor.setText(
            _translate("Preferences", "First frame is donor excitation")
        )
        self.hmmGroup.setTitle(_translate("Preferences", "HMM Settings"))
        self.radioButton_hmm_fitE.setText(
            _translate("Preferences", "E_FRET Fitting")
        )
        self.radioButton_hmm_fitDD.setText(
            _translate("Preferences", "DD/DA Fitting")
        )
        self.checkBox_hmm_local.setText(
            _translate(
                "Preferences",
                "Idealize HMM for each trace individually (Advanced)",
            )
        )
        self.label_hmm_bic.setText(
            _translate(
                "Preferences",
                "BIC Strictness (raise this to detect fewer states)",
            )
        )
        self.label_hmm_checkbox.setText(
            _translate(
                "Preferences",
                "     N.B. This compromises the statistical assumptions of the Hidden Markov Model \n"
                "    and could lead to bad results! Try with more states instead!",
            )
        )
        self.groupBox.setTitle(
            _translate("Preferences", "Colocalization Tolerance")
        )
        self.toleranceComboBox.setItemText(
            0, _translate("Preferences", "Loose")
        )
        self.toleranceComboBox.setItemText(
            1, _translate("Preferences", "Moderate")
        )
        self.toleranceComboBox.setItemText(
            2, _translate("Preferences", "Strict")
        )
        self.groupBox_2.setTitle(
            _translate("Preferences", "Default Detection Per Movie")
        )
        self.label_6.setText(_translate("Preferences", "Pairs"))


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    Preferences = QtWidgets.QWidget()
    ui = Ui_Preferences()
    ui.setupUi(Preferences)
    Preferences.show()
    sys.exit(app.exec_())
