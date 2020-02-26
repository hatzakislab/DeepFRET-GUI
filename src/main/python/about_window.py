from PyQt5.QtWidgets import QDialog
from global_variables import GlobalVariables as gvars
from ui._AboutWindow import Ui_About


class AboutWindow(QDialog):
    """
    'About this application' window with version numbering.
    """

    def __init__(self):
        super().__init__()
        self.ui = Ui_About()
        self.ui.setupUi(self)
        self.setWindowTitle("")

        self.ui.label_APPNAME.setText(gvars.APPNAME)
        self.ui.label_APPVER.setText("version {}".format(gvars.APPVERSION))
        self.ui.label_AUTHORS.setText(gvars.AUTHORS)
        self.ui.label_LICENSE.setText(gvars.LICENSE)