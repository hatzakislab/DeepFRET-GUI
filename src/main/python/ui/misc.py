import multiprocessing

multiprocessing.freeze_support()

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from lib.misc import timeit


class SheetInspector(QDialog):
    """
    Addon class for modal dialogs. Must be placed AFTER the actual PyQt superclass
    that the window inherits from.
    """

    def __init__(self, parent):
        super().__init__()
        self.setParent(parent)
        self.parent = parent
        self.setModal(True)
        self.setWindowFlag(Qt.Sheet)

    def setInspectorConfigs(self, params):
        """
        Writes defaults for the Ui, by passing the params tuple from parent window
        """
        for key, new_val in zip(self.keys, params):
            self.setConfig(key, new_val)

    def connectUi(self, parent):
        """
        Connect Ui to parent functions. Override in parent
        """
        pass

    def setUi(self):
        """
        Setup UI according to last saved preferences. Override in parent
        """
        pass

    def returnInspectorValues(self):
        """
        Returns values from inspector window to be used in parent window
        """
        pass


class RestartDialog(QMessageBox, SheetInspector):
    """
    Triggers a modal restart dialog if imaging mode is changed.
    Otherwise, the GUI will crash, because the plotting backend isn't set up properly.
    """

    def __init__(self, parent):
        super().__init__(parent=parent)

        self.status = None

        self.setText("Imaging mode changed")
        self.setInformativeText(
            "Restart application for new settings to take effect."
        )
        self.minimumSizeHint()

        self.setStandardButtons(QMessageBox.Yes | QMessageBox.No)

        self.buttonYes = self.button(QMessageBox.Yes)
        self.buttonYes.setText("Restart")

        self.buttonNo = self.button(QMessageBox.No)
        self.buttonNo.setText("Cancel")

        self.buttonClicked.connect(self.returnButton)

    def returnButton(self, i):
        self.status = "Restart" if i.text() == "Restart" else "Cancel"


class ExportDialog(QFileDialog):
    """
    Custom export dialog to change labels on the accept button.
    Cancel button doesn't work for whatever reason on MacOS (Qt bug?).
    """

    def __init__(self, init_dir, accept_label="Accept"):
        super().__init__()
        self.setFileMode(self.DirectoryOnly)
        self.setLabelText(self.Accept, accept_label)
        self.setDirectory(init_dir)


class ProgressBar(QProgressDialog, SheetInspector):
    """
    Displays a progressbar, using known length of a loop.
    """

    def __init__(self, parent, loop_len=0):
        super().__init__(parent=parent)
        self.minimumSizeHint()
        self.setValue(0)
        self.setMinimum(0)
        self.setMaximum(
            loop_len
        )  # Corrected because iterations start from zero, but minimum length is 1
        self.show()

    def increment(self):
        """
        Increments progress by 1
        """
        self.setValue(self.value() + 1)


class UpdatingList:
    """
    Class for dynamically updating attributes that need to be iterable,
    e.g. instead of keeping separate has_blu, has_grn, has_red checks,
    collect them in this class to make them mutable.
    """

    def __iter__(self):
        return (
            self.__getattribute__(i)
            for i in dir(self)
            if not i.startswith("__")
        )


class CheckBoxDelegate(QStyledItemDelegate):
    """
    Implement into dynamic checkboxes to check their states (see also ListView below).
    """

    def editorEvent(self, event, model, option, index):
        checked = index.data(Qt.CheckStateRole)
        ret = QStyledItemDelegate.editorEvent(self, event, model, option, index)

        if checked != index.data(Qt.CheckStateRole):
            self.parent().checked.emit(index)
        return ret


class Delegate(QStyledItemDelegate):
    """
    Triggers a return event whenever a checkbox is triggered.
    """

    def editorEvent(self, event, model, option, index):
        checked = index.data(Qt.CheckStateRole)
        ret = QStyledItemDelegate.editorEvent(self, event, model, option, index)
        if checked != index.data(Qt.CheckStateRole):
            self.parent().checked.emit(index)
        return ret


class ListView(QListView):
    """
    Custom ListView implementation which handles checkbox triggers.
    """

    checked = pyqtSignal(QModelIndex)

    def __init__(self, *args, **kwargs):
        super(ListView, self).__init__(*args, **kwargs)
        self.setItemDelegate(Delegate(self))
        self.setMaximumWidth(450)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
