from functools import partial
from typing import Dict

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QWidget

from global_variables import GlobalVariables as gvars
from ui._CorrectionFactorInspector import Ui_CorrectionFactorInspector
from ui._DensityWindowInspector import Ui_DensityWindowInspector
from ui._TraceWindowInspector import Ui_TraceWindowInspector


class SheetInspector(QDialog):
    """
    Addon class for modal dialogs. Must be placed AFTER the actual PyQt
    superclass that the window inherits from.
    """

    def __init__(self, parent: QWidget):
        super().__init__()
        self.setParent(parent)
        self.windows = parent.windows
        self.parent = parent
        self.parent_name = parent.__class__.__name__
        self.getConfig = parent.getConfig
        self.setConfig = parent.setConfig
        self.addSelfToParentInspectorTracking(parent)

        # All parent window references
        self.trace_window = self.windows[gvars.TraceWindow]
        self.histogram_window = self.windows[gvars.HistogramWindow]
        self.transition_density_window = self.windows[
            gvars.TransitionDensityWindow
        ]
        self.video_window = self.windows[gvars.VideoWindow]

        self.setModal(True)
        self.setWindowFlag(Qt.Sheet)

    def addSelfToParentInspectorTracking(self, parent):
        """
        Adds current instance to parent, so each parent window can track their
        assigned inspectors
        """
        parent.inspectors[self.__class__.__name__] = self

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


class DensityWindowInspector(SheetInspector):
    """
    Window for managing slider sheets for windows that use density plots. Use similar template for other sliders.
    See also refreshPlot for usage, because it needs a try/except clause to function properly.
    """

    def __init__(self, parent):
        super().__init__(parent=parent)
        self.getConfig = parent.getConfig
        self.setConfig = parent.setConfig
        self.ui = Ui_DensityWindowInspector()
        self.ui.setupUi(self)

        if self.parent_name == gvars.HistogramWindow:
            self.keys = gvars.keys_hist
            parent.inspector = self
        elif self.parent_name == gvars.TransitionDensityWindow:
            self.keys = gvars.keys_tdp
            parent.inspector = self
        else:
            raise NotImplementedError

        self.setUi()
        self.connectUi(parent)

    def connectUi(self, parent):
        """Connect Ui to parent functions"""
        if hasattr(
            parent, "canvas"
        ):  # Avoid refreshing canvas before it's even instantiated on the parent
            for slider in (
                self.ui.smoothingSlider,
                self.ui.resolutionSlider,
                self.ui.colorSlider,
                self.ui.pointAlphaSlider,
            ):
                if self.parent_name in (
                    gvars.HistogramWindow,
                    gvars.TransitionDensityWindow,
                ):
                    slider.valueChanged.connect(parent.refreshPlot)
                else:
                    raise NotImplementedError

            self.ui.overlayCheckBox.clicked.connect(parent.refreshPlot)

    def setUi(self):
        """
        Setup UI according to last saved preferences.
        """
        bandwidth, resolution, n_colors, overlay_pts, pts_alpha = [
            self.getConfig(key) for key in self.keys
        ]

        self.ui.smoothingSlider.setValue(bandwidth)
        self.ui.resolutionSlider.setValue(resolution)
        self.ui.colorSlider.setValue(n_colors)
        self.ui.overlayCheckBox.setChecked(bool(overlay_pts))
        self.ui.pointAlphaSlider.setValue(pts_alpha)

    def returnInspectorValues(self):
        """
        Returns values from inspector window to be used in parent window.
        """
        bandwidth = self.ui.smoothingSlider.value()
        resolution = self.ui.resolutionSlider.value()
        n_colors = self.ui.colorSlider.value()
        overlay_pts = self.ui.overlayCheckBox.isChecked()
        pts_alpha = self.ui.pointAlphaSlider.value()

        return bandwidth, resolution, n_colors, overlay_pts, pts_alpha


class CorrectionFactorInspector(SheetInspector):
    def __init__(self, parent):
        super().__init__(parent=parent)
        self.ui = Ui_CorrectionFactorInspector()
        self.ui.setupUi(self)

        self.alphaFactor = self.getConfig(gvars.key_alphaFactor)
        self.deltaFactor = self.getConfig(gvars.key_deltaFactor)

        self.ui.alphaFactorBox.setValue(self.alphaFactor)
        self.ui.deltaFactorBox.setValue(self.deltaFactor)

        self.ui.alphaFactorBox.valueChanged.connect(
            partial(self.setCorrectionFactors, "alpha")
        )
        self.ui.deltaFactorBox.valueChanged.connect(
            partial(self.setCorrectionFactors, "delta")
        )

    def setCorrectionFactors(self, factor):
        """
        Sets the global correction factors
        """
        trace_window = self.windows[gvars.TraceWindow]
        histogram_window = self.windows[gvars.TraceWindow]

        parent = self.parent
        self.alphaFactor = self.ui.alphaFactorBox.value()
        self.deltaFactor = self.ui.deltaFactorBox.value()

        if factor == "alpha":
            parent.setConfig(gvars.key_alphaFactor, self.alphaFactor)
        elif factor == "delta":
            parent.setConfig(gvars.key_deltaFactor, self.deltaFactor)

        if trace_window.isVisible():
            trace_window.refreshPlot()

        if histogram_window.isVisible():
            histogram_window.refreshPlot()

    def showEvent(self, event):
        self.ui.alphaFactorBox.setValue(self.alphaFactor)
        self.ui.deltaFactorBox.setValue(self.deltaFactor)
        self.ui.alphaFactorBox.repaint()
        self.ui.deltaFactorBox.repaint()


class AdvancedSortInspector(SheetInspector):
    """
    Inspector for the advanced sorting sheet.
    """

    def __init__(self, parent):
        super().__init__(parent=parent)

        self.ui = Ui_TraceWindowInspector()
        self.ui.setupUi(self)
        self.keys = gvars.keys_trace

        self.spinBoxes = (
            self.ui.spinBoxStoiLo,
            self.ui.spinBoxStoiHi,
            self.ui.spinBoxFretLo,
            self.ui.spinBoxFretHi,
            self.ui.spinBoxMinFrames,
            self.ui.spinBoxConfidence,
            self.ui.spinBoxDynamics,
        )

        self.connectUi(parent)
        self.setUi()

    def setUi(self):
        """Setup UI according to last saved preferences."""
        for spinBox, key in zip(self.spinBoxes, self.keys):
            spinBox.setValue(self.getConfig(key))

    def connectUi(self, parent):
        """Connect Ui to parent functions."""
        self.ui.pushButtonFind.clicked.connect(self.findPushed)

    def findPushed(self):
        """Sorts list by conditions set by spinBoxes and closes."""
        self.trace_window.sortListByCondition(setup="advanced")
        if self.histogram_window.isVisible():
            self.histogram_window.refreshPlot(True)
        self.close()

    def returnInspectorValues(self):
        """Returns inspector values to parent window"""
        (
            S_med_lo,
            S_med_hi,
            E_med_lo,
            E_med_hi,
            min_n_frames,
            confidence,
            dynamics,
        ) = [spinBox.value() for spinBox in self.spinBoxes]

        bleached_only = self.ui.checkBoxBleach.isChecked()
        return (
            S_med_lo,
            S_med_hi,
            E_med_lo,
            E_med_hi,
            min_n_frames,
            confidence,
            dynamics,
            bleached_only,
        )
