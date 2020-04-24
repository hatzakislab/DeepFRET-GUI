from global_variables import GlobalVariables as gvars
from ui._DensityWindowInspector import Ui_DensityWindowInspector
from widgets.misc import SheetInspector
from widgets.histogram_window import HistogramWindow
from widgets.transition_density_window import TransitionDensityWindow


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

        if isinstance(
            parent, HistogramWindow
        ):  # Avoids an explicit reference in parent class, for easier copy-paste
            self.keys = gvars.keys_hist
            parent.inspector = self
        elif isinstance(parent, TransitionDensityWindow):
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

                # Avoids an explicit reference in parent class,
                # for easier copy-paste
                if isinstance(parent, HistogramWindow) or isinstance(
                    parent, TransitionDensityWindow
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
