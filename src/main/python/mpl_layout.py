import multiprocessing
multiprocessing.freeze_support()

from global_variables import GlobalVariables as gvars
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path
from matplotlib.figure import Figure
from PyQt5.QtWidgets import *
import numpy as np
import pandas as pd
from lib.misc import timeit
from matplotlib.gridspec import GridSpec


class MatplotlibCanvas(FigureCanvas):
    """
    This is the matplotlib plot inside that controls all the visuals.
    Subplots are placed in a grid of nrows, ncols, and a position
    e.g. (131) means 1 row, 3 columns, position 1 (going from top left).
    """

    def __init__(self, parent=None, ax_setup=None, ax_window=None, width=6, height=2, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=False)
        super(MatplotlibCanvas, self).__init__(self.fig)
        # FigureCanvas.__init__(self, self.fig)

        self.setParent(parent)
        self.defaultImageName = "Blank"  # Set default image name
        self.get_window_title()

        self.ax_setup = ax_setup
        self.ax_window = ax_window

        if ax_setup == "dual":
            if ax_window == "img":
                self.setupTwoColorImageLayout()
            elif ax_window == "trace":
                self.setupTwoColorTraceLayout()
            else:
                raise ValueError

        if ax_setup == "2-color":
            if ax_window == "img":
                self.setupTwoColorImageLayout()
            elif ax_window == "trace":
                self.setupTwoColorTraceLayout()
            else:
                raise ValueError

        if ax_setup == "2-color-inv":
            if ax_window == "img":
                self.setupTwoColorImageLayout()
            elif ax_window == "trace":
                self.setupTwoColorTraceLayout()
            else:
                raise ValueError

        if ax_setup == "plot":
            if ax_window == "jointgrid":
                self.setupJointGridLayout()
            elif ax_window == "correction":
                self.setupDoubleAxesPlotLayout()
            elif ax_window == "single":
                self.setupSinglePlotLayout()
            elif ax_window == "dynamic":
                self.setupDynamicGridLayout()
            else:
                raise ValueError

    def setupTwoColorImageLayout(self):
        """
        3-view layout for dual cam microscope setup (green, red, green/red).
        """
        self.ax_grn = self.fig.add_subplot(131)  # Green
        self.ax_red = self.fig.add_subplot(132)  # Red
        self.ax_grn_red = self.fig.add_subplot(133)  # Blend

        self.axes_single = self.ax_grn, self.ax_red
        self.axes_blend = (self.ax_grn_red,)
        self.axes_all = self.ax_grn, self.ax_red, self.ax_grn_red

        self.fig.subplots_adjust(left=0.05, right=0.95, hspace = 0, wspace = 0.04)  # Increase to add space between plot and GUI

    def setupTwoColorTraceLayout(self):
        """
        Setup for viewing traces with 2 colors.
        """
        gs = self.fig.add_gridspec(
            nrows = 5, ncols = 1, height_ratios = [2, 2, 2, 2, 1]
        )

        self.ax_grn = self.fig.add_subplot(gs[0])  # Green
        self.ax_red = self.ax_grn.twinx()  # Red
        self.ax_alx = self.fig.add_subplot(gs[1])  # ALEX
        self.ax_fret = self.fig.add_subplot(gs[2])  # FRET
        self.ax_stoi = self.fig.add_subplot(gs[3])  # Stoichiometry (should be the bottom-most)
        self.ax_ml = self.fig.add_subplot(gs[4])  # ML predictions

        self.axes = self.ax_grn, self.ax_red, self.ax_alx, self.ax_fret, self.ax_stoi, self.ax_ml
        self.axes_c = list(zip((self.ax_grn, self.ax_red, self.ax_alx), ("D", "A", "A-direct")))

        self.fig.subplots_adjust(hspace=0, left=0.06, right=0.94, top=0.96, bottom=0.04)
        self.traceOutlineColor()


    def setupDynamicGridLayout(self):
        """
        Sets up a dynamic grid with post-adjustable number of subplots
        """
        self.gs = GridSpec(1, 1)
        self.fig.add_subplot(self.gs[0])
        self.axes = self.figure.axes

        m = 0.05
        self.fig.subplots_adjust(hspace=0, left=m, right=1-m, top=1-m, bottom=m)

    def setupSinglePlotLayout(self):
        """
        Setup for a single panel plot.
        """
        self.ax = self.fig.add_subplot(111, aspect="equal")
        self.fig.subplots_adjust(
            left=0.08, right=0.92, top=1.00, bottom=0.08
        )  # Increase to add space between plot and GUI

    def setupDoubleAxesPlotLayout(self):
        """
        Setup for two plots on top of each other
        """
        self.ax_top = self.fig.add_subplot(211)
        self.ax_btm = self.fig.add_subplot(212)

        self.axes = self.ax_top, self.ax_btm

        self.fig.subplots_adjust(hspace=0.30, wspace=0, left=0.02, right=0.98, bottom=0.09, top=0.98)

    def setupTripleAxesPlotLayout(self):
        """
        Horizontal layout for three plots at a time
        """
        self.ax_lft = self.fig.add_subplot(131)
        self.ax_ctr = self.fig.add_subplot(132)
        self.ax_rgt = self.fig.add_subplot(133)

        self.fig.subplots_adjust(hspace=0.30, wspace=0, left=0.02, right=0.98, bottom=0.09, top=0.98)

    def setupJointGridLayout(self):
        """
        Sets up a 2D-histogram layout similar to a seaborn JointGrid, but manually through matplotlib for compatibility reasons.
        """
        space_between = 0  # 0.01
        left, right = 0.08, 0.7
        bottom, height = 0.08, 0.7
        bottom_h = left_h = left + right + space_between

        rect_center = left, bottom, right, height
        rect_hist_top = left, bottom_h, right, 0.2
        rect_hist_right = left_h, bottom, 0.2, height

        self.ax_ctr = self.fig.add_axes(rect_center)
        self.ax_top = self.fig.add_axes(rect_hist_top)
        self.ax_rgt = self.fig.add_axes(rect_hist_right)

        self.axes = self.ax_ctr, self.ax_top, self.ax_rgt
        self.axes_marg = self.ax_top, self.ax_rgt

    def traceOutlineColor(self):
        """
        Updates the box outline and ticks for the traces displayer.
        """
        for ax in self.axes:
            ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
            ax.yaxis.label.set_color(gvars.color_gui_text)

        if hasattr(self, "ax_ml"):
            self.ax_ml.tick_params(axis = "x", which = "both", bottom = True, labelbottom = True)
            self.ax_ml.set_xlabel("Frames")  # Assuming this is the bottom trace
        else:
            if self.ax_setup != "bypass":
                self.ax_stoi.tick_params(axis="x", which="both", bottom=True, labelbottom=True)
                self.ax_stoi.set_xlabel("Frames")  # Assuming this is the bottom trace

    def get_window_title(self):  # overwrite class method for default image name
        return self.defaultImageName


class PlotWidget(QWidget):
    """
    Creates a wrapper around the canvas to add the matplotlib toolbar.
    The toolbar is hidden by default, and is only used for the save dialog.
    """

    def __init__(self, **kwargs):
        QWidget.__init__(self)
        self.setLayout(QVBoxLayout())
        self.canvas = MatplotlibCanvas(parent=self, **kwargs)

        # Visible set to False. Only using this to use the save file dialog from MPL
        self.toolbar = NavigationToolbar(self.canvas, self, coordinates=True)
        self.toolbar.setVisible(False)

        self.layout().addWidget(self.toolbar)
        self.layout().addWidget(self.canvas)


class PolygonSelection:
    """
    Creates a selection tool for plotted points
    """

    def __init__(self, ax, collection, z, parent):
        self.parent = parent
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.xy = collection.get_offsets()
        self.z = z
        self.idx = []

        self.poly = PolygonSelector(
            ax=ax,
            onselect=self.onselect,
            lineprops=dict(color="lightblue", linestyle="--"),
            markerprops=dict(alpha=0, markersize=4),
            useblit=True,
        )

    def onselect(self, vertices):
        path = Path(vertices)
        self.idx = np.nonzero(path.contains_points(self.xy))[0]

        x = self.xy[self.idx][:, 0]
        y = self.xy[self.idx][:, 1]
        z = self.z[self.idx]

        self.parent.selected_data = pd.DataFrame(dict(x=x, y=y, z=z))

        self.poly.disconnect_events()
        self.parent.refreshPlot()
