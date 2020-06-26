import multiprocessing

multiprocessing.freeze_support()

from global_variables import GlobalVariables as gvars
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import (
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from PyQt5.QtWidgets import *
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


# TODO: subplots_adjust should be part of the init to clean up and make it
#  easier to control when creating the window
class PlotWidget(QWidget):
    """
    Creates a wrapper around the canvas to add the matplotlib toolbar.
    The toolbar is hidden by default, and is only used for the save dialog.

    Some windows (like MainWindow and TraceWindow) dynamically create
    the layoutbox, so as to add customized listviews. In these cases the canvas
    needs to be added manually
    """

    def __init__(self, use_layoutbox=False, **kwargs):
        QWidget.__init__(self)
        self.canvas = MatplotlibCanvas(parent=self, **kwargs)
        self.toolbar = NavigationToolbar(self.canvas, self, coordinates=True)
        self.toolbar.setVisible(False)

        if not use_layoutbox:
            self.setLayout(QVBoxLayout())
            self.layout().addWidget(self.canvas)


class MatplotlibCanvas(FigureCanvas):
    """
    This is the matplotlib plot inside that controls all the visuals.
    Subplots are placed in a grid of nrows, ncols, and a position
    e.g. (131) means 1 row, 3 columns, position 1 (going from top left).
    """

    def __init__(
        self, ax_type, parent=None, width=6, height=2, dpi=100,
    ):
        self.fig = Figure(figsize=(width, height), dpi=dpi,)
        self.fig.set_facecolor(gvars.color_gui_bg)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        self.defaultImageName = "Blank"  # Set default image name
        FigureCanvas.setSizePolicy(
            self, QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        FigureCanvas.updateGeometry(self)

        self.ax_type = ax_type

        if ax_type == "img":
            self.setupTwoColorImageLayout()
        elif ax_type == "trace":
            self.setupTwoColorTraceLayout()
        elif ax_type == "plot":
            self.setupSinglePlotLayout()
        elif ax_type == "histwin":
            self.setupHistogramWindowPlot()
        elif ax_type == "dynamic":
            self.setupDynamicPlotLayout()
        else:
            raise ValueError("Invalid ax_type")

    @staticmethod
    def adjust_margins(fig, margin=None, **kwargs):
        """
        Adjusts all margins if set to value, else individual elements
        can be controlled as usual
        """
        if type(fig) == GridSpec:
            fig.adjust = fig.update
        elif type(fig) == Figure:
            fig.adjust = fig.subplots_adjust
        else:
            raise ValueError("Invalid figure type")

        if margin is not None:
            fig.adjust(
                left=margin,
                right=1 - margin,
                top=1 - margin,
                bottom=margin,
                **kwargs
            )
        else:
            fig.adjust(**kwargs)

    @staticmethod
    def setupJointGrid(f: Figure, gs: GridSpecFromSubplotSpec):
        """
        Sets up JointGrid Setup as in Seaborn, from existing GridSpec.
        :param f: Figure to plot axes on
        :param gs: GridSpec to use. Should be square
        :return axes: Tuple of Axes, order center, top, right
        Example:
        for a 4x4 GridSpec, the three axes will be distributed as:
        tttx
        cccr
        cccr
        cccr
        where t is the top, c is center, and r is right axis
        """
        ax_ctr = f.add_subplot(gs[1:, :-1])
        ax_rgt = f.add_subplot(gs[1:, -1], sharey=ax_ctr)
        ax_top = f.add_subplot(gs[0, :-1], sharex=ax_ctr)

        return ax_ctr, ax_top, ax_rgt

    def setupHistogramWindowPlot(self):
        """
        Sets up plot for HistogramWindow using nested GridSpec.
        Outer grid is a 2x2 grid, and the inner grids are from top left, clockwise:
        6x6, 6x6, 2x1, 1x1
        """
        # outer_grid = GridSpec(nrows=1, ncols=1, figure=self.fig,)  # 2x2 grid
        # self.adjust_margins(
        #     fig=outer_grid, margin=0.10, hspace=0.10, wspace=0.10,
        # )
        #
        # gs_top_lft = GridSpecFromSubplotSpec(
        #     nrows=1, ncols=1, subplot_spec=outer_grid[0, 0], wspace=0, hspace=0,
        # )
        #
        # gs_top_rgt = GridSpecFromSubplotSpec(
        #     nrows=6, ncols=6, subplot_spec=outer_grid[0, 1], wspace=0, hspace=0,
        # )
        #
        # gs_btm_lft = GridSpecFromSubplotSpec(
        #     nrows=2,
        #     ncols=1,
        #     subplot_spec=outer_grid[1, 0],
        #     wspace=0,
        #     hspace=0.15,
        # )
        # gs_btm_rgt = GridSpecFromSubplotSpec(
        #     nrows=1, ncols=1, subplot_spec=outer_grid[1, 1], wspace=0, hspace=0,
        # )
        #
        # self.tl_ax_ctr, self.tl_ax_top, self.tl_ax_rgt = self.setupJointGrid(
        #     self.fig, gs_top_lft
        # )
        # self.tr_ax_ctr, self.tr_ax_top, self.tr_ax_rgt = self.setupJointGrid(
        #     self.fig, gs_top_rgt
        # )
        #
        # self.bl_ax_t = self.fig.add_subplot(gs_btm_lft[0])
        # self.bl_ax_b = self.fig.add_subplot(gs_btm_lft[1])
        # self.br_ax = self.fig.add_subplot(gs_btm_rgt[0])
        #
        # self.axes = (
        #     self.tl_ax_ctr,
        #     self.tl_ax_top,
        #     self.tl_ax_rgt,
        # self.tr_ax_ctr,
        # self.tr_ax_top,
        # self.tr_ax_rgt,
        # self.bl_ax_b,
        # self.bl_ax_t,
        # self.br_ax,
        # )
        # self.axes_marg = (
        #     self.tl_ax_top,
        #     self.tl_ax_rgt,
        # self.tr_ax_top,
        # self.tr_ax_rgt,
        # )
        """
        Sets up a 2D-histogram layout similar to a seaborn JointGrid,
        but manually through matplotlib for compatibility reasons.
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

    def setupTwoColorImageLayout(self):
        """
        3-view layout for dual cam microscope setup (green, red, green/red).
        """
        self.ax_grn = self.fig.add_subplot(131)  # Green
        self.ax_red = self.fig.add_subplot(132)  # Red
        self.ax_grn_red = self.fig.add_subplot(133)  # Blend

        self.axes_single = self.ax_grn, self.ax_red
        self.axes_all = self.ax_grn, self.ax_red, self.ax_grn_red

        self.adjust_margins(
            fig=self.fig, left=0.05, right=0.95, hspace=0, wspace=0.04
        )

    def setupTwoColorTraceLayout(self):
        """
        Setup for viewing traces with 2 colors.
        """
        gs = self.fig.add_gridspec(
            nrows=5, ncols=1, height_ratios=[2, 2, 2, 2, 1]
        )

        self.ax_grn = self.fig.add_subplot(gs[0])  # Green
        self.ax_acc = self.ax_grn.twinx()  # Red
        self.ax_red = self.fig.add_subplot(gs[1])  # ALEX
        self.ax_fret = self.fig.add_subplot(gs[2])  # FRET
        self.ax_stoi = self.fig.add_subplot(
            gs[3]
        )  # Stoichiometry (should be the bottom-most)
        self.ax_ml = self.fig.add_subplot(gs[4])  # ML predictions

        self.axes = (
            self.ax_grn,
            self.ax_acc,
            self.ax_red,
            self.ax_fret,
            self.ax_stoi,
            self.ax_ml,
        )
        self.axes_c = list(
            zip((self.ax_grn, self.ax_acc, self.ax_red), ("D", "A", "A-direct"))
        )

        self.adjust_margins(
            fig=self.fig, hspace=0, left=0.06, right=0.94, top=0.96, bottom=0.04
        )

        self.traceOutlineColor()

    def setupDynamicPlotLayout(self):
        """
        Setup for a single panel plot.
        """
        self.axes = (self.fig.add_subplot(111, aspect="equal"),)
        for ax in self.axes:
            ax.set_xticks(())
            ax.set_yticks(())

        self.adjust_margins(fig=self.fig, margin=0.02)

    def setupSinglePlotLayout(self):
        """
        Setup for a single panel plot.
        """
        self.ax = self.fig.add_subplot(111, aspect="equal")
        self.adjust_margins(fig=self.fig, margin=0.02)

    def setupDoubleAxesPlotLayout(self):
        """
        Setup for two plots on top of each other
        """
        self.ax_top = self.fig.add_subplot(211)
        self.ax_btm = self.fig.add_subplot(212)

        self.axes = self.ax_top, self.ax_btm

        self.adjust_margins(
            fig=self.fig,
            hspace=0.30,
            wspace=0,
            left=0.02,
            right=0.98,
            bottom=0.09,
            top=0.98,
        )

    def setupTripleAxesPlotLayout(self):
        """
        Horizontal layout for three plots at a time
        """
        self.ax_lft = self.fig.add_subplot(131)
        self.ax_ctr = self.fig.add_subplot(132)
        self.ax_rgt = self.fig.add_subplot(133)

        self.adjust_margins(
            fig=self.fig,
            hspace=0.30,
            wspace=0,
            left=0.02,
            right=0.98,
            bottom=0.09,
            top=0.98,
        )

    def traceOutlineColor(self):
        """
        Updates the box outline and ticks for the traces displayer.
        """
        for ax in self.axes:
            ax.tick_params(
                axis="x", which="both", bottom=False, labelbottom=False
            )
            ax.yaxis.label.set_color(gvars.color_gui_text)

        if hasattr(self, "ax_ml"):
            self.ax_ml.tick_params(
                axis="x", which="both", bottom=True, labelbottom=True
            )
            self.ax_ml.set_xlabel("Frames")  # Assuming this is the bottom trace
        else:
            if self.ax_setup != "bypass":
                self.ax_stoi.tick_params(
                    axis="x", which="both", bottom=True, labelbottom=True
                )
                self.ax_stoi.set_xlabel(
                    "Frames"
                )  # Assuming this is the bottom trace

    def get_window_title(self):  # overwrite class method for default image name
        return self.defaultImageName
