import sys
from main import AppContext
from widgets.inspectors import DensityWindowInspector
from widgets.histogram_window import HistogramWindow
import pandas as pd
import numpy as np


class setUp(HistogramWindow):
    def __init__(self):
        super(setUp, self).__init__()

    def setFile(self, path):
        df = pd.read_csv(path, skiprows=5, sep="\t")
        self.E = df["E"].values
        self.S = df["S"].values
        self.DA = df["A-Dexc-rw"]
        self.DD = df["D-Dexc-rw"]
        self.trace_median_len = len(df)
        self.lengths = np.random.randint(
            1, 100, 100
        )  # TODO: plot lifetimes in a modular manner

    def testPlot(self, **kwargs):
        self.setFile(**kwargs)
        self.plotAll(corrected=True)
        self.show()


if __name__ == "__main__":
    ctxt = AppContext()
    ctxt.load_resources()

    cls = setUp()
    cls.testPlot(path="../resources/traces/TraceLong3State.txt")

    exit_code = ctxt.run()
    sys.exit(exit_code)
