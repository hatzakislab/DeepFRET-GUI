import sys

from main import AppContext, TraceContainer, TraceWindow


class SetUp(TraceWindow):
    class Data:
        traces = {}

    def __init__(self):
        TraceWindow.data = self.Data()
        super(SetUp, self).__init__()

        self.setFile()

    def setFile(self):
        trace = TraceContainer(
            filename="../resources/traces/simulated_trace.txt",
            loaded_from_ascii=True,
        )
        self.data.traces[trace.name] = trace
        self.currName = trace.name
        self._currName = trace.name
        self.refreshPlot()

    def predict(self):
        self.classifyTraces()  # resets self.currentName
        self.currName = self._currName
        self.refreshPlot()

    def plot(self):
        for ax in self.canvas.axes:
            ax.clear()

        trace = self.currentTrace()

        self.canvas.ax_grn.plot(trace.grn.int, color="green")
        self.canvas.ax_red.plot(trace.acc.int, color="red")
        self.canvas.ax_alx.plot(trace.red.int, color="red")
        self.canvas.ax_fret.plot(trace.fret, color="orange")
        self.canvas.ax_stoi.plot(trace.stoi, color="purple")

        self.canvas.draw()


if __name__ == "__main__":
    ctxt = AppContext()
    ctxt.load_resources()

    cls = SetUp()
    # cls.predict()
    # cls.plot()
    cls.show()

    exit_code = ctxt.run()
    sys.exit(exit_code)
