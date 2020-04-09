import sys

from main import AppContext, TraceContainer, TraceWindow


class SetUp(TraceWindow):
    class Data:
        traces = {}

    def __init__(self):
        super(SetUp, self).__init__()

        # Patch over the missing container (no MainWindow)
        self.data = self.Data()

    def setFile(self, path):
        """
        Basic setup
        """
        trace = TraceContainer(filename=path, loaded_from_ascii=True,)
        self.data.traces[trace.name] = trace
        self.currName = trace.name
        self._currName = trace.name

    def classify(self):
        """
        Predicts on the current trace
        """
        # resets self.currentName to None without a listView
        self.classifyTraces()

        self.currName = self._currName
        self.refreshPlot()

    def testFullTrace(self, **kwargs):
        self.setFile(**kwargs)
        self.classify()
        self.show()

    def testNoAATrace(self, **kwargs):
        self.setFile(**kwargs)
        self.classify()
        self.show()

    def testKinsoftTrace(self, **kwargs):
        self.setFile(**kwargs)
        self.classify()
        self.show()


if __name__ == "__main__":
    ctxt = AppContext()
    ctxt.load_resources()

    cls = SetUp()

    # cls.testFullTrace(path = "../resources/traces/Trace.txt")
    # cls.testNoAATrace(path = "../resources/traces/TraceNoAA.txt")
    cls.testKinsoftTrace(path="../resources/traces/kinsoftTrace.dat")

    exit_code = ctxt.run()
    sys.exit(exit_code)
