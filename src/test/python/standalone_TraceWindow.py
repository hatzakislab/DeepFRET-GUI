import sys
from main import TraceWindow, TraceContainer, AppContext, MovieData
from lib.misc import print_elapsed
from time import time


def setUp(trace_file_path):
    """
    Set up the essentials for the window to launch
    """
    mock_trace_file = trace_file_path
    trace = TraceContainer(mock_trace_file)
    TraceWindow.data = MovieData()
    TraceWindow.data.traces[trace.name] = trace
    TraceWindow.currName = trace.name
    TraceWindow_ = TraceWindow()
    TraceWindow_.classifyTraces(checked_only=False, single=False)
    TraceWindow_.currName = trace.name
    TraceWindow_.refreshPlot()
    return TraceWindow_


if __name__ == "__main__":
    ctxt = AppContext()
    ctxt.load_resources()

    start = time()

    TraceWindow_ = setUp("../resources/traces/fiddler_3dim_0.txt")
    TraceWindow_.show()

    end = time()
    print_elapsed(start, end, "TraceWindow")

    exit_code = ctxt.run()
    sys.exit(exit_code)
