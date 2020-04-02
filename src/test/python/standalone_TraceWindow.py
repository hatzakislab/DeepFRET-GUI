import sys
from main import TraceWindow, TraceContainer, AppContext, MovieData

def build():
    mock_trace_file = '../resources/traces/fiddler_3dim_0.txt'
    trace = TraceContainer(mock_trace_file)
    TraceWindow.data = MovieData()
    TraceWindow.data.traces[trace.name] = trace
    TraceWindow.currName = trace.name
    TraceWindow_ = TraceWindow()
    TraceWindow_.currName = trace.name
    return TraceWindow_

if __name__ == "__main__":
    ctxt = AppContext()
    ctxt.load_resources()

    TraceWindow_ = build()

    TraceWindow_.refreshPlot()
    TraceWindow_.show()

    exit_code = ctxt.run()
    sys.exit(exit_code)