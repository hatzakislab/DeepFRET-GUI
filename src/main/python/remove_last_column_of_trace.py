from lib.container import TraceContainer
from src.main.python.main import TraceWindow
from io import StringIO



def reduce_trace(filename: str) -> TraceContainer:
    # _TW = TraceWindow()
    trace = TraceWindow.loadTraceFromAscii(filename)
    trace.acc.int = None
    trace.acc.bg = None
    trace.stoi = None
    return trace




if __name__ == '__main__':
    filename = '/Users/mag/Desktop/Sample traces/trace_0_20200212_1734.txt'
    _trace = reduce_trace(filename)

