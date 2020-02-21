import time

from lib.container import TraceContainer
from src.main.python.main import TraceWindow
from io import StringIO
import pandas as pd


def reduce_trace(filename: str) -> TraceContainer:
    # _TW = TraceWindow()
    trace = TraceWindow.loadTraceFromAscii(filename)
    trace.acc.int = None
    trace.acc.bg = None
    trace.stoi = None
    return trace


def save_trace(trace:TraceContainer, savename='temp.txt'):

    with open(savename, "w") as f:
        f.write(outstr)


if __name__ == '__main__':
    filename = '/Users/mag/Desktop/Sample traces/trace_0_20200212_1734.txt'
    _trace = reduce_trace(filename)
    save_trace(_trace)
    _trace2 = TraceWindow.loadTraceFromAscii('temp.txt')
