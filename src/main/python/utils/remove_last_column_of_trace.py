import os

from lib.container import TraceContainer


def load_and_reduce_trace(filename: str) -> TraceContainer:
    # _TW = TraceWindow()
    trace = TraceContainer(filename)
    trace.red.int[:] = None
    trace.red.bg[:] = None
    trace.stoi[:] = None
    return trace


if __name__ == "__main__":
    filename = "/Users/mag/Desktop/Sample traces/trace_0_20200212_1734.txt"
    file_path = "temp.txt"
    _trace = load_and_reduce_trace(filename)
    _trace.savename = file_path
    s1 = _trace.get_export_txt()

    _trace.export_trace_to_txt(keep_nan_columns=False)

    _trace2 = TraceContainer(file_path)
    s2 = _trace2.get_export_txt()
    print(_trace.get_export_df().head())
    print(_trace2.get_export_df().head())
    # print(s1[:200])
    # print(s2[:200])
    # assert s1[:200] == s2[:200]
    os.remove(file_path)
