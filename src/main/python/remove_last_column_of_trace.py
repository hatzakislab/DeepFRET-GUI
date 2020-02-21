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
    exp_txt = "Exported by DeepFRET"
    date_txt = "Date: {}".format(time.strftime("%Y-%m-%d, %H:%M"))
    if trace.y_pred is None:
        df = pd.DataFrame(
            {
                "D-Dexc-bg": trace.grn.bg,
                "A-Dexc-bg": trace.acc.bg,
                "A-Aexc-bg": trace.red.bg,
                "D-Dexc-rw": trace.grn.int,
                "A-Dexc-rw": trace.acc.int,
                "A-Aexc-rw": trace.red.int,
                "S": trace.stoi,
                "E": trace.fret,
            }
        ).round(4)
    else:
        df = pd.DataFrame(
            {
                "D-Dexc-bg": trace.grn.bg,
                "A-Dexc-bg": trace.acc.bg,
                "A-Aexc-bg": trace.red.bg,
                "D-Dexc-rw": trace.grn.int,
                "A-Dexc-rw": trace.acc.int,
                "A-Aexc-rw": trace.red.int,
                "S": trace.stoi,
                "E": trace.fret,
                "p_blch": trace.y_pred[:, 0],
                "p_aggr": trace.y_pred[:, 1],
                "p_stat": trace.y_pred[:, 2],
                "p_dyna": trace.y_pred[:, 3],
                "p_nois": trace.y_pred[:, 4],
                "p_scrm": trace.y_pred[:, 5],
            }
        ).round(4)

    mov_txt = "Movie filename: {}".format(trace.movie)
    id_txt = "FRET pair #{}".format(trace.n)
    bl_txt = (
        "Donor bleaches at: {} - "
        "Acceptor bleaches at: {}".format(
            trace.grn.bleach, trace.red.bleach
        )
    )
    outstr = ("{0}\n"
              "{1}\n"
              "{2}\n"
              "{3}\n"
              "{4}\n\n"
              "{5}".format(
        exp_txt,
        date_txt,
        mov_txt,
        id_txt,
        bl_txt,
        df.to_csv(index=False, sep="\t"),
    )
    )
    # print(outstr)
    with open(savename, "w") as f:
        f.write(outstr)


if __name__ == '__main__':
    filename = '/Users/mag/Desktop/Sample traces/trace_0_20200212_1734.txt'
    _trace = reduce_trace(filename)
    save_trace(_trace)
    _trace2 = TraceWindow.loadTraceFromAscii('temp.txt')
