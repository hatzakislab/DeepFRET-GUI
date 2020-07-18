import itertools
import multiprocessing
import re
import sys
import random

from matplotlib.ticker import FuncFormatter

multiprocessing.freeze_support()

import time
import os
import pandas as pd
import numpy as np
from typing import Union, Tuple, List


def pairwise(array):
    """Unpacks elements of an array (1,2,3,4...) into pairs,
    i.e. (1,2), (3,4), ..."""
    return zip(array[0::2], array[1::2])


def filter_nonetype(ls):
    """Filters out Nonetype objects from a list"""
    new = [v for v in ls if v is not None]
    return None if new == [] else new


def min_none(ls) -> Union[float, None]:
    """Returns minimum value of list, and None if all elements are None"""
    v = [x for x in ls if x is not None]
    return np.min(v) if v else None


def all_nonetype(ls):
    """Returns True if all values in iterable are None"""
    return all(v is None for v in ls)


def merge_tuples(*t):
    """
    Merges any number of tuples into one flattened
    """
    return tuple(j for i in (t) for j in (i if isinstance(i, tuple) else (i,)))


def print_elapsed(start, end, name=""):
    """Print the time elapsed given start and end time() points"""
    print("'{}' {:.2f} ms".format(name, (end - start) * 1e3))


def timeit(method, *args, **kwargs):
    """Decorator to time functions and methods for optimization"""

    def timed(*args, **kwargs):
        ts = time.time()
        result = method(*args, **kwargs)
        te = time.time()
        print_elapsed(name=method.__name__, start=ts, end=te)
        return result

    return timed


def m_append(objects: tuple, to: tuple, method="append"):
    """
    Appends multiple objects to multiple lists, for readability

    Parameters
    ----------
    objects:
        Tuple of objects to append
    to:
        Tuple of lists to append to, in the given order
    method:
        Whether to use append or extend method for list
    """

    if len(objects) != len(to):
        raise ValueError("Tuples must be of equal length")

    if method == "append":
        for o, ls in zip(objects, to):
            ls.append(o)
    elif method == "extend":
        for o, ls in zip(objects, to):
            ls.extend(o)
    else:
        raise ValueError("Method must be 'append' or 'extend'")


def seek_line(
    line_starts: Union[str, Tuple[str, str]], path: str, timeout: int = 10
):
    """Seeks the file until specified line start is encountered in the start of
     the line."""
    with open(path, encoding="utf-8") as f:
        n = 0
        if isinstance(line_starts, str):
            line_starts = (line_starts,)
        s = [False] * len(line_starts)
        while not any(s):
            line = f.readline()
            s = [line.startswith(ls) for ls in line_starts]
            n += 1
            if n > timeout:
                return None
        return line


def csv_skip_to(path, line, timeout=10, **kwargs):
    """Seeks the file until specified header is encountered in the start of
    the line."""
    if os.stat(path).st_size == 0:
        raise ValueError("File is empty")
    with open(path, encoding="utf-8") as f:
        n = 0
        pos = 0
        cur_line = f.readline()
        while not cur_line.startswith(line):
            pos = f.tell()
            cur_line = f.readline()
            n += 1
            if n > timeout:
                return None
        f.seek(pos)
        return pd.read_csv(f, **kwargs)


def generate_unique_name(full_filename, array):
    """
    Checks for a name in a given array. If name already exists,
    append _n numbering to make unique, and return the unique name.
    """
    name = os.path.basename(full_filename)
    # Test if name has already been loaded
    if name in array:
        n = 1
        # Create a new unique name
        unique_name = name + "_" + str(n)
        while unique_name in array:
            n += 1
        else:
            return unique_name
    else:
        return name


def labels_to_binary(y, one_hot, to_ones):
    """Converts group labels to binary labels, given desired targets"""
    if one_hot:
        y = y.argmax(axis=2)
    y[~np.isin(y, to_ones)] = -1
    y[y != -1] = 1
    y[y != 1] = 0
    return y


def sim_to_ascii(df, trace_len, outdir):
    """
    Saves simulated traces to ASCII .txt files
    """
    df.index = np.arange(0, len(df), 1) // trace_len

    y = []
    exp_txt = "Simulated trace"
    for idx, trace in df.groupby(df.index):
        l = trace["label"].values
        l = l[l != 0]
        l = 0 if l.size == 0 else l[0]
        y.append(int(l))

        bg = np.zeros(len(trace))
        path = os.path.join(
            outdir, "trace_{}_{}.txt".format(idx, time.strftime("%Y%m%d_%H%M"))
        )

        df = pd.DataFrame(
            {
                "D-Dexc-bg": bg,
                "A-Dexc-bg": bg,
                "A-Aexc-bg": bg,
                "D-Dexc-rw": trace["DD"],
                "A-Dexc-rw": trace["DA"],
                "A-Aexc-rw": trace["AA"],
                "S": trace["S"],
                "E": trace["E"],
            }
        ).round(4)

        date_txt = "Date: {}".format(time.strftime("%Y-%m-%d, %H:%M"))
        vid_txt = "Video filename: {}".format(None)
        id_txt = "FRET pair #{}".format(idx)
        bl_txt = "Bleaches at {}".format(trace["fb"].values[0])

        with open(path, "w") as f:
            f.write(
                "{0}\n"
                "{1}\n"
                "{2}\n"
                "{3}\n"
                "{4}\n\n"
                "{5}".format(
                    exp_txt,
                    date_txt,
                    vid_txt,
                    id_txt,
                    bl_txt,
                    df.to_csv(index=False, sep="\t"),
                )
            )
    y = pd.Series(y)
    y = labels_to_binary(y, one_hot=False, to_ones=(2, 3))
    y.to_csv(os.path.join(outdir, "y.txt"), sep="\t")


def numstring_to_ls(s):
    """
    Transforms any string of numbers into a list of floats,
    regardless of separators

    Zero is ignored!
    """
    num_s = re.findall(r"\d+(\.\d+)?\s*", s)
    return [float(s) for s in num_s if s != ""]


def random_seed_mp(verbose=False):
    """Initializes a pseudo-random seed for multiprocessing use"""
    seed_val = int.from_bytes(os.urandom(4), byteorder="little")
    np.random.seed(seed_val)
    if verbose:
        print("Random seed value: {}".format(seed_val))


def count_adjacent_values(arr):
    """
    Returns start index and length of segments of equal values.

    Example for plotting several axvspans:
    --------------------------------------
    adjs, lns = lib.count_adjacent_true(score)
    t = np.arange(1, len(score) + 1)

    for ax in axes:
        for starts, ln in zip(adjs, lns):
            alpha = (1 - np.mean(score[starts:starts + ln])) * 0.15
            ax.axvspan(xmin = t[starts], xmax = t[starts] + (ln - 1),
            alpha = alpha, color = "red", zorder = -1)
    """
    arr = arr.ravel()

    n = 0
    same = [(g, len(list(l))) for g, l in itertools.groupby(arr)]
    starts = []
    lengths = []
    for v, l in same:
        _len = len(arr[n : n + l])
        _idx = n
        n += l
        lengths.append(_len)
        starts.append(_idx)
    return starts, lengths


def nice_string_output(
    names: List[str], values: List[str], extra_spacing: int = 0,
):
    """
    Makes a single multiline string of names and values, for plotting
    :param names: List of strings with names
    :param values: List of strings with values
    :param extra_spacing: spacing between names and values, for wider plots
    :return string: formatted string
    Example:
    ---------
    >>>In: nice_string_output(["Carrots","Peas"], ["Many", "Few"])
    >>>Carrots Many
    >>>Peas     Few
    """
    max_values = len(max(values, key=len))
    max_names = len(max(names, key=len))
    string = ""
    for name, value in zip(names, values):
        string += "{0:s} {1:>{spacing}} \n".format(
            name,
            value,
            spacing=extra_spacing + max_values + max_names - len(name),
        )
    return string[:-2]


@FuncFormatter
def format_string_to_k(x, pos):
    """
    Matplotlib tick formatter.
    Removes 3 0s from end and appends a k.
    for example, 3000 -> 3k
    :param x: tick string
    :param pos: tick position, not used
    :return s: formatted string
    """
    s = f"{int(x):d}"
    if s.endswith("000"):
        s = s[:-3] + "k"
    elif s.endswith("500"):
        s = s[:-3] + ".5k"
    return s


def remove_newlines(s) -> str:
    """
    Removes all newlines from string
    """
    return "".join(s.splitlines(keepends=False))


def generate_name(length=10, module=None):
    """
    Generates a random ID for a module
    """
    if module is None:
        module = sys.modules[__name__]
    while True:
        name = "id{:0{length}d}".format(
            random.randint(0, 10 ** length - 1), length=length
        )
        if not hasattr(module, name):
            return name


def global_function(func):
    """
    Decorate a local function to make it global, thus enabling
    multiprocessing pickling of it
    """
    module = sys.modules[func.__module__]
    func.__name__ = func.__qualname__ = generate_name(module=module)
    setattr(module, func.__name__, func)
    return func
