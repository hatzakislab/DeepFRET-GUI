import multiprocessing

multiprocessing.freeze_support()

import time
import os
import pandas as pd
from typing import Union, Tuple


def pairwise(array):
    """Unpacks elements of an array (1,2,3,4...) into pairs, i.e. (1,2), (3,4), ..."""
    return zip(array[0::2], array[1::2])


def filter_nonetype(ls):
    """Filters out Nonetype objects from a list"""
    new = [v for v in ls if v is not None]
    return None if new == [] else new


def min_none(ls) -> Union[float, None]:
    """Returns minimum value of list, and None if all elements are None"""
    try:
        return min(ls)
    except TypeError:
        return None


def all_nonetype(ls):
    """Returns True if all values in iterable are None"""
    return all(v is None for v in ls)

def merge_tuples(*t):
    """
    Merges any number of tuples into one flattened
    """
    return tuple(j for i in (t) for j in (i if isinstance(i, tuple) else (i,)))


def timeit(method):
    """Decorator to time functions and methods for optimization"""

    def timed(*args, **kwargs):
        ts = time.time()
        result = method(*args, **kwargs)
        te = time.time()
        print("'{}' {:.2f} ms".format(method.__name__, (te - ts) * 1e3))
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
    """Seeks the file until specified line start is encountered in the start of the line."""
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
    """Seeks the file until specified header is encountered in the start of the line."""
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
