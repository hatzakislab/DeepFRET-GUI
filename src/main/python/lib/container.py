import multiprocessing
import os.path
import re
import time
import warnings

multiprocessing.freeze_support()

from global_variables import GlobalVariables as gvars
from matplotlib.colors import LinearSegmentedColormap
from typing import Union, Tuple
import numpy as np
import pandas as pd
import skimage.io
import lib.imgdata
import lib.math
import lib.misc


class ImageContainer:
    """
    Class for storing individual image information.
    """

    def __init__(self):
        # stores the raw data
        self.img = None  # type: Union[None, np.ndarray]
        self.indices = None  # type: Union[None, np.ndarray]
        self.width = None  # type: Union[None, int]
        self.height = None  # type: Union[None, int]
        self.roi_radius = None  # type: Union[None, int]
        self.channels = None  # type: Union[None, Tuple[ImageChannel]]
        self.coloc_frac = None  # type: Union[None, float]

        # Image color channels
        self.grn = ImageChannel("green")
        self.red = ImageChannel("red")
        self.acc = ImageChannel("red")

        # Colocalized channels
        self.coloc_grn_red = ColocalizedParticles("green", "red")
        self.coloc_all = ColocalizedAll()

        # Blended channels
        self.grn_red_blend = None  # type: Union[None, np.ndarray]

        # Create a class to store traces and associated information for every
        # image name
        self.traces = {}

        # To iterate over all channels
        self.all = (self.grn, self.red, self.coloc_grn_red, self.coloc_all)


class ImageChannel:
    """
    Class for storing image channel information.
    """

    def __init__(self, color):
        if color == "green":
            cmap = LinearSegmentedColormap.from_list(
                "", ["black", gvars.color_green]
            )
        elif color == "red":
            cmap = LinearSegmentedColormap.from_list(
                "", ["black", gvars.color_red]
            )
        else:
            raise ValueError(
                "Invalid color. Available options are 'green' or 'red'"
            )

        self.exists = True  # type: bool
        self.cmap = cmap  # type: LinearSegmentedColormap
        self.color = color  # type: str
        self.raw = None  # type: Union[None, np.ndarray]
        self.mean = None  # type: Union[None, np.ndarray]
        self.rgba = None  # type: Union[None, np.ndarray]
        self.spots = None  # type: Union[None, np.ndarray]
        self.n_spots = 0  # type: int


class ColocalizedParticles:
    """
    Class for storing colocalized particle information
    """

    def __init__(self, color1, color2):
        self.color1 = color1  # type: str
        self.color2 = color2  # type: str

        self.spots = None  # type: Union[None, pd.DataFrame]
        self.n_spots = 0  # type: int


class ColocalizedAll:
    """
    Class for storing colocalized particles for all channels
    """

    def __init__(self):
        self.spots = None  # type: Union[None, pd.DataFrame]
        self.n_spots = 0  # type: int


class TraceChannel:
    """
    Class for storing trace information for individual channels
    """

    def __init__(self, color):
        self.color = color  # type: str
        self.int = None  # type: Union[None, np.ndarray]
        self.bg = None  # type: Union[None, np.ndarray]
        self.bleach = None  # type: Union[None, int]


class TraceContainer:
    """
    Class for storing individual newTrace information.
    """

    ml_column_names = [
        "p_bleached",
        "p_aggegate",
        "p_noisy",
        "p_scramble",
        "p_1-state",
        "p_2-state",
        "p_3-state",
        "p_4-state",
        "p_5-state",
    ]

    def __init__(self, filename, name=None, movie=None, n=None):
        self.filename = filename  # type: str
        self.name = (
            name if name is not None else os.path.basename(filename)
        )  # type: str
        self.movie = movie  # type: str
        self.n = n  # type: str

        self.tracename = None  # type: Union[None, str]
        self.savename = None  # type: Union[None, str]

        self.load_successful = False

        self.is_checked = False  # type: bool
        self.xdata = []  # type: [int, int]

        self.grn = TraceChannel(color="green")
        self.red = TraceChannel(color="red")
        self.acc = TraceChannel(color="red")

        self.first_bleach = None  # int
        self.zerobg = None  # type: (None, np.ndarray)

        self.fret = None  # type: Union[None, np.ndarray]
        self.stoi = None  # type: Union[None, np.ndarray]
        self.hmm = None  # type: Union[None, np.ndarray]
        self.hmm_idx = None  # type: Union[None, np.ndarray]
        self.transitions = None  # type: Union[None, pd.DataFrame]
        self.y_pred = None  # type: Union[None, np.ndarray]
        self.y_class = None  # type: Union[None, np.ndarray]
        self.confidence = None  # type: Union[None, float]

        self.a_factor = np.nan  # type: float
        self.d_factor = np.nan  # type: float
        self.frames = None  # type: Union[None, int]
        self.frames_max = None  # type: Union[None, int]
        self.framerate = None  # type: Union[None, float]

        self.channels = self.grn, self.red, self.acc
        try:
            self.load_from_ascii()
        except (TypeError, FileNotFoundError) as e:
            try:
                self.load_from_dat()
            except (TypeError, FileNotFoundError) as e:
                warnings.warn(
                    "Warning! No data loaded for this trace!", UserWarning
                )

    def load_from_ascii(self):
        """
        Reads a trace from an ASCII text file. Several checks are included to
        include flexible compatibility with different versions of trace exports.
        Also includes support for all iSMS traces.
        """
        colnames = [
            "D-Dexc-bg.",
            "A-Dexc-bg.",
            "A-Aexc-bg.",
            "D-Dexc-rw.",
            "A-Dexc-rw.",
            "A-Aexc-rw.",
            "S",
            "E",
        ]
        if self.filename.endswith(".dat"):
            raise TypeError("Datafile is not the right type for this function!")

        with open(self.filename) as f:
            txt_header = [next(f) for _ in range(5)]

        # This is for iSMS compatibility
        if txt_header[0].split("\n")[0] == "Exported by iSMS":
            df = pd.read_csv(self.filename, skiprows=5, sep="\t", header=None)
            if len(df.columns) == colnames:
                df.columns = colnames
            else:
                try:
                    df.columns = colnames
                except ValueError:
                    colnames = colnames[3:]
                    df.columns = colnames
        # Else DeepFRET trace compatibility
        else:
            df = lib.misc.csv_skip_to(
                path=self.filename, line="D-Dexc", sep="\s+"
            )
        try:
            pair_n = lib.misc.seek_line(
                path=self.filename, line_starts="FRET pair"
            )
            self.n = int(pair_n.split("#")[-1])

            movie = lib.misc.seek_line(
                path=self.filename, line_starts="Movie filename"
            )
            self.movie = movie.split(": ")[-1]

        except (ValueError, AttributeError):
            pass

        self.load_successful = True

        # Add flag to see if incomplete trace
        if not any(s.startswith("A-A") for s in df.columns):
            df["A-Aexc-rw"] = np.nan
            df["A-Aexc-bg"] = np.nan
            df["A-Aexc-I"] = np.nan

        if "D-Dexc_F" in df.columns:
            warnings.warn(
                "This trace is created with an older format.",
                DeprecationWarning,
            )
            self.grn.int = df["D-Dexc_F"].values
            self.acc.int = df["A-Dexc_I"].values
            self.red.int = df["A-Aexc_I"].values

            zeros = np.zeros(len(self.grn.int))
            self.grn.bg = zeros
            self.acc.bg = zeros
            self.red.bg = zeros

        else:
            if "p_bleached" in df.columns:
                colnames += self.ml_column_names
                self.y_pred = df[self.ml_column_names].values
                self.y_class, self.confidence = lib.math.seq_probabilities(
                    self.y_pred
                )

            # This strips periods if present
            df.columns = [c.strip(".") for c in df.columns]

            self.grn.int = df["D-Dexc-rw"].values
            self.acc.int = df["A-Dexc-rw"].values
            self.red.int = df["A-Aexc-rw"].values

            try:
                self.grn.bg = df["D-Dexc-bg"].values
                self.acc.bg = df["A-Dexc-bg"].values
                self.red.bg = df["A-Aexc-bg"].values
            except KeyError:
                zeros = np.zeros(len(self.grn.int))
                self.grn.bg = zeros
                self.acc.bg = zeros
                self.red.bg = zeros

        self.calculate_fret()
        self.calculate_stoi()

        self.frames = np.arange(1, len(self.grn.int) + 1, 1)
        self.frames_max = self.frames.max()

    def load_from_dat(self):
        """
        Loading from .dat files, as supplied in the kinSoft challenge
        """
        with open(self.filename) as f:
            arr = np.loadtxt(str(f))

        l = len(arr)
        zeros = np.zeros(len(arr))

        self.grn.int = arr[:, 1]
        self.acc.int = arr[:, 2]
        self.red.int = zeros * np.nan

        self.grn.bg = zeros
        self.acc.bg = zeros
        self.red.bg = zeros * np.nan

        self.framerate = int(1 / (arr[0, 1] - arr[0, 0]))

        self.calculate_fret()
        self.calculate_stoi()

        self.frames = np.arange(1, l + 1, 1)
        self.frames_max = self.frames.max()

        self.load_successful = True

    def get_intensities(self):
        """
        Convenience function to return trace get_intensities
        """
        grn_int = self.grn.int  # type: Union[None, np.ndarray]
        grn_bg = self.grn.bg  # type: Union[None, np.ndarray]
        acc_int = self.acc.int  # type: Union[None, np.ndarray]
        acc_bg = self.acc.bg  # type: Union[None, np.ndarray]
        red_int = self.red.int  # type: Union[None, np.ndarray]
        red_bg = self.red.bg  # type: Union[None, np.ndarray]

        return grn_int, grn_bg, acc_int, acc_bg, red_int, red_bg

    def get_bleaches(self):
        """
        Convenience function to return trace bleaching times
        """
        grn_bleach = self.grn.bleach  # type: Union[None, int]
        acc_bleach = self.acc.bleach  # type: Union[None, int]
        red_bleach = self.red.bleach  # type: Union[None, int]
        return grn_bleach, acc_bleach, red_bleach

    def get_export_df(self, keep_nan_columns: Union[bool, None] = None):
        """
        Returns the DataFrame to use for export
        """
        if keep_nan_columns is None:
            keep_nan_columns = True
        dfdict = {
            "D-Dexc-bg": self.grn.bg,
            "A-Dexc-bg": self.acc.bg,
            "A-Aexc-bg": self.red.bg,
            "D-Dexc-rw": self.grn.int,
            "A-Dexc-rw": self.acc.int,
            "A-Aexc-rw": self.red.int,
            "S": self.stoi,
            "E": self.fret,
        }

        if self.y_pred is not None:
            # Add predictions column names and values
            dfdict.update(dict(zip(self.ml_column_names, self.y_pred.T)))

        df = pd.DataFrame(dfdict).round(4)

        if keep_nan_columns is False:
            df.dropna(axis=1, how="all", inplace=True)

        return df

    def get_export_txt(
        self,
        df: Union[None, pd.DataFrame] = None,
        exp_txt: Union[None, str] = None,
        date_txt: Union[None, str] = None,
        keep_nan_columns: Union[bool, None] = None,
    ):
        """
        Returns the string to use for saving the trace as a txt
        """
        if df is None:
            df = self.get_export_df(keep_nan_columns=keep_nan_columns)
        if (exp_txt is None) or (date_txt is None):
            exp_txt = "Exported by DeepFRET"
            date_txt = "Date: {}".format(time.strftime("%Y-%m-%d, %H:%M"))

        mov_txt = "Movie filename: {}".format(self.movie)
        id_txt = "FRET pair #{}".format(self.n)
        bl_txt = "Donor bleaches at: {} - " "Acceptor bleaches at: {}".format(
            self.grn.bleach, self.red.bleach
        )
        return (
            "{0}\n"
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
                df.to_csv(index=False, sep="\t", na_rep="NaN"),
            )
        )

    def get_tracename(self) -> str:
        if self.tracename is None:
            if self.movie is None:
                name = "Trace_pair{}.txt".format(self.n)
            else:
                name = "Trace_{}_pair{}.txt".format(
                    self.movie.replace(".", "_"), self.n
                )

            # Scrub mysterious \n if they appear due to filenames
            name = "".join(name.splitlines(keepends=False))
            self.tracename = name

        return self.tracename

    def get_savename(self, dir_to_join: Union[None, str] = None):
        if self.savename is None:
            if dir_to_join is not None:
                self.savename = os.path.join(dir_to_join, self.get_tracename())
            else:
                self.savename = self.get_tracename()
        return self.savename

    def export_trace_to_txt(
        self,
        dir_to_join: Union[None, str] = None,
        keep_nan_columns: Union[bool, None] = None,
    ):
        savename = self.get_savename(dir_to_join=dir_to_join)
        with open(savename, "w") as f:
            f.write(self.get_export_txt(keep_nan_columns=keep_nan_columns))

    def calculate_fret(self):
        self.fret = lib.math.calc_E(self.get_intensities())

    def calculate_stoi(self):
        self.stoi = lib.math.calc_S(self.get_intensities())


class MovieData:
    """
    Data wrapper that contains a dict with filenames and associated data.
    """

    def __init__(self):
        self.movies = {}
        self.traces = {}
        self.currName = None

    def get(self, name) -> ImageContainer:
        """Shortcut to return the metadata of selected movie."""
        if self.currName is not None:
            return self.movies[name]

    def _data(self) -> ImageContainer:
        """
        Shortcut for returning the metadata of movie currently being loaded,
        for internal use.
        Frontend should use get() instead
        """
        if self.currName is not None:
            return self.movies[self.currName]

    def load_img(self, path, name, setup, bg_correction: bool = False):
        """
        Loads movie and extracts all parameters depending on the number of
        channels

        Parameters
        ----------
        path:
            Full path of video to be loaded
        name:
            Unique identifier (to prevent videos being loaded in duplicate)
        setup:
            Imaging setup from config
        bg_correction:
            Corrects for illumination profile and crops misaligned edges
            slightly
        """
        # Instantiate container
        self.currName = name
        self.name = name
        self.movies[name] = ImageContainer()
        self._data().traces = {}
        # Populate metadata
        self._data().img = skimage.io.imread(path)

        swapflag = False
        try:
            swapflag = (setup != "dual") and (self._data().img.shape[3] > 3)
        except IndexError as e:
            pass

        if swapflag:
            self._data().img = self._data().img.swapaxes(3, 1).swapaxes(2, 1)

        self._data().height = self._data().img.shape[1]
        self._data().width = self._data().img.shape[2]
        # Scaling factor for ROI
        self._data().roi_radius = (
            max(self._data().height, self._data().width) / 80
        )
        if self._data().height == self._data().width:  # quadratic image
            if setup == "dual":
                c1, c2, c3, _ = lib.imgdata.image_channels(4)

                # Acceptor (Dexc-Aem)
                self._data().acc.raw = self._data().img[c1, :, :]
                # ALEX (Aexc-Aem)
                self._data().red.raw = self._data().img[c2, :, :]
                # Donor (Dexc-Dem)
                self._data().grn.raw = self._data().img[c3, :, :]

                self._data().channels = self._data().grn, self._data().red

            elif setup == "2-color":
                top, btm, lft, rgt = lib.imgdata.image_quadrants(
                    height=self._data().height, width=self._data().width
                )

                self._data().grn.raw = self._data().img[:, top, lft, 0]
                self._data().acc.raw = self._data().img[:, top, rgt, 0]
                self._data().red.raw = self._data().img[:, top, rgt, 1]

                self._data().channels = self._data().grn, self._data().red

            elif setup == "2-color-inv":
                top, btm, lft, rgt = lib.imgdata.image_quadrants(
                    height=self._data().height, width=self._data().width
                )

                if self._data().img.shape[3] == 2:
                    self._data().grn.raw = self._data().img[:, btm, rgt, 0]
                    self._data().acc.raw = self._data().img[:, btm, lft, 0]
                    self._data().red.raw = self._data().img[:, btm, lft, 1]

                    self._data().channels = self._data().grn, self._data().red

                else:
                    raise ValueError("Format not supported.")

            elif setup == "bypass":
                self._data().grn.raw = self._data().img[:, :, :, 0]
                self._data().red.raw = self._data().img[:, :, :, 1]

                self._data().channels = (
                    self._data().grn,
                    self._data().red,
                )

                self._data().red.exists = True
                self._data().acc.exists = False
            else:
                raise ValueError("Format not supported.")
        else:
            lft, rgt = lib.imgdata.rectangle_quadrants(
                h=self._data().height, w=self._data().width
            )
            # ALEX (Aexc-Aem)
            self._data().red.raw = self._data().img[0::2, :, lft]
            # Acceptor (Dexc-Aem)
            self._data().acc.raw = self._data().img[1::2, :, lft]
            # Donor (Dexc-Dem)
            self._data().grn.raw = self._data().img[1::2, :, rgt]

            self._data().channels = self._data().grn, self._data().red

            self._data().red.exists = True
            self._data().acc.exists = True
            self._data().grn.exists = True

        for c in self._data().channels + (self._data().acc,):
            if c.raw is not None:
                c.raw = np.abs(c.raw)
                t, h, w = c.raw.shape

                # Crop 2% of sides to avoid messing up background detection
                crop_h = int(h // 50)
                crop_w = int(w // 50)
                c.raw = c.raw[:, crop_h : h - crop_h, crop_w : w - crop_w]
                c.mean = c.raw[0 : t // 20, :, :].mean(axis=0)
                c.mean = lib.imgdata.zero_one_scale(c.mean)
                if bg_correction:
                    c.mean_nobg = lib.imgdata.subtract_background(
                        c.mean, by="row", return_bg_only=False
                    )
                else:
                    c.mean_nobg = c.mean

        if self._data().red.exists:
            self._data().indices = np.indices(self._data().red.mean.shape)
        else:
            self._data().indices = np.indices(self._data().grn.mean.shape)
