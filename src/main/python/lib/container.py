import multiprocessing
import os.path
import time
import warnings

import PIL.Image
import PIL.TiffTags

from lib.utils import timeit

multiprocessing.freeze_support()

from global_variables import GlobalVariables as gvars
from matplotlib.colors import LinearSegmentedColormap
from typing import Union, Tuple, Optional
import numpy as np
import pandas as pd
import skimage.io
import lib.imgdata
import lib.math
import lib.utils
import astropy.io.fits


class VideoContainer:
    """
    Class for storing individual video information.
    """

    def __init__(self):
        # stores the raw data
        self.array = None  # type: Optional[np.ndarray]
        self.alex = None  # type: Optional[bool]
        self.indices = None  # type: Optional[np.ndarray]
        self.width = None  # type: Optional[int]
        self.height = None  # type: Optional[int]
        self.roi_radius = None  # type: Optional[int]
        self.channels = None  # type: Optional[Tuple[ImageChannel]]
        self.coloc_frac = None  # type: Optional[float]

        # Image color channels
        self.grn = ImageChannel("green")
        self.red = ImageChannel("red")
        self.acc = ImageChannel("red")

        # Colocalized channels
        self.coloc_grn_red = ColocalizedParticles("green", "red")
        self.coloc_all = ColocalizedAll()

        # Blended channels
        self.grn_red_blend = None  # type: Optional[np.ndarray]

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
        self.raw = None  # type: Optional[np.ndarray]
        self.mean = None  # type: Optional[np.ndarray]
        self.mean_nobg = None  # type: Optional[np.ndarray]
        self.rgba = None  # type: Optional[np.ndarray]
        self.spots = None  # type: Optional[np.ndarray]
        self.n_spots = 0  # type: int


class ColocalizedParticles:
    """
    Class for storing colocalized particle information
    """

    def __init__(self, color1, color2):
        self.color1 = color1  # type: str
        self.color2 = color2  # type: str

        self.spots = None  # type: Optional[pd.DataFrame]
        self.n_spots = 0  # type: int


class ColocalizedAll:
    """
    Class for storing colocalized particles for all channels
    """

    def __init__(self):
        self.spots = None  # type: Optional[pd.DataFrame]
        self.n_spots = 0  # type: int


class TraceChannel:
    """
    Class for storing trace information for individual channels
    """

    def __init__(self, color):
        self.color = color  # type: str
        self.int = None  # type: Optional[np.ndarray]
        self.bg = None  # type: Optional[np.ndarray]
        self.bleach = None  # type: Optional[int]


class TraceContainer:
    """
    Class for storing individual trace information.
    """

    # TODO: make these names switchable depending on global vars and model config
    ml_column_names = [
        "p_bleached",
        "p_aggregated",
        "p_noisy",
        "p_scrambled",
        "p_static",
        "p_dynamic",
    ]

    def __init__(
        self,
        filename=None,
        name=None,
        video=None,
        n=None,
        hmm_idealized_config=None,
        loaded_from_ascii=False,
    ):
        self.filename = filename  # type: str
        self.name = (
            name if name is not None else os.path.basename(filename)
        )  # type: str
        self.video = video  # type: str
        self.n = n  # type: str

        self.tracename = None  # type: Optional[str]
        self.savename = None  # type: Optional[str]

        self.load_successful = False

        self.is_checked = False  # type: bool
        self.xdata = []  # type: [int, int]

        self.grn = TraceChannel(color="green")
        self.red = TraceChannel(color="red")
        self.acc = TraceChannel(color="red")

        self.first_bleach = None  # int
        self.zerobg = None  # type: (None, np.ndarray)

        self.fret = None  # type: Optional[np.ndarray]
        self.stoi = None  # type: Optional[np.ndarray]

        # hmm configuration
        self.hmm_idealized_config = (
            hmm_idealized_config
            if hmm_idealized_config is not None
            else "global"
        )  # type: Optional[str]
        # hmm predictions here
        self.hmm = None  # type: Optional[np.ndarray]
        self.hmm_idx = None  # type: Optional[np.ndarray]
        self.hmm_state = None  # type: Optional[np.ndarray]
        # detailed hmm data here
        self.hmm_local_raw = None  # type: Optional[np.ndarray]
        self.hmm_global_raw = None  # type: Optional[np.ndarray]
        self.hmm_local_fret = None  # type: Optional[np.ndarray]
        self.hmm_global_fret = None  # type: Optional[np.ndarray]
        self.transitions = None  # type: Optional[pd.DataFrame]

        # deep learning data here
        self.y_pred = None  # type: Optional[np.ndarray]
        self.y_class = None  # type: Optional[np.ndarray]
        self.confidence = None  # type: Optional[float]

        self.a_factor = np.nan  # type: float
        self.d_factor = np.nan  # type: float
        self.frames = None  # type: Optional[int]
        self.frames_max = None  # type: Optional[int]
        self.framerate = None  # type: Optional[float]

        self.channels = self.grn, self.red, self.acc
        # file loading
        # TODO: make compatible w pathlib
        # TODO: set flag differently?
        if loaded_from_ascii:
            if self.filename.endswith("txt"):
                try:
                    self.load_from_txt()
                except (TypeError, FileNotFoundError) as e:
                    warnings.warn(
                        "Warning! No data loaded for this trace!", UserWarning
                    )
            elif self.filename.endswith("dat"):
                try:
                    self.load_from_dat()
                except (TypeError, FileNotFoundError) as e:
                    warnings.warn(
                        "Warning! No data loaded for this trace!", UserWarning
                    )

    @property
    def hmm(self):
        """
        The self.hmm is now a property, so that we can set a global or local flag for the traces.
        This maintains compatibility and allows us to change the type of output dynamically.
        """
        if self.hmm_idealized_config.lower().startswith("glo"):
            if self.hmm_global_fret is not None:
                return self.hmm_global_fret
            else:
                return self.hmm_local_fret
        elif self.hmm_idealized_config.lower().startswith("loc"):
            if self.hmm_local_fret is not None:
                return self.hmm_local_fret
            else:
                return self.hmm_global_fret

    @hmm.setter
    def hmm(self, val):
        if self.hmm_idealized_config.lower().startswith("glo"):
            self.hmm_global_fret = val
        elif self.hmm_idealized_config.lower().startswith("loc"):
            self.hmm_local_fret = val

    def load_from_txt(self):
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
                    if len(df.keys()) == 3:  # Non ALEX
                        colnames = colnames[3:5] + [colnames[-1]]
                    else:
                        colnames = colnames[3:]
                    df.columns = colnames
        # Else DeepFRET trace compatibility
        else:
            df = lib.utils.csv_skip_to(
                path=self.filename, line="D-Dexc", sep="\s+"
            )
        try:
            pair_n = lib.utils.seek_line(
                path=self.filename, line_starts="FRET pair"
            )
            self.n = int(pair_n.split("#")[-1])
        except (ValueError, AttributeError):
            pass

        try:
            video = lib.utils.seek_line(
                path=self.filename, line_starts="Video filename"
            )
            self.video = video.split(": ")[-1]
        except (ValueError, AttributeError):
            pass
        try:
            bleach = lib.utils.seek_line(
                path=self.filename, line_starts="Bleaches"
            )
            bleach_str = bleach.split(" ")[-1]
            self.first_bleach = int(bleach_str)
            self.red.bleach = self.first_bleach
            self.grn.bleach = self.first_bleach
            self.acc.bleach = self.first_bleach
            # TODO: consider changing this to be a property of TraceContainer

        except (ValueError, AttributeError):
            pass

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
                # Try/except to ignore if using incompatible older traces
                try:
                    colnames += self.ml_column_names
                    self.y_pred = df[self.ml_column_names].values
                    (
                        self.y_class,
                        self.confidence,
                        _,
                    ) = lib.math.seq_probabilities(self.y_pred)
                except KeyError:
                    pass

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

        self.load_successful = True

    def load_from_dat(self):
        """
        Reads and loads trace data from .dat file, as supplied in the kinSoft challenge.
        These traces were supplied as non-ALEX traces, so the red TraceChannel is all NaNs.
        """
        arr = np.loadtxt(self.filename)

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
        grn_int = self.grn.int  # type: Optional[np.ndarray]
        grn_bg = self.grn.bg  # type: Optional[np.ndarray]
        acc_int = self.acc.int  # type: Optional[np.ndarray]
        acc_bg = self.acc.bg  # type: Optional[np.ndarray]
        red_int = self.red.int  # type: Optional[np.ndarray]
        red_bg = self.red.bg  # type: Optional[np.ndarray]

        return grn_int, grn_bg, acc_int, acc_bg, red_int, red_bg

    def get_bleaches(self):
        """
        Convenience function to return trace bleaching times
        """
        grn_bleach = self.grn.bleach  # type: Optional[int]
        acc_bleach = self.acc.bleach  # type: Optional[int]
        red_bleach = self.red.bleach  # type: Optional[int]
        return grn_bleach, acc_bleach, red_bleach

    def set_from_df(self, df):
        """
        Set intensities from a dataframe
        """
        self.grn.bg = df["D-Dexc-bg"].values
        self.acc.bg = df["A-Dexc-bg"].values
        self.red.bg = df["A-Aexc-bg"].values
        self.grn.int = df["D-Dexc-rw"].values
        self.acc.int = df["A-Dexc-rw"].values
        self.red.int = df["A-Aexc-rw"].values
        self.stoi = df["S"].values
        self.fret = df["E"].values
        self.first_bleach = df["_bleaches_at"].values[0]

        # TODO: check if correct and define in init
        self.y_class = df["label"].values
        self.fret_true = df["E_true"].values
        self.frames = np.arange(1, len(self.grn.int) + 1, 1)
        self.frames_max = self.frames.max()

    def get_export_df(self, keep_nan_columns: Union[bool, None] = None):
        """
        Returns the DataFrame to use for export.
        This should get passed on to get_export_txt
        """
        if keep_nan_columns is None:
            keep_nan_columns = True

        df_dict = {
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
            df_dict.update(dict(zip(self.ml_column_names, self.y_pred.T)))

        df_idx = range(len(self.fret))
        df = pd.DataFrame(df_dict, index=df_idx).round(4)

        if keep_nan_columns is False:
            df.dropna(axis=1, how="all", inplace=True)

        return df

    def get_export_txt(
        self,
        df: Optional[pd.DataFrame] = None,
        exp_txt: Optional[str] = None,
        date_txt: Optional[str] = None,
        keep_nan_columns: Union[bool, None] = None,
    ):
        """
        Returns the string to use for saving the trace as a txt.
        Option of passing a DF manually to convert it to a txt added to simplify testing.
        :param df: DataFrame to convert
        :param exp_txt: string to include in header
        :param date_txt: string to specify date.
        :param keep_nan_columns: bool, whether or not to include columns with NaNs. Passed on to get_export_df
        """
        if df is None:
            df = self.get_export_df(keep_nan_columns=keep_nan_columns)
        if exp_txt is None:
            exp_txt = "Exported by DeepFRET"
        if date_txt is None:
            date_txt = "Date: {}".format(time.strftime("%Y-%m-%d, %H:%M"))

        vid_txt = "Video filename: {}".format(self.video)
        id_txt = "FRET pair #{}".format(self.n)
        bl_txt = "Bleaches at {}".format(self.first_bleach)

        return (
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
                df.to_csv(index=False, sep="\t", na_rep="NaN"),
            )
        )

    def get_tracename(self) -> str:
        """
        Checks for and sets the tracename, if not defined, based on the current videoname and the pair number
        :return self.tracename: str
        """
        if self.tracename is None:  # define the tracename if it doesn't exist
            self.tracename = os.path.basename(
                lib.utils.remove_newlines(self.filename)
            )
        return self.tracename

    def get_savename(self, dir_to_join: Optional[str] = None):
        """
        Returns the name with which the trace should be saved.
        Option for specifying output directory.
        :param dir_to_join: output directory, optional
        :return self.savename: str
        """
        self.savename = (
            os.path.join(dir_to_join, self.get_tracename())
            if dir_to_join is not None
            else self.get_tracename()
        )
        return self.savename

    def export_trace_to_txt(
        self,
        dir_to_join: Optional[str] = None,
        keep_nan_columns: Union[bool, None] = None,
    ):
        """
        Exports the trace to the file location specified by self.savename.
        :param dir_to_join: output directory, optional, passed to get_savename
        :param keep_nan_columns: Whether to keep columns that are nan, passed to get_export_txt
        """
        savename = self.get_savename(dir_to_join=dir_to_join)
        if not savename.endswith(".txt"):
            savename = ".".join(savename.split(".")[:-1] + ["txt"])

        with open(savename, "w") as f:
            f.write(self.get_export_txt(keep_nan_columns=keep_nan_columns))

    def calculate_fret(self):
        """
        Calculates fret value for current trace
        """
        self.fret = lib.math.calc_E(self.get_intensities())

    def calculate_stoi(self):
        """
        calculates stoichiometry values for current trace.
        :return:
        """
        self.stoi = lib.math.calc_S(self.get_intensities())

    def calculate_transitions(self):
        """
        Calculates and sets the transition df (self.transitions).
        Includes lifetime of states and transitions between states.
        :return:
        """
        lf = pd.DataFrame()
        lf["state"] = self.hmm_state
        lf["e_fit"] = self.hmm
        lf["time"] = lf["e_fit"].index + 1

        # Find y_after from y_before
        lf["e_after"] = np.roll(lf["e_fit"], -1)

        # Find out when there's a change in state, depending on the minimum
        # transition size set

        lf["state_jump"] = lf["e_fit"].transform(
            lambda group: (abs(group.diff()) > 0).cumsum()
        )

        # Drop duplicates
        lf.drop_duplicates(subset="state_jump", keep="last", inplace=True)

        # Find the difference for every time
        lf["lifetime"] = np.append(np.nan, np.diff(lf["time"]))

        lf.rename(columns={"e_fit": "e_before"}, inplace=True)
        lf = lf[["e_before", "e_after", "lifetime"]]
        lf = lf[:-1]

        self.transitions = lf


class HistogramData:
    """
    Class with the data from HistogramWindow
    """

    def __init__(self):
        self.alpha = None
        self.beta = None
        self.gamma = None
        self.delta = None

        self.E = None
        self.S = None
        self.DD = None
        self.DA = None
        self.corrs = None
        self.E_un = None
        self.S_un = None

        self.lengths = None

        self.gauss_params = None
        self.best_k = None

        self.n_samples = None

        self.trace_median_len = None


class TDPData:
    def __init__(self):
        self.df = None
        self.state_lifetime = None
        self.state_before = None
        self.state_after = None


class DataContainer:
    """
    Data wrapper that contains a dict with filenames and associated data.
    """

    def __init__(self):
        self.example_traces = {}
        self.simulated_traces = {}
        self.videos = {}
        self.traces = {}
        self.currName = None
        self.histData = HistogramData()
        self.tdpData = TDPData()

    def get(self, name) -> VideoContainer:
        """Shortcut to return the metadata of selected video."""
        if self.currName is not None:
            return self.videos[name]

    def getCurrent(self) -> VideoContainer:
        """
        Shortcut for returning the metadata of video currently being loaded,
        for internal use.
        Frontend should use get() instead
        """
        if self.currName is not None:
            return self.videos[self.currName]

    def load_video_data(
        self,
        path: str,
        name: str,
        view_setup: str,
        alex: bool,
        donor_is_left: bool = True,
        donor_is_first: bool = True,
        bg_correction: bool = False,
    ):
        """
        Loads video and extracts all parameters depending on the number of
        channels

        Parameters
        ----------
        path:
            Full path of video to be loaded
        name:
            Unique identifier (to prevent videos being loaded in duplicate)
        bg_correction:
            Corrects for illumination profile and crops misaligned edges
            slightly
        """
        # Instantiate container
        self.currName = name
        self.name = name
        self.videos[name] = VideoContainer()

        video: VideoContainer = self.videos[name]
        video.traces = {}

        # Populate metadata
        ext = path.split(".")[-1]
        if ext == "fits":
            video.array = astropy.io.fits.open(path, memmap=False)[0].data
        else:
            video.array = skimage.io.imread(path)

        # If video channels are not last, make them so:
        if len(video.array.shape) == 4:
            channel_pos = int(np.argmin(video.array.shape))
            video.array = np.moveaxis(video.array, channel_pos, 3)
            video.height = video.array.shape[1]
            video.width = video.array.shape[2]

            # Interleave channels to get rid of the extra dimension
            video.array = np.hstack((video.array[..., 0], video.array[..., 1]))
            video.array = video.array.reshape((-1, video.height, video.width))

        else:
            video.height = video.array.shape[1]
            video.width = video.array.shape[2]

        # Scaling factor for ROI
        video.roi_radius = max(video.height, video.width) / 80

        # Swap channels
        _0, _1 = (0, 1) if donor_is_first else (1, 0)

        if view_setup == gvars.key_viewsetupInterleaved:
            if alex:
                # Acceptor (Dexc-Aem)
                video.acc.raw = video.array[0::4, ...]

                # ALEX (Aexc-Aem)
                video.red.raw = video.array[1::4, ...]

                # Donor (Dexc-Dem)
                video.grn.raw = video.array[2::4, ...]

                # Aexc-Dem [3::4] is ignored because it's ~0
            else:
                # Donor (Dexc-Dem)
                video.grn.raw = video.array[_0::2, ...]

                # Acceptor (Dexc-Aem)
                video.acc.raw = video.array[_1::2, ...]

        else:
            # Remove upper/lower half of quad
            if view_setup == gvars.key_viewSetupQuad:
                top, btm, lft, rgt = lib.imgdata.quadrant_indices(
                    height=video.height, width=video.width
                )
                # Figure out whether intensity is in top or bottom row, and
                # keep this only
                top_mean, btm_mean = [
                    video.array[:, idx, ...].mean(axis=(0, 1, 2))
                    for idx in (top, btm)
                ]
                video.array = (
                    video.array[:, top]
                    if top_mean > btm_mean
                    else video.array[:, btm]
                )

            # Do this for either dual/quad after preprocessing
            lft, rgt = lib.imgdata.left_right_indices(width=video.width)

            # Swap left/right
            if not donor_is_left:
                lft, rgt = rgt, lft

            if alex:
                # Donor (Dexc-Dem)
                video.grn.raw = video.array[_0::2, :, lft]

                # Acceptor (Dexc-Aem)
                video.acc.raw = video.array[_0::2, :, rgt]

                # ALEX (Aexc-Aem)
                video.red.raw = video.array[_1::2, :, rgt]

            # Ignore D/A order and assume no channels
            else:
                # Donor (Dexc-Dem)
                video.grn.raw = video.array[:, :, lft]

                # Acceptor (Dexc-Aem)
                video.acc.raw = video.array[:, :, rgt]

        for c in video.grn, video.red, video.acc:
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

        video.alex = alex
        if not alex:
            video.red = video.acc

        video.channels = video.grn, video.red
        video.indices = np.indices(video.grn.mean.shape)
