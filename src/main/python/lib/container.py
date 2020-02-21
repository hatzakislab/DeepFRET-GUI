import multiprocessing
import time

multiprocessing.freeze_support()

from global_variables import GlobalVariables as gvars
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import skimage.io
from lib import imgdata
from typing import Union, Tuple


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
        self.blu = ImageChannel("blue")
        self.grn = ImageChannel("green")
        self.red = ImageChannel("red")
        self.acc = ImageChannel("red")

        # Colocalized channels
        self.coloc_blu_grn = ColocalizedParticles("blue", "green")
        self.coloc_blu_red = ColocalizedParticles("blue", "red")

        self.coloc_grn_red = ColocalizedParticles("green", "red")
        self.coloc_all = ColocalizedAll()

        # Blended channels
        self.grn_red_blend = None  # type: Union[None, np.ndarray]

        # Create a class to store traces and associated information for every image name
        self.traces = {}

        # To iterate over all channels
        self.all = (self.grn, self.red, self.coloc_grn_red, self.coloc_all)


class ImageChannel:
    """
    Class for storing image channel information.
    """

    def __init__(self, color):

        if color == "blue":
            cmap = LinearSegmentedColormap.from_list(
                "", ["black", gvars.color_blue]
            )
        elif color == "green":
            cmap = LinearSegmentedColormap.from_list(
                "", ["black", gvars.color_green]
            )
        elif color == "red":
            cmap = LinearSegmentedColormap.from_list(
                "", ["black", gvars.color_red]
            )
        else:
            raise ValueError(
                "Invalid color. Available options are 'green' or 'red' (or 'blue' for compatibility reasons)."
            )

        self.exists = True  # type: bool
        self.cmap = cmap  # type: LinearSegmentedColormap
        self.color = color  # type: str
        self.raw = None  # type: Union[None, np.ndarray]
        self.mean = None  # type: Union[None, np.ndarray]
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

    def __init__(self, name, movie, n):
        self.name = name  # type: str
        self.movie = movie  # type: str
        self.n = n  # type: str
        self.tracename = None  # type: Union[None, str]


        self.is_checked = False  # type: bool
        self.xdata = []  # type: [int, int]

        # self.blu = TraceChannel("blue")
        self.grn = TraceChannel("green")
        self.red = TraceChannel("red")
        self.acc = TraceChannel("red")

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

        self.channels = self.grn, self.red, self.acc


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

    def get_export_df(self):
        """
        Returns the DataFrame to use for export
        """
        if self.y_pred is None:
            df = pd.DataFrame(
                {
                    "D-Dexc-bg": self.grn.bg,
                    "A-Dexc-bg": self.acc.bg,
                    "A-Aexc-bg": self.red.bg,
                    "D-Dexc-rw": self.grn.int,
                    "A-Dexc-rw": self.acc.int,
                    "A-Aexc-rw": self.red.int,
                    "S": self.stoi,
                    "E": self.fret,
                }
            ).round(4)
        else:
            df = pd.DataFrame(
                {
                    "D-Dexc-bg": self.grn.bg,
                    "A-Dexc-bg": self.acc.bg,
                    "A-Aexc-bg": self.red.bg,
                    "D-Dexc-rw": self.grn.int,
                    "A-Dexc-rw": self.acc.int,
                    "A-Aexc-rw": self.red.int,
                    "S": self.stoi,
                    "E": self.fret,
                    "p_blch": self.y_pred[:, 0],
                    "p_aggr": self.y_pred[:, 1],
                    "p_stat": self.y_pred[:, 2],
                    "p_dyna": self.y_pred[:, 3],
                    "p_nois": self.y_pred[:, 4],
                    "p_scrm": self.y_pred[:, 5],
                }
            ).round(4)
        return df

    def get_export_txt(self, df: Union[None, pd.DataFrame] = None, exp_txt: Union[None, str] = None,
                       date_txt: Union[None, str] = None):
        """
        Returns the string to use for saving the trace as a txt
        """
        if df is None:
            df = self.get_export_df()
        if (exp_txt is None) or (date_txt is None):
            exp_txt = "Exported by DeepFRET"
            date_txt = "Date: {}".format(time.strftime("%Y-%m-%d, %H:%M"))

        mov_txt = "Movie filename: {}".format(self.movie)
        id_txt = "FRET pair #{}".format(self.n)
        bl_txt = (
            "Donor bleaches at: {} - "
            "Acceptor bleaches at: {}".format(
                self.grn.bleach, self.red.bleach
            )
        )
        out_txt = "{0}\n" \
                  "{1}\n" \
                  "{2}\n" \
                  "{3}\n" \
                  "{4}\n\n" \
                  "{5}".format(
            exp_txt,
            date_txt,
            mov_txt,
            id_txt,
            bl_txt,
            df.to_csv(index=False, sep="\t"))

        return out_txt

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
        Shortcut for returning the metadata of movie currently being loaded, for internal use.
        Frontend should use get() instead
        """
        if self.currName is not None:
            return self.movies[self.currName]

    def load_img(self, path, name, setup, bg_correction):
        """
        Loads movie and extracts all parameters depending on the number of channels

        Parameters
        ----------
        path:
            Full path of video to be loaded
        name:
            Unique identifier (to prevent videos being loaded in duplicate)
        setup:
            Imaging setup from config
        bg_correction:
            Corrects for illumination profile and crops misaligned edges slightly
        """
        # Instantiate container
        self.currName = name
        self.name = name
        self.movies[name] = ImageContainer()
        self._data().traces = {}
        # Populate metadata
        self._data().img = skimage.io.imread(path)

        if setup != "dual" and self._data().img.shape[3] > 3:
            self._data().img = (
                self._data().img.swapaxes(3, 1).swapaxes(2, 1)
            )

        self._data().height = self._data().img.shape[1]
        self._data().width = self._data().img.shape[2]
        # Scaling factor for ROI
        self._data().roi_radius = (
                max(self._data().height, self._data().width) / 80
        )

        if setup == "dual":
            c1, c2, c3, _ = imgdata.image_channels(4)

            # Acceptor   (Dexc-Aem)
            self._data().acc.raw = self._data().img[c1, :, :]
            # ALEX  (Aexc-Aem)
            self._data().red.raw = self._data().img[c2, :, :]
            # Donor (Dexc-Dem)
            self._data().grn.raw = self._data().img[c3, :, :]

            self._data().channels = self._data().grn, self._data().red

        elif setup in ["2-color", "3-color"]:
            top, btm, lft, rgt = imgdata.image_quadrants(
                height=self._data().height, width=self._data().width
            )

            if (
                    self._data().img.shape[3] == 3
                    and self._data().img.shape[0] < 99
            ):
                # self._data().blu.raw = self._data().img[:, btm, lft, 0]
                self._data().grn.raw = self._data().img[:, top, lft, 1]
                self._data().red.raw = self._data().img[:, top, rgt, 2]
                self._data().acc.raw = self._data().img[:, top, rgt, 1]

                self._data().channels = (
                    # self._data().blu,
                    self._data().grn,
                    self._data().red,
                )

            # for FRET movies, channels ordered (green, red, blue)
            elif (
                    self._data().img.shape[3] == 3
                    and self._data().img.shape[0] > 99
            ):
                # self._data().blu.raw = self._data().img[:, btm, lft, 2]
                self._data().grn.raw = self._data().img[:, top, lft, 0]
                self._data().red.raw = self._data().img[:, top, rgt, 1]
                self._data().acc.raw = self._data().img[:, top, rgt, 0]

                self._data().channels = (
                    # self._data().blu,
                    self._data().grn,
                    self._data().red,
                )

            # if recording standard ALEX FRET with no blue
            elif self._data().img.shape[3] == 2:
                self._data().grn.raw = self._data().img[:, top, lft, 0]
                self._data().acc.raw = self._data().img[:, top, rgt, 0]
                self._data().red.raw = self._data().img[:, top, rgt, 1]

                self._data().channels = self._data().grn, self._data().red

                # self._data().blu.exists = False

        elif setup == "2-color-inv":
            top, btm, lft, rgt = imgdata.image_quadrants(
                height=self._data().height, width=self._data().width
            )

            # No support for blue!
            if self._data().img.shape[3] == 2:
                self._data().grn.raw = self._data().img[:, btm, rgt, 0]
                self._data().acc.raw = self._data().img[:, btm, lft, 0]
                self._data().red.raw = self._data().img[:, btm, lft, 1]

                self._data().channels = self._data().grn, self._data().red
                # self._data().blu.exists = False

            else:
                raise ValueError("Format not supported.")

        elif setup == "bypass":
            self._data().grn.raw = self._data().img[:, :, :, 0]
            self._data().red.raw = self._data().img[:, :, :, 1]
            # self._data().blu.raw = self._data().img[:, :, :, 2]

            self._data().channels = (
                # self._data().blu,
                self._data().grn,
                self._data().red,
            )

            # self._data().blu.exists = True
            self._data().red.exists = True
            self._data().acc.exists = False

        else:
            raise ValueError("Invalid value. Config.ini corrupted?")

        for c in self._data().channels + (self._data().acc,):
            if c.raw is not None:
                c.raw = np.abs(c.raw)
                t, h, w = c.raw.shape

                if bg_correction:
                    # Crop 2% of sides to avoid messing up background detection
                    crop_h = int(h // 50)
                    crop_w = int(w // 50)

                    c.raw = c.raw[:, crop_h: h - crop_h, crop_w: w - crop_w]
                    c.mean = c.raw[0: t // 20, :, :].mean(axis=0)
                    c.mean = imgdata.zero_one_scale(c.mean)
                    c.mean_nobg = imgdata.subtract_background(
                        c.mean, by="row", return_bg_only=False
                    )
                else:
                    c.mean = imgdata.zero_one_scale(c.mean)
                    c.mean_nobg = c.mean

        if self._data().red.exists:
            self._data().indices = np.indices(self._data().red.mean.shape)
        else:
            self._data().indices = np.indices(self._data().grn.mean.shape)
