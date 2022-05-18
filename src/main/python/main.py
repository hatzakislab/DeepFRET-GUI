# coding=utf-8
import multiprocessing

from PyQt5.QtCore import qWarning

from widgets.base_window import AboutWindow, BaseWindow, PreferencesWindow
from widgets.histogram_window import HistogramWindow
from widgets.inspectors import (
    AdvancedSortInspector,
    CorrectionFactorInspector,
    DensityWindowInspector,
)
from widgets.simulator_window import SimulatorWindow
from widgets.trace_window import TraceWindow
from widgets.transition_density_window import TransitionDensityWindow
from widgets.video_window import VideoWindow

multiprocessing.freeze_support()

import sys
from typing import Union
import matplotlib
from PyQt5.QtCore import *
from configobj import ConfigObj

from global_variables import GlobalVariables as gvars

matplotlib.use("qt5agg")
import pandas as pd
from tensorflow_core.python.keras.models import load_model, Model

from fbs_runtime.application_context.PyQt5 import ApplicationContext

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 10000)

import warnings

warnings.filterwarnings("ignore")

from appdirs import user_config_dir
from pathlib import Path


class AppContext(ApplicationContext):
    """
    Entry point for running the application. Only loads resources and holds
    the event loop.
    """

    def __init__(self):
        super().__init__()

        self.keras_two_channel_model = None
        self.keras_three_channel_model = None
        self.config = None
        self.app_version = None
        self.load_resources()

    def load_resources(self):
        """
        Loads initial resources from disk to application
        """
        # model_experimental is better but undocumented
        self.keras_two_channel_model = load_model(
            self.get_resource("FRET_2C_keras_model.h5")
        )  # type: Model
        self.keras_three_channel_model = load_model(
            self.get_resource("FRET_3C_keras_model.h5")
        )  # type: Model

        userConfigFile = self.getUserConfigFile()

        self.config = ConfigObj(str(userConfigFile))

    def getUserConfigFile(self):
        """
        Creates a configuration file in `$XDG_CONFIG_HOME/DeepFRET/config.ini` (or
        similar) if no exists. Then populates it with the default values fetched
        from the default_config.ini in the executable/source.
        """
        userConfigDir = Path(user_config_dir(gvars.APPNAME, gvars.APPNAME))
        userConfigFile = userConfigDir.joinpath("config.ini")

        if not userConfigFile.exists():
            userConfigDir.mkdir(parents=True, exist_ok=True)
            userConfigFile.touch(exist_ok=False)

            tempConfig = ConfigObj(self.get_resource("default_config.ini"))
            tempConfig.filename = userConfigFile
            tempConfig.write()

        return userConfigFile

    def getConfig(self, key: str) -> Union[bool, str, float]:
        """
        Shortcut for reading config from file and returning key
        """
        self.config.reload()

        value = self.config.get(key)

        if value is None:
            qWarning(
                "{} = {} returned NoneType. Ensure that the correct value is "
                "set in the GUI".format(key, value)
            )
            return 0

        # To handle 0/1/True/False as ints
        if value in gvars.boolMaps:
            value = gvars.boolMaps[value]
        else:
            try:
                value = float(value)
            except ValueError:
                value = str(value)
        return value

    def setConfig(self, key, value):
        """
        Shortcut for writing config to file.
        """
        self.config[key] = value
        self.config.write()

    def assign(self):
        """
        Assigns resources and functions to the right windows
        before they're instantiated
        """
        # Assigns global config variables to BaseWindow and PreferencesWindow
        # so they both point to the base functions in AppContext
        for Window in (BaseWindow, PreferencesWindow):
            Window.config = self.config
            Window.processEvents = self.app.processEvents
            Window.getConfig = self.getConfig
            Window.setConfig = self.setConfig

        AboutWindow.app_version = self.build_settings["version"]
        BaseWindow.keras_two_channel_model = self.keras_two_channel_model
        BaseWindow.keras_three_channel_model = self.keras_three_channel_model

    def run(self):
        """
        Returns main loop exit code to be put into sys.exit()
        """
        return self.app.exec_()


if __name__ == "__main__":
    # Fixes https://github.com/mherrmann/fbs/issues/87
    multiprocessing.freeze_support()
    # Create the app
    # Load app
    _ctxt = AppContext()
    _ctxt.assign()

    # Windows
    _VideoWindow = VideoWindow()
    _TraceWindow = TraceWindow()
    _HistogramWindow = HistogramWindow()
    _TransitionDensityWindow = TransitionDensityWindow()
    _SimulatorWindow = SimulatorWindow()

    # Inspector sheets
    _HistogramInspector = DensityWindowInspector(_HistogramWindow)
    _TransitionDensityInspector = DensityWindowInspector(
        _TransitionDensityWindow
    )
    _CorrectionFactorInspector = CorrectionFactorInspector(_TraceWindow)
    _AdvancedSortInspector = AdvancedSortInspector(_TraceWindow)

    exit_code = _ctxt.run()
    sys.exit(exit_code)