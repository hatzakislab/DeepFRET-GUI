from dataclasses import dataclass
from collections import OrderedDict


@dataclass
class GlobalVariables:
    """
    Store keys to obtain default values here, and some hardcoded defaults for plots.
    """

    APPNAME = "DeepFRET"
    AUTHORS = "created by Johannes Thomsen"
    LICENSE = "DeepFRET is free software, distributed under the terms of the MIT open source license"
    CONFIGNAME = "config.ini"

    # Window instance keys
    VideoWindow = "VideoWindow"
    TraceWindow = "TraceWindow"
    TransitionDensityWindow = "TransitionDensityWindow"
    HistogramWindow = "HistogramWindow"

    # Inspector instance keys
    CorrectionFactorInspector = "CorrectionFactorInspector"
    AdvancedSortInspector = "AdvancedSortInspector"
    DensityWindowInspector = "DensityWindowInspector"

    # Bool maps for configs to avoid errors between strings/ints/bools
    boolMaps = {"True": 1, "1": 1, "False": 0, "0": 0}

    # List of possible user-configurable options (i.e. available keys),
    # so they'll be available with autocomplete and refactoring
    # States are edited through the getConfig interface
    key_batchLoadingMode = "batchLoadingMode"
    key_unColocRed = "unColocRed"  # Uncolocalized red
    key_illuCorrect = "illuCorrect"  # Illumination profile correction
    key_fitSpots = "fitSpots"  # Whether to use LoG-based spot detection

    key_hmmBICStrictness = "hmmBICStrictness"
    key_hmmLocal = (
        "hmmLocal"  # Whether to use global or local hmm, gets passed to traces
    )

    key_firstFrameIsDonor = "firstFrameIsDonor"
    key_donorLeft = "donorLeft"
    key_medianPearsonCorr = "medianPearsonCorr"
    key_lagsPearsonCorr = "lagsPearsonCorr"

    keys_globalCheckBoxes = (
        key_batchLoadingMode,
        key_unColocRed,
        key_illuCorrect,
        key_fitSpots,
        key_hmmLocal,
        key_firstFrameIsDonor,
        key_donorLeft,
        key_medianPearsonCorr,
    )

    key_hmmMode = "hmmMode"
    key_hmmModeE = "E"
    key_hmmModeDA = "DA"
    keys_hmmModes = key_hmmModeE, key_hmmModeDA

    key_spotDetection = "spotDetection"  # Spot detection speed
    key_lastOpenedDir = "lastOpenedDir"  # Sets recent directory

    key_HmmModeE = "E"
    key_HmmModeDD = "DD"

    keys_HmmModes = key_HmmModeE, key_HmmModeDD

    # MainWindow
    key_contrastBoxHiBluVal = "contrastBoxHiBluVal"
    key_contrastBoxHiGrnVal = "contrastBoxHiGrnVal"
    key_contrastBoxHiRedVal = "contrastBoxHiRedVal"

    keys_contrastBoxHiVals = (
        key_contrastBoxHiBluVal,
        key_contrastBoxHiGrnVal,
        key_contrastBoxHiRedVal,
    )

    key_alphaFactor = "alphaFactor"
    key_deltaFactor = "deltaFactor"

    key_colocTolerance = "colocTolerance"
    key_autoDetectPairs = "autoDetectPairs"

    # TraceWindow
    key_traceStoiLo = "traceStoiLo"
    key_traceStoiHi = "traceStoiHi"
    key_traceFretLo = "traceFretLo"
    key_traceFretHi = "traceFretHi"
    key_traceMinFrames = "traceMinFrames"

    keys_trace = (
        key_traceStoiLo,
        key_traceStoiHi,
        key_traceFretLo,
        key_traceFretHi,
        key_traceMinFrames,
    )

    # TransitionDensityWindow
    key_tdpBandwidth = "tdpBandwidth"
    key_tdpResolution = "tdpResolution"
    key_tdpColors = "tdpColors"
    key_tdpOverlayPts = "tdpOverlayPts"
    key_tdpPtsAlpha = "tdpPtsAlpha"

    keys_tdp = (
        key_tdpBandwidth,
        key_tdpResolution,
        key_tdpColors,
        key_tdpOverlayPts,
        key_tdpPtsAlpha,
    )

    # HistogramWindow
    key_histBandwidth = "histBandwidth"
    key_histResolution = "histResolution"
    key_histColors = "histColors"
    key_histOverlayPts = "histOverlayPts"
    key_histPtsAlpha = "histPtsAlpha"

    keys_hist = (
        key_histBandwidth,
        key_histResolution,
        key_histColors,
        key_histOverlayPts,
        key_histPtsAlpha,
    )

    # ROIs
    roi_math_radius = 5  # How large ROI circles are, mathematicaly
    roi_draw_radius = (
        7  # How large circles are, visually (has no impact on colocalization)
    )
    roi_draw_linewidth = 0.9  # ROI linewidth
    roi_inner_area = (
        2.3  # Inner area of ROI pixel intensity mask (adjusted to iSMS)
    )
    roi_outer_area = (
        5.6  # Outer area of ROI pixel intensity mask (adjusted to iSMS)
    )
    roi_gap_space = (
        4.4  # Gap space of ROI pixel intensity mask (adjusted to iSMS)
    )
    roi_coloc_overlap_factor = (
        2  # Degree of overlap before it counts (this is roughly 90%)
    )

    roi_coloc_tolerances = {"loose": 2, "moderate": 5, "strict": 20}

    # Parameters for functions
    peak_local_max_p = {
        "min_distance": 5
    }  # Spot finder parameters for localization

    bg_p = {"alpha": 0.5, "linestyle": "--"}

    cmask_p = {
        "inner_area": roi_inner_area,
        "outer_area": roi_outer_area,
        "gap_space": roi_gap_space,
    }

    circle_p = {
        "linewidth": 1,
        "alpha": 1,
        "fill": False,
        "radius": roi_draw_radius,
    }

    # Colors
    color_gui_bg = "#ECECEC"
    color_hud_white = "#DCDDDC"
    color_hud_black = "#1F2022"
    color_gui_text = "darkgrey"

    color_circle_roi = "black"
    color_coloc_roi = "black"

    color_grey = "darkgrey"
    color_green = "seagreen"
    color_red = "orangered"
    color_orange = "#FA9B3D"
    color_blue = "royalblue"
    color_purple = "#BC3587"
    color_cyan = "cyan"
    color_yellow = "yellow"
    color_white = "white"

    # For model data and classification
    model_classes_full = OrderedDict(
        (
            (0, "bleached"),
            (1, "aggregated"),
            (2, "noisy"),
            (3, "scrambled"),
            (4, "1-state"),
            (5, "2-state"),
            (6, "3-state"),
            (7, "4-state"),
            (8, "5-state"),
        )
    )

    model_colors_full = OrderedDict(
        (
            (0, "darkgrey"),
            (1, "red"),
            (2, "blue"),
            (3, "purple"),
            (4, "orange"),
            (5, "lightgreen"),
            (6, "green"),
            (7, "mediumseagreen"),
            (8, "darkolivegreen"),
        )
    )
