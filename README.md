<img src="screenshots/screenshot.png" height="400">

# DeepFRET
A fully open-source, all-inclusive software platform for doing total internal reflection microscopy (TIRFm) single molecule
FRET (smFRET) fast and efficiently. The key feature is reverse-loading of ASCII traces, and *Keras/TensorFlow-powered automatic trace sorting*. Features include

- Advanced trace sorting
- Optical correction factors
- Hidden Markov model fitting and lifetime plotting
- Memory-free movie batch loading
- Distribution plotting and fitting
- Backwards-compatibility with iSMS-exported data

If you'd like to play around with just the Keras/TensorFlow model, please go to https://github.com/komodovaran/DeepFRET-Model

### How to cite
Preprint can be found at:
https://www.biorxiv.org/content/10.1101/2020.06.26.173260v1

````
DeepFRET: Rapid and automated single molecule FRET data classification using deep learning
Johannes Thomsen, Magnus B. Sletfjerding, Stefano Stella, Bijoya Paul, Simon Bo Jensen, Mette G. Malle, Guillermo Montoya, Troels C. Petersen, Nikos S. Hatzakis
bioRxiv 2020.06.26.173260; doi: https://doi.org/10.1101/2020.06.26.173260
````

## Launching the DeepFRET GUI

TODO Fix this section https://github.com/komodovaran/DeepFRET-GUI/issues/56

Tested on Python 3.6.10

From source:
1. Download the repository contents. Install requirements.txt either globally or in a venv (strongly recommended)
2. Run `src/main/python/main.py`

With the `.dmg` file:
1. Installing the `DeepFRET.app` from binary
2. Double click the `DeepFRET.app` file.

## Loading data
1. To load videos, open the application's main window ('Images')

<img src="screenshots/window_images.png" height="300">

and go to File &rightarrow; Open files. The current
version of DeepFRET supports *only* videos made with alternating laser excitation (ALEX). Donor order and appearance
can be set in Preferences. The rest of the video layout is auto-guessed by the software and doesn't allow for
fine-tuning, so be sure to set up new experiments properly!

<img src="screenshots/donor_acceptor_prefs.png" height="500">

2. If you would like to extract traces from a large number of videos that don't fit into memory, you can tick the
'batch mode' option in Preferences, as well as set a number of detections per movie. This disables the option to
interactively re-run spot detection analysis, but allows the software to process an essentially unlimited number of
videos. 

3. If you've already exported a lot of smFRET traces from another software, but want to do your analysis in DeepFRET,
you can even load in traces directly, without movies. Simply make sure that the trace window ('Traces') is the active
window, and follow the same steps as above for loading data. This works both for traces with and without ALEX.

<img src="screenshots/window_traces.png" height="300">
 
## Classifying the data
1. Go to Analyze &rightarrow; Predict to predict the trace type using deep learning. A confidence score will be given
for each trace, which tells you how certain the model is that this is a true smFRET trace.

<img src="screenshots/classification.png" height="300">
 
2. To sort traces by different things, go to the View menu. The option "advanced sort" includes, among other things,
a lower confidence threshold.

## Windows
The following shortcuts can be used to navigate the different windows, also found in the menu option `Window`:

|Window|Shortcut|
|---|---|
|Images |`⌘1`|
|Traces |`⌘2`|
|Histogram |`⌘3`|
|Transition Density |`⌘4`|
|Trace Simulator |`⌘5`|

## Images Window
- To load a video file, use the shortcut `File > Open Files` or the hotkey `⌘O`.
- To analyze and extract traces, use the options in the menu tab `Analyze`

## Traces Window
- To load traces, use the shortcut `File > Open Files` or the hotkey `⌘O`.
- To use the DeepFRET Deep Learning trace selection model, use the options in `Analyze > Predict` to predict trace types for selected or all traces.
- To fit a Hidden Markov Model to all traces, highlight all traces with `Edit > Select All` or `⌘A` and analyze by `Analyze > Fit Hidden Markov To Selected`.

## Histogram Window 
- View distribution of EFRET and Stoichiometry values for a given number of frames.
- Fit a Gaussian Mixture by defining the number or press "Auto" to use a BIC-optimized number of Gaussians.

## Transition Density Window
- View Transition Density of transitions in the Hidden Markov Model fit in the Traces Window. 
Only works if a Hidden Markov Model has been fit to the traces
- Set the number of clusters (transitions) per half of the TDP plot to extract lifetimes for each transition. 
The number of clusters per half is typically equal to the number of states.
- N.B. In order to see wider distributions, change the Preferences to Idealize traces individually. (requires restart)

## Trace Simulator Window
- Choose the parameters with which to simulate traces and press the `Refresh` button. 
- To export traces, select the number of traces to export, and press the `Export` button. 

## Statistical analysis
1. To get an overview of data distributions, go to Windows &rightarrow; Histogram. This also allows to fit the FRET
distribution with gaussians to estimate the number of underlying conformational states. These plots update with the
number of traces selected. Because a large number of operations have to be re-computed every time a trace is
selected/de-selected, it can be a bit slow if the window is left open while traces are being selected.

2. Additionally, DeepFRET includes a Hidden Markov model with the possibility to fit each trace individually
(as a smoothing, "step-finding" method), as well as a global fit option, for a true Hidden Markov model fit. This also
automatically fits transition lifetimes and plots transition density plots (which can be found in
Windows &rightarrow; Transition Density Plot). One can base the model on the donor/acceptor signal, or directly on the
FRET signal (though this is less accurate)

The whole process should be cross-platform, but has only been tested on MacOS.

## Supported data formats
* Traces exported by DeepFRET or iSMS can be loaded by opening the Traces window, and then using the Open Files in the menu.
* Other traces must support some, or all of the following format:

    |Column|Meaning|
    |---|---|
    |`D-Dexc-rw` | Donor excitation, donor emission signal|
    |`A-Dexc-rw` | Donor excitation, acceptor emission signal|
    |`A-Aexc-rw` | Acceptor excitation, acceptor emission signal|
    |`D-Dexc-bg` | Donor excitation, donor emission background|
    |`A-Dexc-bg` | Donor excitation, acceptor emission background|
    |`A-Aexc-bg` | Acceptor excitation, acceptor emission background|

* CSV files are supported with variable empty spaces between columns (1, 2, tabs, etc)
At the very minimum, your trace must have a `D-Dexc-` column first, as DeepFRET seeks for this to find the rest.
Background columns can be ommitted or set to 0, and missing `A-Aexc-` columns can be either ommitted or set to NaN (do not set `A-Aexc-rw` to 0, as this indicates the presence of a channel).
At the very minimum, DeepFRET is able to load a trace with just the columns `D-Dexc-rw` and `A-Dexc-rw`.
Other columns not in the specification are simply ignored.

<img src="screenshots/trace.png" height="200">

* When saving traces, they will outputted in DeepFRET format.

* DeepFRET does not support time columns, and works on a per-frame basis.

* Currently, .tiff videos are supported as either overlaid full frames in the correct sequence or 2-channel QuadView.
They can be loaded by choosing Open Files when the Images window is active. If batch loading is enable in the preferences,
you will be able to load an "unlimited" amount of videos and extract a set number of particles, with all video interactivity disabled.

![video_seq](video_seq.png)
* If you're having trouble with your image format, please file an issue and we'll try to add compatibility.

## Modifying and compiling the DeepFRET GUI to a standalone executable:

TODO Fix this section https://github.com/komodovaran/DeepFRET-GUI/issues/56

1. Download all contents to a directory.
2. Open a terminal and navigate to the root of the directory.
3. Create a venv with `python3 -m venv venv` in the current directory.
4. Activate environment with `source venv/bin/activate` if on MacOS/Linux or `call venv\scripts\activate.bat` if on Windows.
5. While still in the environment, install all packages with `pip install -r requirements.txt`
6. Unzip the `hooks.zip` and overwrite the files in `venv/lib/python3.7/site-packages/PyInstaller/hooks/`.
7. Run `compile.py`

If the above steps worked, you can now edit any part of the code, and re-compile it (or just run it from the main.py
script, if desired). The `.ui` files for the interface can be edited through Qt Creator and converted with `generate_ui.py`

<img src="screenshots/sorting.png" height="200">


# Development

## Development environment

This is only needed if you want to change the code. If you only want to run
DeepFRET-GUI, see Install. (TODO add link.
https://github.com/komodovaran/DeepFRET-GUI/issues/56)

### Windows

#### Install dependencies

This needs to be done only once for the machine. The optional steps is needed if
you want to use `fbs` to make a new release. If you only want to change the
python code, you can skip them.

1. Python:
    1. Download and open the python 3.6 from [the python
       webpage](https://www.python.org/downloads/windows/). Version 3.6.8 is the
       latest micro version with an executable installer for Windows.
    2. During the installation wizard make sure to check the "Add Python 3.6 to
       PATH" option.
2. Git from [the git webpage](https://git-scm.com/download/win).
3. (Optional) Microsoft dependencies:
    1. Install [Windows 10
       SDK](https://developer.microsoft.com/en-us/windows/downloads/windows-10-sdk).
    2. Install [Visual C++ Redistributable for Visual Studio
       2010](https://www.microsoft.com/en-us/download/details.aspx?id=14632).
    3. Install [Visual C++ Redistributable for Visual Studio
       2012](https://www.microsoft.com/en-us/download/details.aspx?id=30679).
    4. If `fbs freeze` complains about missing DDLs and tells you to install
       Visual C++ Redistributable for Visual Studio 2012, it could be actually
       be another version. See [fbs issue
       147](https://github.com/mherrmann/fbs/issues/147) for details.
4. (Optional) Nullsoft Install System:
    1. Download and run the installer from [NSIS](https://nsis.sourceforge.io/Download).
    2. The installer does not add the executables like `makensis` to the path.
    You need to add `C:\Program Files (x86)\NSIS` to the path. You can add it
    permanently by following [this
    guide](https://www.architectryan.com/2018/03/17/add-to-the-path-on-windows-10/).
    Alternatively, you can execute `$env:PATH += "C:\Program Files (x86)\NSIS;"`
    in the PowerShell to set it for the session of the PowerShell.
5. Normal users on Windows is missing the privileged to execute PowerShell
   scripts. This is needed to enable the Python virtual environment. Open a
   PowerShell and run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy
   Unrestricted`. Answer `Y` when prompted.
6. (Optional) If you have installed Windows 10 SDK, a C++ Redistributable or
   NSIS: Restart the machine.

#### Get the code and install python requirements

1. Open a PowerShell.
2. Clone the repo with `git clone https://github.com/komodovaran/DeepFRET-GUI.git`.
3. Change into the directory with `cd .\DeepFRET-GUI\`.
3. Create a python virtual environment with `python -m venv venv`.
4. Activate the virtual environment with `.\venv\Scripts\Activate.ps1`.
5. Install the python requirements with `pip install -r requirements-win.txt`.

You should now be able to open the application with `python
.\src\main\python\main.py`.

## Release

TODO Expand this section. https://github.com/komodovaran/DeepFRET-GUI/issues/56

1. Bump the version in `src/build/settings/base.json`
2. Use fbs to package the installer with:
    1. `fbs clean`
    2. `fbs freeze`
    3. `fbs installer`

If the correct `fbs` is not in your path, you can call it directly. On Unix, it
at `./venv/bin/fbs`. On Windows it is at `.\venv\Scripts\fbs.exe`.

## Requirements files and pip-tools

TODO Expand this section. https://github.com/komodovaran/DeepFRET-GUI/issues/56

## About fbs and PyInstaller

We want inject some extra hooks into PyInstaller. Unfortunately `fsb` does not
expose a way to do that. There is a [pull
request](https://github.com/mherrmann/fbs/pull/157) by NileGraddis that
introduces such a functionality, but the author of `fsb`has decided not to
implement it. We therefore use [the fork by
NileGraddis](https://github.com/NileGraddis/fbs/tree/additional_hooks_dir). It
enables us to define `additional_hooks_dir` in `src/build/settings/base.json`.

fsb have
[pinned](https://github.com/mherrmann/fbs/commit/84fe1bc3fb9369000abe03c3c3bc133693d8d9ff)
PyInstaller to version 3.4 [due to some
incompatibilities](https://github.com/mherrmann/fbs/issues/169). The NileGraddis
fork we use does not have the pinning of PyInstaller in it, so we need to pin it
manually. There is [a issue](https://github.com/mherrmann/fbs/issues/188)
tracking the support for PyInstaller 3.6. This pinning can be removed when there
is no reason to use for fork anymore or fsb starts supporting PyInstaller 3.6
and the fork is updated.

