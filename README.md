![screenshot](screenshot.png)

### DeepFRET
A fully open-source, all-inclusive software platform for doing total internal reflection microscopy (TIRFm) single molecule FRET (smFRET) fast and efficiently. The key feature is reverse-loading of ASCII traces, and *Keras/TensorFlow-powered automatic trace sorting*. Features include

- Advanced trace sorting
- Optical correction factors
- Hidden Markov model fitting and lifetime plotting
- Memory-free movie batch loading
- Distribution plotting and fitting
- Backwards-compatibility with iSMS-exported data

If you'd like to play around with just the Keras/TensorFlow model, please go to https://github.com/komodovaran/DeepFRET-Model

### How to cite
Publication coming soon!

### Launching the DeepFRET GUI
From source:
1. Download the repository contents. Install requirements.txt either globally or in a venv (strongly recommended)
2. Run `src/main/python/main.py`

With the `.dmg` file:
1. Installing the `DeepFRET.app` from binary
2. Double click the `DeepFRET.app` file.

### Using DeepFRET

#### Windows
The following shortcuts can be used to navigate the different windows, also found in the menu option `Window`:

|Window|Shortcut|
|---|---|
|Images |`⌘1`|
|Traces |`⌘2`|
|Histogram |`⌘3`|
|Transition Density |`⌘4`|
|Trace Simulator |`⌘5`|

##### Images Window
- To load a video file, use the shortcut `File > Open Files` or the hotkey `⌘O`.
- To analyze and extract traces, use the options in the menu tab `Analyze`

##### Traces Window
- To load traces, use the shortcut `File > Open Files` or the hotkey `⌘O`.
- To use the DeepFRET Deep Learning trace selection model, use the options in `Analyze > Predict` to predict trace types for selected or all traces.
- To fit a Hidden Markov Model to all traces, highlight all traces with `Edit > Select All` or `⌘A` and analyze by `Analyze > Fit Hidden Markov To Selected`.

##### Histogram Window 
- View distribution of EFRET and Stoichiometry values for a given number of frames.
- Fit a Gaussian Mixture by defining the number or press "Auto" to use a BIC-optimized number of Gaussians.


##### Transition Density Window
- View Transition Density of transitions in the Hidden Markov Model fit in the Traces Window. 
Only works if a Hidden Markov Model has been fit to the traces
- Set the number of clusters (transitions) per half of the TDP plot to extract lifetimes for each transition. 
The number of clusters per half is typically equal to the number of states.
- N.B. In order to see wider distributions, change the Preferences to Idealize traces individually. (requires restart)

##### Trace Simulator Window
- Choose the parameters with which to simulate traces and press the `Refresh` button. 
- To export traces, select the number of traces to export, and press the `Export` button. 
 
### Modifying and compiling the DeepFRET GUI to a standalone executable:
1. Download all contents to a directory.
2. Open a terminal and navigate to the root of the directory.
3. Create a venv with `python3 -m venv venv` in the current directory.
4. Activate environment with `source venv/bin/activate` if on MacOS/Linux or `call venv\scripts\activate.bat` if on Windows.
5. While still in the environment, install all packages with `pip install -r requirements.txt`
6. Unzip the `hooks.zip` and overwrite the files in `venv/lib/python3.7/site-packages/PyInstaller/hooks/`.
7. Run `compile.py`

If the above steps worked, you can now edit any part of the code, and re-compile it (or just run it from the main.py script, if desired). The `.ui` files for the interface can be edited through Qt Creator and converted with `generate_ui.py` 

The whole process should be cross-platform, but has only been tested on MacOS.


### Supported data formats
* Traces exported by DeepFRET or iSMS can be loaded by opening the Traces window, and then using the Open Files in the menu.
* Currently, .tiff videos are supported as either overlaid full frames in the correct sequence or 2-channel QuadView. They can be loaded by choosing Open Files when the Images window is active. If batch loading is enable in the preferences, you will be able to load an "unlimited" amount of videos and extract a set number of particles, with all video interactivity disabled.

![video_seq](video_seq.png)
* If you're having trouble with your image format, please file an Issue and I'll add compatibility.
