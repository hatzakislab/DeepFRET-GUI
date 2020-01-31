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

### How to cite:
Publication coming soon!

#### Launching the DeepFRET GUI
See below for information on supported data formats.

##### Option A:
This option is the easiest for cross-platform compatibility
1. Download the repository contents. Install requirements.txt either globally or in a venv (strongly recommended)
2. Run `src/main/python/main.py`

##### Option B:
Download a pre-compiled application (currently only available for MacOS)

#### Modifying and compiling the DeepFRET GUI:
1. Download all contents to a directory.
2. Open a terminal and navigate to the root of the directory.
3. Create a venv with `python3 -m venv venv` in the current directory.
4. Activate environment with `source venv/bin/activate` if on MacOS/Linux or `call venv\scripts\activate.bat` if on Windows.
5. While still in the environment, install all packages with `pip install -r requirements.txt`
6. Unzip the `hooks.zip` and overwrite the files in `venv/lib/python3.7/site-packages/PyInstaller/hooks/`.
7. While still in the venv, write `fbs freeze` and wait for the process to finish. If everything went well, there will be a `target/` directory with the DeepFRET application inside.

If the above steps worked, you can now edit any part of the code, and re-compile it (or just run it from the main.py script, if desired). The `.ui` files for the interface can be edited through Qt Creator and converted with `generate_ui.py` 

The whole process should be cross-platform, but has only been tested on MacOS.


#### Supported data formats
* Traces exported by DeepFRET or iSMS can be loaded by opening the Traces window, and then using the Open Files in the menu.
* Currently, .tiff videos are supported as either overlaid full frames in the correct sequence or 2-channel QuadView. They can be loaded by choosing Open Files when the Images window is active. If batch loading is enable in the preferences, you will be able to load an "unlimited" amount of videos and extract a set number of particles, with all video interactivity disabled.

![video_seq](video_seq.png)
* If you're having trouble with your image format, please file an Issue and I'll add compatibility.
