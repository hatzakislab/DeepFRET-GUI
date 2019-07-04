## How to launch the DeepFRET GUI

Option 1:
This option is the easiest for cross-platform compatibility
a. Download the repository contents. Install requirements.txt either globally or in a venv (strongly recommended)
b. Launch src/main/python/main.py


Option 2:
Download a pre-compiled application (currently only available for MacOS)


## How to modify the DeepFRET GUI:

1. Initiate fbs environment (tutorial)
2. Enter the venv and install the the requirements.txt
3. Download the pyinstaller hooks and overwrite venv/lib/python3.7/site-packages/PyInstaller/hooks/
4. Overwrite the /src with the one in this repository
5. While still in the venv, write fbs freeze and wait for the process to finish. If everything went well, there will be a target/ directory with the DeepFRET application inside.

This build process should be cross-platform, but has only been tested on MacOS
