#### How to launch the DeepFRET GUI

##### Option 1
This option is the easiest for cross-platform compatibility
1. Download the repository contents. Install requirements.txt either globally or in a venv (strongly recommended)
2. Launch src/main/python/main.py

##### Option 2:
Download a pre-compiled application (currently only available for MacOS)


#### How to modify the DeepFRET GUI:
1. Create a venv with `python3 -m venv venv` in the current directory.
2. Activate environment with `source venv/bin/activate` if on MacOS/Linux or `call venv\scripts\activate.bat`
3. Install fbs and PyQt5 with `pip install fbs PyQt5==5.9.2`
4. Install all other packages with `pip install requirements.txt -r`
5. Download the pyinstaller hooks and overwrite those in `venv/lib/python3.7/site-packages/PyInstaller/hooks/`
6. Overwrite the /src with the one in this repository
7. While still in the venv, write `fbs freeze` and wait for the process to finish. If everything went well, there will be a `target/` directory with the DeepFRET application inside.

This build process should be cross-platform, but has only been tested on MacOS
