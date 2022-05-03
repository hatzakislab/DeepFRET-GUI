# Steps to reproduce environment as per 3/5/2022

At the time of writing, the free version of fbs only supports up to python 3.6, whereas tensorflow for M1 macs requires python 3.8. As such, it was necessary to buy a one-time license in order to create the installer. The program could still be run through python with the free version of fbs.

Use miniforge conda for environment setup. Set up an environment with python 3.8. `conda create -n env_name python=3.8`

Install tensorflow for m1 as described in https://caffeinedev.medium.com/how-to-install-tensorflow-on-m1-mac-8e9b91d93706

`conda install -c apple tensorflow-deps`

`pip install tensorflow-macos`

`pip install tensorflow-metal`

Install pyqt5 specifically with:

`conda install -c conda-forge pyqt`

This is the only way I have gotten pyqt5 to work with my mac.

Install other dependencies with `pip install -r M1_requirements/pip_requirements.txt`

Install free fbs with `pip install fbs` Or use the tar provided upon purchase.

Run inject_hooks.py from the top level folder in order to inject the custom hooks in `src/build/pyinstaller-hooks` to the fbs installation. If 

`python inject-hooks.py`

`python inject-hooks.py -d` can optionally be run after building the installer in order to clean up the fbs installation.

Check that the program can be run by running it through python

`python src/main/python/main.py`

And fbs (command has to be run from top-level folder)

`fbs run`

If these two succeed, a binary can be built with

`fbs freeze`

And an installer with

`fbs installer`

At the time of writing, `fbs installer` fails at a late step due to a new version of macos conflicting with `create-dmg`. My fix was to update create-dmg with `brew install create-dmg`. I then extracted the command that failed when calling `fbs installer` and manually ran the `create-dmg` with my system-wide `create-dmg` installation instead of the one packaged with fbs. Again run from the top-level folder:

`create-dmg --no-internet-enable --volname DeepFRET --app-drop-link 170 10 --icon DeepFRET.app 0 10 target/DeepFRET.dmg target/DeepFRET.app`

The app is packaged into an installable .dmg!