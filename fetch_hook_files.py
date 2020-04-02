import os
import shutil

cwd = os.getcwd()

hooks_dir = os.path.join(cwd, "venv/lib/python3.7/site-packages/PyInstaller/hooks")

shutil.make_archive(os.path.join(cwd, "hooks"), 'zip', hooks_dir)