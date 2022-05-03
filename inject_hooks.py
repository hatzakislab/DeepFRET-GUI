import fbs.freeze.hooks
import os
import argparse

parser = argparse.ArgumentParser("Inject hooks into fbs directory, or remove them afterwards")
parser.add_argument("-d", "--delete", action="store_true", help = "Deletes the files from the fbs directory rather than inserting them")
args = parser.parse_args()

_to = fbs.freeze.hooks.__file__
_to = _to.replace("/__init__.py", "")
hookdir = "src/build/pyinstaller-hooks"
files = [file for file in os.listdir(hookdir) if file.endswith(".py") and file != "__init__.py"]

if not args.delete:
    for file in files:
        _from = os.path.join(hookdir, file)
        cmd = f"cp {_from} {_to}"
        os.system(cmd)
else:
    for file in files:
        _dlt = os.path.join(_to, file)
        cmd = f"rm {_dlt}"
        os.system(cmd)