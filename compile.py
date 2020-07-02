import os
import json
from configobj import ConfigObj
from src.main.python.lib.utils import git_app_version
import zipfile
import sys
import warnings


def check_python_ver():
    class CompilationWarning(Warning):
        pass

    if sys.version.split(" ")[0] != "3.7.0b1":
        warnings.warn(
            "Compiling is only tested to work on 3.7.0b1. "
            "You seem to have a different version. ",
            stacklevel=2,
            category=CompilationWarning,
        )
        input("Press enter to continue anyway: ")


def patch_pyinstaller():
    with zipfile.ZipFile("patch.zip", "r") as zip_ref:
        zip_ref.extractall("venv/lib/python3.7/site-packages/PyInstaller/")


def write_to_config_ini():
    config = ConfigObj("src/main/resources/base/config.ini")
    config["appVersion"] = int(git_app_version())
    config.write()


def write_to_base_json():
    with open("src/build/settings/base.json", "r+") as base_json:
        data = json.load(base_json)
        data["version"] = str(git_app_version()).rstrip("\n")

        base_json.seek(0)
        base_json.write(json.dumps(data, indent=4, sort_keys=True))
        base_json.truncate()


def compile():
    os.system("fbs clean")
    os.system("fbs freeze")
    os.system("fbs installer")


if __name__ == "__main__":
    check_python_ver()
    patch_pyinstaller()
    write_to_config_ini()
    write_to_base_json()
    compile()
