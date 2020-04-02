import os
import glob
from os.path import join, dirname, basename


def generatePyFromUi(verbose=True):
    """Add Qt .interface Example_Files here to dynamically convert to .py on
    launch"""
    ui_files = glob.glob("ui/*.ui")

    for filename in ui_files:
        if os.path.exists(filename):
            out_name = (
                join(dirname(filename), "_" + basename(filename)).rstrip(".ui")
            )

            if verbose:
                print("{} was converted".format(filename))
            os.system(
                "exec python3 -m PyQt5.uic.pyuic {0} -o {1}.py -x".format(
                    filename, out_name
                )
            )


if __name__ == "__main__":
    generatePyFromUi()
