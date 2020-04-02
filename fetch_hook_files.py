import os
import shutil

def fetch_hook_files():
    """
    Fetches entire PyInstaller 'hooks' directory and packages them as hooks.zip

    Unpack this file into 'site-packages/PyInstaller/hooks' to make sure that
    fbs/PyInstaller can find all imports and compile properly
    """
    cwd = os.getcwd()

    hooks_dir = os.path.join(cwd, "venv/lib/python3.7/site-packages/PyInstaller/hooks")

    shutil.make_archive(os.path.join(cwd, "hooks"), 'zip', hooks_dir)

if __name__ == "__main__":
    fetch_hook_files()