import os
import json
from src.main.python.lib.utils import git_app_version


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
    write_to_base_json()
    compile()
