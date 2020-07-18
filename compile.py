import os
import json
from configobj import ConfigObj
from src.main.python.lib.utils import git_app_version

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
    write_to_config_ini()
    write_to_base_json()
    compile()
