# -*- coding: utf-8

import yaml

def load_cfg(filepath="./config.yaml"):
    """Load YAML configuration file

    Params
    ======
    filepath (str): The path of the YAML configuration file
    """
    with open(filepath, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)
