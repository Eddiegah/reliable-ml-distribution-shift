from pathlib import Path

import yaml


def load_config(path: str | Path = "configs/default.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)
