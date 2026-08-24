from src.utils.config import load_config


def test_default_config_loads_and_has_expected_top_level_keys():
    config = load_config("configs/default.yaml")
    for key in ("data", "model", "calibration", "conformal", "paths"):
        assert key in config
