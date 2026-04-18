"""Expose `configs/config.py` as top-level `config` for legacy `from config import config`."""
import importlib.util
import os

_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "config.py")
_spec = importlib.util.spec_from_file_location("pinns_configs_config", _cfg_path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
config = _module.config
