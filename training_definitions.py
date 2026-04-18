"""Project-root shim for `src/training/training_definitions.py`."""
from _pinns_src_import import export_all, load_src_module

_mod = load_src_module("pinns_flat_training_definitions", "training", "training_definitions.py")
export_all(globals(), _mod)
