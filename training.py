"""Project-root shim for `src/training/training.py` (legacy flat imports)."""
from _pinns_src_import import export_all, load_src_module

_mod = load_src_module("pinns_flat_training", "training", "training.py")
export_all(globals(), _mod)
