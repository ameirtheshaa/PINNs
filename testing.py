"""Project-root shim for `src/training/testing.py`."""
from _pinns_src_import import export_all, load_src_module

_mod = load_src_module("pinns_flat_testing", "training", "testing.py")
export_all(globals(), _mod)
