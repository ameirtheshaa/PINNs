"""Project-root shim for `src/visualization/plotting_definitions.py`."""
from _pinns_src_import import export_all, load_src_module

_mod = load_src_module("pinns_flat_plotting_definitions", "visualization", "plotting_definitions.py")
export_all(globals(), _mod)
