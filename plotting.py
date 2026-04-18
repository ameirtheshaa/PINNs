"""Project-root shim for `src/visualization/plotting.py`."""
from _pinns_src_import import export_all, load_src_module

_mod = load_src_module("pinns_flat_plotting", "visualization", "plotting.py")
export_all(globals(), _mod)
