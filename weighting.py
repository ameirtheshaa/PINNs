"""Project-root shim for `utils.weighting`."""
from _pinns_src_import import export_all, load_src_module

_mod = load_src_module("pinns_flat_weighting", "utils", "weighting.py")
export_all(globals(), _mod)
