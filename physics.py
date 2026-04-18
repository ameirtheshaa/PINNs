"""Project-root shim for physics package."""
from _pinns_src_import import export_all, load_src_module

_mod = load_src_module("pinns_flat_physics", "physics", "physics.py")
export_all(globals(), _mod)
