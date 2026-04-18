"""Project-root shim for `boundary_conditions.boundary`."""
from _pinns_src_import import export_all, load_src_module

_mod = load_src_module("pinns_flat_boundary", "boundary_conditions", "boundary.py")
export_all(globals(), _mod)
