"""Project-root shim for `boundary_conditions.boundary_testing`."""
from _pinns_src_import import export_all, load_src_module

_mod = load_src_module("pinns_flat_boundary_testing", "boundary_conditions", "boundary_testing.py")
export_all(globals(), _mod)
