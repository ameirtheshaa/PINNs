"""Project-root shim for `from PINN import *` inside `src`."""
from _pinns_src_import import export_all, load_src_module

_mod = load_src_module("pinns_flat_PINN", "models", "PINN.py")
export_all(globals(), _mod)
