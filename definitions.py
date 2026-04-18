"""Project-root shim: expose `utils.definitions` as legacy top-level `definitions`."""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_ROOT, "src")
_CFG = os.path.join(_ROOT, "configs")
for _p in (_CFG, _SRC, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.definitions import *  # noqa: E402,F403
