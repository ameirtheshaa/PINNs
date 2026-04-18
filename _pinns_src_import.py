"""Load modules from `src/` by file path without shadowing package names."""
import importlib.util
import os


def load_src_module(unique_name: str, *relative_parts: str):
    root = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(root, "src", *relative_parts)
    spec = importlib.util.spec_from_file_location(unique_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def export_all(globs: dict, mod) -> None:
    names = getattr(mod, "__all__", None)
    if names:
        for k in names:
            globs[k] = getattr(mod, k)
        return
    for k, v in vars(mod).items():
        if k.startswith("_"):
            continue
        globs[k] = v
