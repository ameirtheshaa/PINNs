"""Verify root shims and shared config load."""

def test_definitions_shim_imports():
    import definitions as d  # noqa: F401 — side effect: registers utils

    assert hasattr(d, "np")
    assert hasattr(d, "torch")


def test_config_object():
    from config import config

    assert isinstance(config, dict)
    assert "training" in config
    assert "machine" in config


def test_pinns_src_import_helper():
    from _pinns_src_import import load_src_module

    m = load_src_module("pytest_pinns_models", "models", "PINN.py")
    assert hasattr(m, "PINN")
