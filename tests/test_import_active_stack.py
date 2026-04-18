"""
Import checks for the active (non-deprecated) code path.

Heavy imports mirror what `main.py` pulls in via its top-level imports.
"""

import pytest


def test_root_shim_modules_importable():
    """Each shim must load without error (same graph as main)."""
    import boundary  # noqa: F401
    import boundary_testing  # noqa: F401
    import physics  # noqa: F401
    import plotting  # noqa: F401
    import plotting_definitions  # noqa: F401
    import testing as testing_mod  # noqa: F401
    import training  # noqa: F401
    import training_definitions  # noqa: F401
    import weighting  # noqa: F401


@pytest.mark.parametrize(
    "name",
    [
        "src.utils.definitions",
        "src.models.PINN",
        "src.physics.physics",
        "src.training.training",
        "src.training.testing",
        "src.visualization.plotting",
        "src.boundary_conditions.boundary",
    ],
)
def test_src_submodule_import_by_package_name(name):
    """Import canonical src packages (requires repo root + src layout)."""
    __import__(name)
