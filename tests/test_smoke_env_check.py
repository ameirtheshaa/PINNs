"""Smoke: venv tooling + full import graph via main()."""

from config import config
from main import main


def test_smoke_env_check_cli_local_safe_config():
    cfg = dict(config)
    cfg["chosen_machine"] = "mac"
    cfg["machine"] = dict(cfg["machine"])
    cfg["machine"]["mac"] = "/tmp/pinns_smoke_data"
    cfg["base_folder_names"] = ["smoke_tmp"]
    cfg["train_test"] = dict(cfg["train_test"])
    cfg["train_test"].update(
        {
            "train": False,
            "test": False,
            "evaluate": False,
            "boundary_test": False,
            "evaluate_new_angles": False,
        }
    )
    cfg["plotting"] = dict(cfg["plotting"])
    cfg["plotting"].update(
        {
            "make_logging_plots": False,
            "make_data_plots": False,
            "make_div_plots": False,
            "make_RANS_plots": False,
        }
    )
    # No return value; must not raise
    main("smoke_env_check_cli", cfg)
