from main import *
from config import *

config["chosen_machine"] = "laptop_CREATE_NTU"

config["base_folder_name"] = "01012024_adam_datalossonly_infinite_tpu"
config["train_test"]["test"] = True
# config["train_test"]["evaluate"] = True
# config["train_test"]["evaluate_new_angles"] = True
# config["train_test"]["make_logging_plots"] = True
# config["train_test"]["make_div_plots"] = True
# config["train_test"]["make_data_plots"] = True

if __name__ == "__main__":
    main(config)