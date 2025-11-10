from main import *
from config import *

# config["chosen_machine"] = "CREATE"

config["base_folder_names"] = ["16012024_adam_datalossonly_infinite"]
# config["training"]["output_params"] = ['Pressure', 'Velocity:0', 'Velocity:1', 'Velocity:2', 'TurbVisc']
# config["training"]["output_params_modf"] = ['Pressure', 'Velocity_X', 'Velocity_Y', 'Velocity_Z', 'TurbVisc']
# config["train_test"]["train"] = True
# config["training"]["print_epochs"] = 1
config["train_test"]["test"] = True
config["train_test"]["evaluate"] = True
# config["train_test"]["evaluate_new_angles"] = True
config["train_test"]["make_logging_plots"] = True
# config["train_test"]["make_div_plots"] = True
# config["train_test"]["make_data_plots"] = True

if __name__ == "__main__":
    chosen_machine = config["chosen_machine"]
    for base_directory_ in config["base_folder_names"]:
        base_directory = os.path.join(config["machine"][chosen_machine], base_directory_)
        print (f"doing {base_directory_} now!")
        main(base_directory, config)