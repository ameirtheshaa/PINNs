from main import *
from config import *

config["chosen_machine"] = "workstation_CREATE"
config["train_test"]["test_size"] = 0.1
config["train_test"]["train"] = True
config["training"]["print_epochs"] = 1
config["training"]["output_params"] = ['Pressure', 'Velocity:0', 'Velocity:1', 'Velocity:2', 'TurbVisc']
config["training"]["output_params_modf"] = ['Pressure', 'Velocity_X', 'Velocity_Y', 'Velocity_Z', 'TurbVisc']
config["training"]["use_batches"] = True
config["training"]["batch_size"] = 2**15
config["training"]["use_data_for_cont_loss"] = True
config["loss_components"]["cont_loss"] = True
config["base_folder_name"] = "14012024_adam_datalosscontloss_paraview_infinite"

if __name__ == "__main__":
    chosen_machine = config["chosen_machine"]
    base_directory = os.path.join(config["machine"][chosen_machine], config["base_folder_name"])
    main(base_directory, config)