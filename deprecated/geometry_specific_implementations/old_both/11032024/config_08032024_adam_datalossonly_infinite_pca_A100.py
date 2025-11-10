from main_pca import *
from config import *

config["chosen_machine"] = "google"
config["train_test"]["test_size"] = 0.1
config["train_test"]["train"] = True
config["training"]["output_params"] = ['Pressure', 'Velocity:0', 'Velocity:1', 'Velocity:2', 'TurbVisc']
config["training"]["output_params_modf"] = ['Pressure', 'Velocity_X', 'Velocity_Y', 'Velocity_Z', 'TurbVisc']
config["training"]["neuron_number"] = 2**11
config["training"]["print_epochs"] = 10
config["training"]["use_epochs"] = True
config["training"]["num_epochs"] = 4000
config["loss_components"]["data_loss"] = True
config["base_folder_name"] = f"08032024_adam_datalossonly_infinite_PCA_A100"

if __name__ == "__main__":
    chosen_machine = config["chosen_machine"]
    base_directory = os.path.join(config["machine"][chosen_machine], config["base_folder_name"])
    main(base_directory, config)