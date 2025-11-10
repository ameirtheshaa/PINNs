from main import *
from config import *

config["machine"] = {
        "mac": os.path.join(os.path.expanduser("~"), "Dropbox", "School", "Graduate", "CERN", "Temp_Files", "nonlineardynamics", "ameir_PINNs", "ladefense"),
        "CREATE": os.path.  join('Z:\\', "ladefense"),
        "google": f"/content/drive/Othercomputers/MacMini/ladefense",
    }

# config["chosen_machine"] = "CREATE"
config["data"]["geometry"] = "ladefense.stl"
config["train_test"]["test_size"] = 0.1
config["train_test"]["train"] = True
config["training"]["PCA"] = True
config["training"]["Incremental_PCA"] = True
config["training"]["output_params"] = ['Pressure', 'Velocity:0', 'Velocity:1', 'Velocity:2', 'TurbVisc']
config["training"]["output_params_modf"] = ['Pressure', 'Velocity_X', 'Velocity_Y', 'Velocity_Z', 'TurbVisc']
config["training"]["boundary"] = [[-2520,2520,-2520,2520,0,1000],100]
config["training"]["neuron_number"] = 128
config["training"]["print_epochs"] = 1
config["training"]["save_metrics"] = 1
config["loss_components"]["data_loss"] = True
config["base_folder_name"] = "pca_test_2"

if __name__ == "__main__":
    device, data_dict, input_params, output_params, model = initialize_data(config)
    chosen_machine = config["chosen_machine"]
    base_directory = os.path.join(config["machine"][chosen_machine], config["base_folder_name"])
    main(base_directory, config, device, data_dict, input_params, output_params, model)