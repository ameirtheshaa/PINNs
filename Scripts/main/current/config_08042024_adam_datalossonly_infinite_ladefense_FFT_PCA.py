from main import *
from config import *

config["machine"] = {
        "mac": os.path.join(os.path.expanduser("~"), "Dropbox", "School", "Graduate", "CERN", "Temp_Files", "nonlineardynamics", "ameir_PINNs", "ladefense"),
        "CREATE": os.path.join('Z:\\', "ladefense"),
        "google": f"/content/drive/Othercomputers/MacMini/ladefense",
    }

config["chosen_machine"] = "CREATE"
config["data"]["geometry"] = "ladefense.stl"
config["train_test"]["train"] = True
config["train_test"]["force_device"] = 'cpu'

config["training"]["use_fft"] = True
config["training"]["features_factor"] = 2
config["training"]["targets_factor"] = 2
config["training"]["boundary"] = [[-2520,2520,-2520,2520,0,1000],100]
config["training"]["print_epochs"] = 1
config["training"]["save_metrics"] = 1
config["training"]["total_model_eval"] = 100

config["plotting"]["make_plots"] = True

config["loss_components"]["data_loss"] = True

config["base_folder_name"] = "08042024_adam_datalossonly_infinite_ladefense_FFT_PCA"

if __name__ == "__main__":
    device, data_dict, input_params, output_params, model = initialize_data(config)
    chosen_machine = config["chosen_machine"]
    base_directory = os.path.join(config["machine"][chosen_machine], config["base_folder_name"])
    main(base_directory, config, device, data_dict, input_params, output_params, model)