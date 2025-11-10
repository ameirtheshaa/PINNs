from main import *
from config import *

config["machine"] = {
        "mac": os.path.join(os.path.expanduser("~"), "Dropbox", "School", "Graduate", "CERN", "Temp_Files", "nonlineardynamics", "ameir_PINNs", "ladefense"),
        "CREATE": os.path.join('Z:\\', "ladefense"),
        "google": f"/content/drive/Othercomputers/MacMini/ladefense",
    }

config["chosen_machine"] = "CREATE"
config["data"]["geometry"] = "ladefense.stl"
config["train_test"]["test_size"] = 0.1
config["train_test"]["train"] = True
config["training"]["use_batches"] = True
config["training"]["batch_size"] = 2**15
config["training"]["print_epochs"] = 1
config["training"]["boundary"] = [[-2520,2520,-2520,2520,0,1000],100]
config["loss_components"]["data_loss"] = True
config["base_folder_name"] = "16012024_adam_datalossonly_infinite"

if __name__ == "__main__":
    chosen_machine = config["chosen_machine"]
    base_directory = os.path.join(config["machine"][chosen_machine], config["base_folder_name"])
    main(base_directory, config)