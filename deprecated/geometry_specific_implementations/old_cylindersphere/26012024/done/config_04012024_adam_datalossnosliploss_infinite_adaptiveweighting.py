from main import *
from config import *

config["chosen_machine"] = "workstation_CREATE"
config["train_test"]["test_size"] = 0.1
config["train_test"]["train"] = True
config["training"]["use_batches"] = True
config["training"]["batch_size"] = 2**15
config["training"]["print_epochs"] = 1
config["loss_components"]["data_loss"] = True
config["loss_components"]["no_slip_loss"] = True
config["loss_components"]["use_weighting"] = True
config["base_folder_name"] = "04012024_adam_datalossnosliploss_infinite_adaptiveweighting"

if __name__ == "__main__":
    chosen_machine = config["chosen_machine"]
    base_directory = os.path.join(config["machine"][chosen_machine], config["base_folder_name"])
    main(base_directory, config)