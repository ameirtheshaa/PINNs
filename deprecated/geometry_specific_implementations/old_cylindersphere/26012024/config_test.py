from main import *
from config import *

config["chosen_machine"] = "CREATE"
# config["chosen_machine"] = "google"

config["base_folder_names"] = ["16012024_adam_datalossonly_infinite_continued", "16012024_adam_datalosscontloss_paraview_infinite", "19012024_adam_cont_boundary_RANS_loss_infinite"]
# config["base_folder_names"] = ["16012024_adam_datalossonly_infinite_continued"]
config["training"]["output_params"] = ['Pressure', 'Velocity:0', 'Velocity:1', 'Velocity:2', 'TurbVisc']
config["training"]["output_params_modf"] = ['Pressure', 'Velocity_X', 'Velocity_Y', 'Velocity_Z', 'TurbVisc']
config["train_test"]["test"] = True
config["train_test"]["evaluate"] = True
config["train_test"]["evaluate_new_angles"] = True
config["plotting"]["make_logging_plots"] = True
config["plotting"]["make_div_plots"] = True
config["plotting"]["make_RANS_plots"] = True
# config["plotting"]["make_data_plots"] = True

# config["base_folder_names"] = ["06012024_adam_datalossonly_infinite"]
# config["training"]["output_params"] = ['Pressure', 'Velocity:0', 'Velocity:1', 'Velocity:2', 'TurbVisc']
# config["training"]["output_params_modf"] = ['Pressure', 'Velocity_X', 'Velocity_Y', 'Velocity_Z', 'TurbVisc']
# config["train_test"]["test"] = True
# config["train_test"]["evaluate"] = True
# config["train_test"]["evaluate_new_angles"] = True
# config["plotting"]["make_logging_plots"] = True
# config["plotting"]["make_div_plots"] = True
# # config["plotting"]["make_data_plots"] = True

# config["base_folder_names"] = ["04012024_adam_datalosscontloss_infinite_adaptiveweighting", "04012024_adam_datalossinletloss_infinite_adaptiveweighting","04012024_adam_datalossnosliploss_infinite_adaptiveweighting"]
# config["train_test"]["test"] = True
# config["train_test"]["evaluate"] = True
# # config["train_test"]["evaluate_new_angles"] = True
# config["plotting"]["make_logging_plots"] = True
# config["plotting"]["make_div_plots"] = True
# config["plotting"]["make_data_plots"] = True

# config["base_folder_names"] = ["01012024_adam_datalossonly_infinite_tpu"]
# config["train_test"]["test"] = True
# config["train_test"]["evaluate"] = True
# config["train_test"]["evaluate_new_angles"] = True
# config["plotting"]["make_logging_plots"] = True
# config["plotting"]["make_div_plots"] = True
# config["plotting"]["make_data_plots"] = True

# config["base_folder_names"] = ["18122023_adam_datalosscontloss_infinite_adaptiveweighting", "18122023_adam_datalossinletloss_infinite_adaptiveweighting","18122023_adam_datalossnosliploss_infinite_adaptiveweighting"]
# config["train_test"]["test"] = True
# config["train_test"]["evaluate"] = True
# config["train_test"]["evaluate_new_angles"] = True
# config["plotting"]["make_logging_plots"] = True
# config["plotting"]["make_div_plots"] = True
# config["plotting"]["make_data_plots"] = True

# config["base_folder_names"] = ["test_4"]
# config["train_test"]["train"] = True
# config["training"]["use_epochs"] = True
# config["training"]["num_epochs"] = 100
# config["training"]["print_epochs"] = 1
# config["loss_components"]["data_loss"] = True
# config["loss_components"]["cont_loss"] = True
# config["loss_components"]["boundary_loss"] = True
# config["loss_components"]["no_slip_loss"] = True
# config["loss_components"]["inlet_loss"] = True
# config["loss_components"]["use_weighting"] = True
# config["train_test"]["test_size"] = 0.99999
# config["training"]["use_batches"] = True
# config["training"]["batch_size"] = 1
# config["train_test"]["data_loss_test_size"] = 0.9

# config["base_folder_names"] = ["26112023_adam_datalossonly_infinite"]
# config["training"]["activation_function"] = nn.ReLU
# config["train_test"]["make_div_plots"] = True
# # config["train_test"]["new_angles"] = [0]
# # config["train_test"]["new_angles"] = config["training"]["all_angles"]
# # config["train_test"]["new_angles_test_size"] = 0.1

if __name__ == "__main__":
    chosen_machine = config["chosen_machine"]
    for base_directory_ in config["base_folder_names"]:
        base_directory = os.path.join(config["machine"][chosen_machine], base_directory_)
        print (f"doing {base_directory_} now!")
        main(base_directory, config)