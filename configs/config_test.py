from main import *
from config import *

config["chosen_machine"] = "CREATE"

# config["chosen_machine"] = "google"
# config["training"]["force_device"] = "cpu"

################################################################################################################################################################################################################################################################################################################################

config["plotting"]["make_logging_plots"] = True
config["train_test"]["test"] = True
config["train_test"]["evaluate"] = True
# # config["plotting"]["make_div_plots"] = True
# # config["plotting"]["make_RANS_plots"] = True
# config["plotting"]["save_csv_predictions"] = True
# config["plotting"]["save_vtk"] = True
config["plotting"]["make_plots"] = True

################################################################################################################################################################################################################################################################################################################################

# config["machine"] = {
#         "mac": os.path.join(os.path.expanduser("~"), "Dropbox", "School", "Graduate", "CERN", "Temp_Files", "nonlineardynamics", "ameir_PINNs", "ladefense"),
#         "CREATE": os.path.join('Z:\\', "ladefense"),
#         "google": f"/content/drive/Othercomputers/MacMini/ladefense",
#     }
# config["data"]["geometry"] = "ladefense.stl"

# config["training"]["boundary"] = [[-2520,2520,-2520,2520,0,1000],100]
# config["plotting"]["lim_min_max"] = [(-0.3, 0.3),(-0.3, 0.3),(0, 1)] 
# # config["plotting"]["plotting_params"] = [['X-Y',5,5], ['Y-Z',0,5], ['X-Z',0,5]]
# config["plotting"]["plotting_params"] = [['X-Y',5,5]]

# # variable_configs = [
# #     {"base_folder_name": "16012024_adam_datalossonly_infinite"},
# #     {"base_folder_name": "19022024_adam_datalossonly_sampled_70_70_10_infinite_ladefense"}
# # ]

variable_configs = [
    # {"base_folder_name": "20032024_adam_datalossonly_infinite_ladefense_PCA", "neuron_number": 128, "use_PCA": True, "epoch_number": None},
    # {"base_folder_name": "29032024_adam_datalossonly_infinite_ladefense_PCA", "neuron_number": 128, "use_PCA": True, "epoch_number": None},
    # {"base_folder_name": "31032024_adam_datalossonly_infinite_ladefense_PCA", "neuron_number": 128, "use_PCA": True, "epoch_number": None},
    # {"base_folder_name": "03042024_adam_datalossonly_infinite_ladefense_FFT_PCA", "neuron_number": 128, "use_fft": True, "features_factor": 2, "targets_factor": 2, "epoch_number": None},
    # {"base_folder_name": "04042024_adam_datalossonly_infinite_ladefense_FFT_PCA", "neuron_number": 128, "use_fft": True, "features_factor": 2, "targets_factor": 2,"epoch_number": None},
    ]




################################################################################################################################################################################################################################################################################################################################


# variable_configs = [
#     {"base_folder_name": "16012024_adam_datalossonly_infinite_continued_test", "output_params": ['Pressure', 'Velocity:0', 'Velocity:1', 'Velocity:2', 'TurbVisc'], "output_params_modf": ['Pressure', 'Velocity_X', 'Velocity_Y', 'Velocity_Z', 'TurbVisc']},
#     {"base_folder_name": "16012024_adam_datalossonly_infinite_continued_test", "boundary_test": True, "lim_min_max": [(-0.05, 0.05),(-0.1, 0.3),(0, 0.3)], "output_params": ['Pressure', 'Velocity:0', 'Velocity:1', 'Velocity:2', 'TurbVisc'], "output_params_modf": ['Pressure', 'Velocity_X', 'Velocity_Y', 'Velocity_Z', 'TurbVisc']},
#     {"base_folder_name": "11032024_adam_datalossonly_infinite_pca_test", "neuron_number": 128, "PCA": True, "output_params": ['Pressure', 'Velocity:0', 'Velocity:1', 'Velocity:2', 'TurbVisc'], "output_params_modf": ['Pressure', 'Velocity_X', 'Velocity_Y', 'Velocity_Z', 'TurbVisc']},
# ]

# variable_configs = [
#      ]

variable_configs = [
#     {"base_folder_name": "pca_test", "neuron_number": 128, "PCA": True, "Kernel_PCA": True, "output_params": ['Pressure', 'Velocity:0', 'Velocity:1', 'Velocity:2', 'TurbVisc'], "output_params_modf": ['Pressure', 'Velocity_X', 'Velocity_Y', 'Velocity_Z', 'TurbVisc'], "epoch_number": None}
#     # {"base_folder_name": "pca_test_81", "angles_to_train": [0,150], "neuron_number": 128, "use_PCA": True, "epoch_number": None},
#     {"base_folder_name": "test_128_0.001__1712478080.987767", "angles_to_train": [0,150], "neuron_number": 128, "use_fft": True, "features_factor": 2, "targets_factor": 2,  "epoch_number": None},
    {"base_folder_name": "fft_test_128_0.001_1712571992.4813795", "neuron_number": 128, "use_fft": True, "features_factor": 2, "targets_factor": 2,  "epoch_number": None},
     ]

################################################################################################################################################################################################################################################################################################################################

if __name__ == "__main__":
    previous_params, current_params = None, None
    for var_config in variable_configs:
        backup = copy.deepcopy(config)
        config["training"].update({k: var_config[k] for k in ["angles_to_train", 'use_PCA', 'use_fft', 'neuron_number', 'output_params', 'output_params_modf', 'features_factor', 'targets_factor'] if k in var_config})
        config["train_test"].update({k: var_config[k] for k in ['boundary_test'] if k in var_config})
        config["plotting"].update({k: var_config[k] for k in ['lim_min_max'] if k in var_config})
        config["testing"].update({k: var_config[k] for k in ['epoch_number'] if k in var_config})
        config["base_folder_names"] = [var_config["base_folder_name"]] 
        chosen_machine = config["chosen_machine"]
        if previous_params is None:
            device, data_dict, input_params, output_params, model = initialize_data(config)
        return_data = reinitialize_data(config, input_params, output_params, previous_params)
        if len(return_data)==1:
            current_params = return_data
        elif len(return_data)==2:
            model, current_params = return_data
        elif len(return_data)==5:
            data_dict, input_params, output_params, model, current_params = return_data
        for base_directory_ in config["base_folder_names"]:
            base_directory = os.path.join(config["machine"][chosen_machine], base_directory_)
            print (f"doing {base_directory_} now!")
            main(base_directory, config, device, data_dict, input_params, output_params, model)
        previous_params = current_params
        config = copy.deepcopy(backup)