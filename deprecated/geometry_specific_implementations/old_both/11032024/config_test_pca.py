from main_pca import *
from config import *

# config["chosen_machine"] = "CREATE"

################################################################################################################################################################################################################################################################################################################################

# config["machine"] = {
#         "mac": os.path.join(os.path.expanduser("~"), "Dropbox", "School", "Graduate", "CERN", "Temp_Files", "nonlineardynamics", "ameir_PINNs", "ladefense"),
#         "CREATE": os.path.join('Z:\\', "ladefense"),
#         "google": f"/content/drive/Othercomputers/MacMini/ladefense",
#     }
# config["data"]["geometry"] = "ladefense.stl"

# config["base_folder_names"] = ["07032024_adam_datalossonly_infinite_ladefense_PCA"]
# config["training"]["boundary"] = [[-2520,2520,-2520,2520,0,1000],100]
# config["plotting"]["lim_min_max"] = [(-0.3, 0.3),(-0.3, 0.3),(0, 1)] 
# config["plotting"]["plotting_params"] = [['X-Y',5,5], ['Y-Z',0,5], ['X-Z',0,5]]
# # config["train_test"]["test"] = True
# # config["train_test"]["evaluate"] = True
# # # config["train_test"]["evaluate_new_angles"] = True
# # config["plotting"]["make_plots"] = True
# config["plotting"]["make_logging_plots"] = True
# # config["plotting"]["make_div_plots"] = True
# # config["plotting"]["make_RANS_plots"] = True
# # # config["plotting"]["make_data_plots"] = True

################################################################################################################################################################################################################################################################################################################################

config["base_folder_names"] = ["10032024_adam_datalossonly_infinite_pca_test"]
config["train_test"]["test"] = True
config["training"]["neuron_number"] = 128
config["training"]["output_params"] = ['Pressure', 'Velocity:0', 'Velocity:1', 'Velocity:2', 'TurbVisc']
config["training"]["output_params_modf"] = ['Pressure', 'Velocity_X', 'Velocity_Y', 'Velocity_Z', 'TurbVisc']
config["plotting"]["save_csv_predictions"] = True
config["plotting"]["make_plots"] = True
config["train_test"]["evaluate"] = True
# config["train_test"]["evaluate_new_angles"] = True
config["plotting"]["make_logging_plots"] = True
# config["plotting"]["make_plots"] = True
# config["plotting"]["make_div_plots"] = True
# config["plotting"]["make_RANS_plots"] = True

################################################################################################################################################################################################################################################################################################################################

if __name__ == "__main__":
    chosen_machine = config["chosen_machine"]
    for base_directory_ in config["base_folder_names"]:
        base_directory = os.path.join(config["machine"][chosen_machine], base_directory_)
        print (f"doing {base_directory_} now!")
        main(base_directory, config)