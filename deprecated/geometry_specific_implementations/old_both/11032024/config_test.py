from main import *
from config import *

config["chosen_machine"] = "CREATE"

################################################################################################################################################################################################################################################################################################################################

# config["machine"] = {
#         "mac": os.path.join(os.path.expanduser("~"), "Dropbox", "School", "Graduate", "CERN", "Temp_Files", "nonlineardynamics", "ameir_PINNs", "ladefense"),
#         "CREATE": os.path.join('Z:\\', "ladefense"),
#         "google": f"/content/drive/Othercomputers/MacMini/ladefense",
#     }
# config["data"]["geometry"] = "ladefense.stl"

# # # config["base_folder_names"] = ["19022024_adam_datalossonly_sampled_70_70_10_infinite_ladefense"]
# # config["base_folder_names"] = ["16012024_adam_datalossonly_infinite"]
# config["base_folder_names"] = ["19022024_adam_datalossonly_sampled_70_70_10_infinite_ladefense", "16012024_adam_datalossonly_infinite"]
# config["training"]["boundary"] = [[-2520,2520,-2520,2520,0,1000],100]
# config["plotting"]["lim_min_max"] = [(-0.3, 0.3),(-0.3, 0.3),(0, 1)] 
# config["plotting"]["plotting_params"] = [['X-Y',5,5], ['Y-Z',0,5], ['X-Z',0,5]]
# config["train_test"]["test"] = True
# config["train_test"]["evaluate"] = True
# # config["train_test"]["evaluate_new_angles"] = True
# config["plotting"]["make_plots"] = True
# config["plotting"]["make_logging_plots"] = True
# # config["plotting"]["make_div_plots"] = True
# # config["plotting"]["make_RANS_plots"] = True
# # # config["plotting"]["make_data_plots"] = True

# config["base_folder_names"] = ["16012024_adam_datalossonly_infinite", "07022024_adam_datalossonly_infinite_paraview_reduced_ladefense", "06022024_adam_datalossonly_infinite_A100"]
# config["training"]["boundary"] = [[-2520,2520,-2520,2520,0,1000],100]
# config["plotting"]["lim_min_max"] = [(-0.3, 0.3),(-0.3, 0.3),(0, 1)] 
# config["plotting"]["plotting_params"] = [['X-Y',5,5]]
# config["train_test"]["test"] = True
# config["train_test"]["evaluate"] = True
# # config["train_test"]["evaluate_new_angles"] = True
# config["plotting"]["make_logging_plots"] = True
# # config["plotting"]["make_div_plots"] = True
# # config["plotting"]["make_RANS_plots"] = True
# # # config["plotting"]["make_data_plots"] = True

# config["base_folder_names"] = ["07022024_adam_datalossonly_infinite_paraview_reduced_ladefense"]
# config["data"]["startname_data"] = "Paraview_Reduced_CFD"
# config["training"]["boundary"] = [[-2520,2520,-2520,2520,0,1000],100]
# config["plotting"]["lim_min_max"] = [(-0.3, 0.3),(-0.3, 0.3),(0, 1)]
# config["plotting"]["plotting_params"] = [['X-Y',5,5]]
# config["train_test"]["test"] = True
# config["train_test"]["evaluate"] = True
# # config["train_test"]["evaluate_new_angles"] = True
# config["plotting"]["make_logging_plots"] = True
# # config["plotting"]["make_div_plots"] = True
# # config["plotting"]["make_RANS_plots"] = True
# # config["plotting"]["make_data_plots"] = True

# config["base_folder_names"] = ["30012024_adam_datalossonly_infinite_reduced"]
# config["data"]["startname_data"] = "Reduced_CFD"
# config["training"]["boundary"] = [[-504,504,-504,504,0,300],100]
# config["plotting"]["lim_min_max"] = [(-1, 1),(-1, 1),(0, 1)] 
# config["plotting"]["plotting_params"] = [['X-Y',5,5]]
# config["train_test"]["test"] = True
# config["train_test"]["evaluate"] = True
# # config["train_test"]["evaluate_new_angles"] = True
# config["plotting"]["make_logging_plots"] = True
# # config["plotting"]["make_div_plots"] = True
# # config["plotting"]["make_RANS_plots"] = True
# # config["plotting"]["make_data_plots"] = True

################################################################################################################################################################################################################################################################################################################################

# config["base_folder_names"] = ["16012024_adam_datalossonly_infinite_continued_test"]
# config["training"]["output_params"] = ['Pressure', 'Velocity:0', 'Velocity:1', 'Velocity:2', 'TurbVisc']
# config["training"]["output_params_modf"] = ['Pressure', 'Velocity_X', 'Velocity_Y', 'Velocity_Z', 'TurbVisc']

config["base_folder_names"] = ["07032024_adam_datalossonly_infinite_PCA_2048_30000"]

# config["train_test"]["test"] = True
# config["train_test"]["evaluate"] = True
# # config["train_test"]["evaluate_new_angles"] = True
config["plotting"]["make_logging_plots"] = True
# config["plotting"]["make_plots"] = True
# config["plotting"]["make_div_plots"] = True
# config["plotting"]["make_RANS_plots"] = True

# config["train_test"]["boundary_test"] = True
# config["plotting"]["lim_min_max"] = [(-0.05, 0.05),(-0.1, 0.3),(0, 0.3)]
# config["plotting"]["plotting_params"] = [['X-Y',50,5],['Y-Z',500,5]]

# # config["plotting"]["make_data_plots"] = True

# config["base_folder_names"] = ["09022024_adam_paraviewreduceddataloss_allphysicsloss_infinite"]
# config["data"]["startname_data"] = "Paraview_Reduced_CFD"
# config["training"]["output_params"] = ['Pressure', 'Velocity:0', 'Velocity:1', 'Velocity:2', 'TurbVisc']
# config["training"]["output_params_modf"] = ['Pressure', 'Velocity_X', 'Velocity_Y', 'Velocity_Z', 'TurbVisc']
# config["train_test"]["test"] = True
# config["train_test"]["evaluate"] = True
# # config["train_test"]["evaluate_new_angles"] = True
# config["plotting"]["make_logging_plots"] = True
# config["plotting"]["make_div_plots"] = True
# config["plotting"]["make_RANS_plots"] = True
# # config["plotting"]["make_data_plots"] = True

################################################################################################################################################################################################################################################################################################################################

if __name__ == "__main__":
    chosen_machine = config["chosen_machine"]
    for base_directory_ in config["base_folder_names"]:
        base_directory = os.path.join(config["machine"][chosen_machine], base_directory_)
        print (f"doing {base_directory_} now!")
        main(base_directory, config)