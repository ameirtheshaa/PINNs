from main import *

config = {
    "lbfgs_optimizer": {
        "type": "LBFGS",
        "learning_rate": 0.00001,
        "max_iter": 200000,
        "max_eval": 50000,
        "history_size": 50,
        "tolerance_grad": 1e-05,
        "tolerance_change": 0.5 * np.finfo(float).eps,
        "line_search_fn": "strong_wolfe"
    },
    "adam_optimizer": {
        "type": "Adam",
        "learning_rate": 0.001,
    },
    "training": {
        "number_of_hidden_layers": 4,
        "neuron_number": 128,
        "input_params": ['Points:0', 'Points:1', 'Points:2', 'cos(WindAngle)', 'sin(WindAngle)'], 
        "input_params_modf": ["X", "Y", "Z", "cos(WindAngle)", "sin(WindAngle)"], 
        "output_params": ['Velocity:0', 'Velocity:1', 'Velocity:2'], #['Pressure', 'Velocity:0', 'Velocity:1', 'Velocity:2', 'TurbVisc']
        "output_params_modf": ['Velocity_X', 'Velocity_Y', 'Velocity_Z'], #['Pressure', 'Velocity_X', 'Velocity_Y', 'Velocity_Z', 'TurbVisc']
        "activation_function": nn.ELU, #nn.ReLU, nn.Tanh,
        "batch_normalization": False,
        "dropout_rate": None,
        "use_epochs": False,
        "num_epochs": 1,
        "use_batches": True,
        "batch_size": 2**15,
        "angles_to_leave_out": [135],
        "angles_to_train": [0,15,30,45,60,75,90,105,120,150,165,180],
        "all_angles": [0,15,30,45,60,75,90,105,120,135,150,165,180],
        "loss_diff_threshold": 1e-5,
        "consecutive_count_threshold": 10,
        "feature_scaler": sklearn.preprocessing.StandardScaler(), #sklearn.preprocessing.MinMaxScaler(feature_range=(-1,1))
        "target_scaler": sklearn.preprocessing.StandardScaler(), #sklearn.preprocessing.MinMaxScaler(feature_range=(-1,1))
    },
    "machine": {
        "mac": os.path.join(os.path.expanduser("~"), "Dropbox", "School", "Graduate", "CERN", "Temp_Files", "nonlineardynamics", "ameir_PINNs", "cylinder_cell"),
        "workstation_CREATE": os.path.join('E:\\','ameir', "Dropbox", "School", "Graduate", "CERN", "Temp_Files", "nonlineardynamics", "ameir_PINNs", "cylinder_cell"),
        "laptop_CREATE": os.path.join('E:\\', "Dropbox", "School", "Graduate", "CERN", "Temp_Files", "nonlineardynamics", "ameir_PINNs", "cylinder_cell"),
        "laptop_CREATE_NTU": os.path.join('D:\\', "Dropbox", "School", "Graduate", "CERN", "Temp_Files", "nonlineardynamics", "ameir_PINNs", "cylinder_cell"),
        "laptop_CREATE_NTU_test": os.path.join('D:\\', "Dropbox"),
        "google": "/content/drive/Othercomputers/My Mac mini/cylinder_cell"
    },
    "data": {
        "density": 1,
        "kinematic_viscosity": 1e-5,
        "data_folder_name": 'data',
        "extension": '.csv',
        "startname_data": 'CFD',
        "startname_meteo": 'meteo_',
        "output_zip_file": 'output.zip',
        "geometry": 'scaled_cylinder_sphere.stl'
    },
    "loss_components": {
        "data_loss": True,
        "cont_loss": True,
        "momentum_loss": False,
        "boundary_loss": True,
        "no_slip_loss":True, 
        "inlet_loss": True,
        "use_weighting": True,
        "weighting_scheme": 'adaptive_weighting', #'gradient_magnitude'
        "adaptive_weighting_initial_weight": 0.9,
        "adaptive_weighting_final_weight": 0.1,
    },
    "train_test": {
        "train": True,
        "test": False,
        "evaluate": False,
        "evaluate_new_angles": False,
        "make_logging_plots": False,
        "make_data_plots": False,
        "make_div_plots": False,
        "test_size": 0.1,
        "data_loss_test_size": None,
        "new_angles_test_size": 0.99999,
        "random_state": 42,
        "new_angles": [0,15,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,75,90,105,120,135,150,165,180]
    },
    "chosen_machine": "mac",
    "chosen_optimizer": "adam_optimizer",
    "base_folder_name": '18122023_adam_datalossinletloss_infinite_adaptiveweighting'
}

config["chosen_machine"] = "laptop_CREATE"
config["train_test"]["test_size"] = 0.1
config["training"]["use_batches"] = True
config["training"]["batch_size"] = 2**15
config["training"]["use_epochs"] = False
config["training"]["num_epochs"] = 10
config["loss_components"]["cont_loss"] = False
config["loss_components"]["no_slip_loss"] = False
config["loss_components"]["inlet_loss"] = True
config["loss_components"]["use_weighting"] = True
config["loss_components"]["weighting_scheme"] = "adaptive_weighting"
config["loss_components"]["adaptive_weighting_initial_weight"] = 0.9
config["loss_components"]["adaptive_weighting_final_weight"] = 0.1
config["base_folder_name"] = "18122023_adam_datalossinletloss_infinite_adaptiveweighting"

if __name__ == "__main__":
    chosen_machine = config["chosen_machine"]
    base_directory = os.path.join(config["machine"][chosen_machine], config["base_folder_name"])
    main(base_directory, config)