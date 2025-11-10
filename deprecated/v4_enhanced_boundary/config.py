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
    "both_optimizers":{
        "type": "both_optimizers",
        "learning_rate_lbfgs": 0.00001,
        "max_iter_lbfgs": 200000,
        "max_eval_lbfgs": 50000,
        "history_size_lbfgs": 50,
        "tolerance_grad_lbfgs": 1e-05,
        "tolerance_change_lbfgs": 0.5 * np.finfo(float).eps,
        "line_search_fn_lbfgs": "strong_wolfe",
        "learning_rate_adam": 0.001,
        "adam_epochs": 2000
    },
    "training": {
        "number_of_hidden_layers": 4,
        "neuron_number": 128,
        "input_params": ['Points:0', 'Points:1', 'Points:2', 'cos(WindAngle)', 'sin(WindAngle)'], 
        "input_params_modf": ["X", "Y", "Z", "cos(WindAngle)", "sin(WindAngle)"], 
        "output_params": ['Pressure', 'Velocity:0', 'Velocity:1', 'Velocity:2', 'TurbVisc'],
        "output_params_modf": ['Pressure', 'Velocity_X', 'Velocity_Y', 'Velocity_Z', 'TurbVisc'],
        # "output_params": ['Velocity:0', 'Velocity:1', 'Velocity:2'],
        # "output_params_modf": ['Velocity_X', 'Velocity_Y', 'Velocity_Z'],
        "activation_function": nn.ReLU,
        # "activation_function": nn.Sigmoid,
        # "activation_function": nn.Tanh,
        "batch_normalization": False,
        "dropout_rate": None,
        "use_epochs": False,
        "num_epochs": 1000,
        "use_batches": True,
        "force": True,
        "batch_size": 2,
        "angle_to_leave_out": [135],
        "angles_to_train": [0,30,60,90,120,150,180],
        "all_angles": [0,30,60,90,120,135,150,180],
        # "angle_to_leave_out": [135],
        # "angles_to_train": [0,15,30,45,60,75,90,105,120,150,165,180],
        # "all_angles": [0,15,30,45,60,75,90,105,120,135,150,165,180],
        "loss_diff_threshold": 1e-5,
        "consecutive_count_threshold": 10,
        "change_scaler": False,
        "scaler": "min_max",
        "min_max_scaler_range": (-1,1),
        "use_custom_points_for_physics_loss": False,
        "number_of_points_per_axis": 3*1E2
    },
    "distributed_training": {
        "use_tcp": False,
        "master_node": True,
        "rank": 0,
        "world_size": 2,
        "ip_address_master": 'localhost',
        "port_number": 29500,
        "backend": 'gloo'
    },
    "machine": {
        "lxplus": '/afs/cern.ch/user/a/abinakbe/PINNs/cylinder_cell',
        "mac": os.path.join(os.path.expanduser("~"), "Dropbox", "School", "Graduate", "CERN", "Temp_Files", "nonlineardynamics", "ameir_PINNs", "cylinder_cell"),
        "mac_test": os.path.join(os.path.expanduser("~"), "Dropbox"),
        "workstation_CREATE": os.path.join('E:\\','ameir', "Dropbox", "School", "Graduate", "CERN", "Temp_Files", "nonlineardynamics", "ameir_PINNs", "cylinder_cell"),
        "laptop_CREATE": os.path.join('E:\\', "Dropbox", "School", "Graduate", "CERN", "Temp_Files", "nonlineardynamics", "ameir_PINNs", "cylinder_cell"),
        "laptop_CREATE_NTU": os.path.join('D:\\', "Dropbox", "School", "Graduate", "CERN", "Temp_Files", "nonlineardynamics", "ameir_PINNs", "cylinder_cell"),
        "laptop_CREATE_NTU_test": os.path.join('D:\\', "Dropbox"),
        "workstation_UofA": os.path.join('C:\\', "Dropbox", "School", "Graduate", "CERN", "Temp_Files", "nonlineardynamics", "ameir_PINNs", "cylinder_cell"),
        "home_PC": os.path.join('C:\\', "Dropbox", "School", "Graduate", "CERN", "Temp_Files", "nonlineardynamics", "ameir_PINNs", "cylinder_cell"),
        "google": "/content/drive/Othercomputers/My Mac mini/cylinder_cell"
    },
    "plotting": {
        "make_pure_data_plots": False,
        "make_pure_data_plots_quiver": True,
        "make_pure_data_plots_total_velocity_arrow": True,
        "make_pure_data_plots_total_velocity": True,
        "make_pure_data_plots_vx": True,
        "make_pure_data_plots_vy": True,
        "make_pure_data_plots_vz": True,
        "make_pure_data_plots_pressure": True,
        "make_pure_data_plots_turbvisc_arrow": True,
        "make_pure_data_plots_turbvisc": True,
        "make_comparison_plots": True,
        "make_comparison_plots_quiver": True,
        "make_comparison_plots_all": False,
        "make_comparison_plots_total_velocity_arrow": True,
        "make_comparison_plots_total_velocity": True,
        "make_comparison_plots_vx": False,
        "make_comparison_plots_vy": False,
        "make_comparison_plots_vz": False,
        "make_comparison_plots_pressure": False,
        "make_comparison_plots_turbvisc_arrow": True,
        "make_comparison_plots_turbvisc": True,
        "make_new_angle_plots": True,
        "make_new_angle_plots_quiver": True,
        "make_new_angle_plots_total_velocity_arrow": True, 
        "make_new_angle_plots_total_velocity": True,
        "make_new_angle_plots_vx": False,
        "make_new_angle_plots_vy": False,
        "make_new_angle_plots_vz": False,
        "make_new_angle_plots_pressure": False,
        "make_new_angle_plots_turbvisc_arrow": True, 
        "make_new_angle_plots_turbvisc": True,
    },
    "data": {
        "one_file_test": False,
        "density": 1,
        "kinematic_viscosity": 1e-5,
        "data_folder_name": 'data',
        "extension": '.csv',
        "startname_data": 'CFD',
        "startname_meteo": 'meteo_',
        "output_zip_file": 'output.zip'
    },
    "loss_components": {
        "data_loss": True,
        "cont_loss": True,
        "momentum_loss": True,
        "boundary_loss": True,
        "use_weighting": False
    },
    "train_test": {
        "train": False,
        "distributed_training": False,
        "test": True,
        "evaluate": True,
        "evaluate_new_angles": False,
        "test_size": 0.999,
        "random_state": 42,
        "test_size_new_angle": 0.999,
        "new_angles": [0,15,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,75,90,105,120,135,150,165,180]
    },
    "chosen_machine": "mac",
    "chosen_optimizer": "adam_optimizer",
    # "base_folder_names": ["01112023_adam_datalossonly_infinite_64neurons", "21102023_adam_datalossonly_infinite", "30102023_adam_datalossonly_infinite_tanh", "29102023_both_datalosscontloss_infinite","29102023_both_datalosscontloss_infinite_colab"]
    # "base_folder_names": ["11112023_both_datalosscontloss_infinite", "21102023_adam_datalossonly_infinite", "30102023_adam_datalossonly_infinite_tanh", "29102023_both_datalosscontloss_infinite","29102023_both_datalosscontloss_infinite_colab"]
    "base_folder_names": ["11112023_both_datalosscontloss_infinite", "21102023_adam_datalossonly_infinite", "26112023_adam_datalossonly_infinite", "29102023_both_datalosscontloss_infinite"]
    # "base_folder_names": ["26112023_adam_datalossonly_infinite", "29112023_adam_datalossboundaryloss_infinite"]
}

if __name__ == "__main__":
    chosen_machine = config["chosen_machine"]
    if chosen_machine == "lxplus":
        if len(sys.argv) > 2:
            base_directory = sys.argv[1]
            output_zip_file = sys.argv[2]
            main(base_directory, config, output_zip_file)
        else:
            print("Please provide the base directory and output zip file as arguments.")
    else:
        for base_directory_ in config["base_folder_names"]:
            base_directory = os.path.join(config["machine"][chosen_machine], base_directory_)
            print (f"doing {base_directory_} now!")
            main(base_directory, config)