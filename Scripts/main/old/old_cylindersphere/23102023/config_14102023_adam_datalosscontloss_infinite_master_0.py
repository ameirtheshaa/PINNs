from main import *

config = {
    "lbfgs_optimizer": {
        "type": "LBFGS",
        "learning_rate": 0.001,
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
        "max_iter": 200000,
        "max_eval": 50000,
        "history_size": 50,
        "tolerance_grad": 1e-05,
        "tolerance_change": 0.5 * np.finfo(float).eps,
        "line_search_fn": "strong_wolfe"
    },
    "loss_components": {
        "data_loss": True,
        "cont_loss": True,
        "momentum_loss": False
    },
    "train_test": {
        "train": True,
        "test": False,
        "evaluate": False,
        "test_size": 0.2,
        "random_state": 42
    },
    "batches": {
        "use_batches": True,
        "force": True,
        "batch_size": 2**13
    },
    "lxplus": {
        "base_name": '/afs/cern.ch/user/a/abinakbe/PINNs/cylinder_cell',
        
    },
    "mac": {
        "base_name": os.path.join(os.path.expanduser("~"), "Dropbox", "School", "Graduate", "CERN", "Temp_Files", "nonlineardynamics", "ameir_PINNs", "cylinder_cell"),
    },
    "workstation_CREATE": {
        "base_name": os.path.join('E:\\','ameir', "Dropbox", "School", "Graduate", "CERN", "Temp_Files", "nonlineardynamics", "ameir_PINNs", "cylinder_cell"),
    },
    "workstation_UofA": {
        "base_name": os.path.join('C:\\', "Dropbox", "School", "Graduate", "CERN", "Temp_Files", "nonlineardynamics", "ameir_PINNs", "cylinder_cell"),
    },
    "home_PC": {
        "base_name": os.path.join('C:\\', "Dropbox", "School", "Graduate", "CERN", "Temp_Files", "nonlineardynamics", "ameir_PINNs", "cylinder_cell"),
    },
    "parallel_training": {
        "distributed_training": True,
        "ip_address_master": '192.168.0.63',
        "world_size": 5,
        "backend": 'gloo'
    },
    "use_epoch": False,
    "make_individual_plots": True,
    "one_file_test": False,
    "chosen_machine": "mac",
    "chosen_optimizer": "adam_optimizer",
    "angle_to_leave_out": 90,
    "epochs": 1000,
    "loss_diff_threshold": 1e-5,
    "consecutive_count_threshold": 10,
    "density": 1,
    "kinematic_viscosity": 1e-5,    
    "extension": '.csv',
    "startname": 'CFD',
    "output_zip_file": 'output.zip',
    "data_folder_name": 'data',
    "base_folder_name": '14102023_adam_datalosscontloss_infinite'
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
        base_directory = os.path.join(config[chosen_machine]["base_name"], config["base_folder_name"])
        output_zip_file = os.path.join(base_directory,config["output_zip_file"])
        main(base_directory, config, output_zip_file)