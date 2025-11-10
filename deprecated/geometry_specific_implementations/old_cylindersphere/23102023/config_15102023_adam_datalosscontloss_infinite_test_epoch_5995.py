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
        "train": False,
        "distributed_training": False,
        "test": False,
        "evaluate": False,
        "test_size": 0.2,
        "random_state": 42
    },
    "training": {
        "use_epochs": False,
        "num_epochs": 1000,
        "use_batches": True,
        "force": True,
        "batch_size": 2**13,
        "angle_to_leave_out": 90,
        "loss_diff_threshold": 1e-5,
        "consecutive_count_threshold": 10
    },
    "distributed_training": {
        "master_node": False,
        "rank": 1,
        "world_size": 2,
        "ip_address_master": 'localhost',
        # "ip_address": '10.0.90.194',
        # "ip_address": '10.25.186.51',
        "ip_address": '192.168.1.2',
        "port_number": 29500,
        "backend": 'gloo'
    },
    "machine": {
        "lxplus": '/afs/cern.ch/user/a/abinakbe/PINNs/cylinder_cell',
        "mac": os.path.join(os.path.expanduser("~"), "Dropbox", "School", "Graduate", "CERN", "Temp_Files", "nonlineardynamics", "ameir_PINNs", "cylinder_cell"),
        "workstation": os.path.join('E:\\','ameir', "Dropbox", "School", "Graduate", "CERN", "Temp_Files", "nonlineardynamics", "ameir_PINNs", "cylinder_cell"),
    },
    "plotting": {
        "make_pure_data_plots": True, 
        "make_total_plots": True,
        "make_individual_plots": True,
        "make_evaluation_plots": True,
        "plot_predictions": True,
        "3d_scatter": True,
        "2d_scatter": True,
        "total_velocity": True,
        "plot_predictions_individual": True,
        "3d_scatter_individual": True,
        "2d_scatter_individual": True,
        "total_velocity_individual": True,
        "plot_predictions_evaluation": False,
        "3d_scatter_evaluation": False,
        "2d_scatter_evaluation": True,
        "total_velocity_evaluation": True,
        "2d_scatter_data": False,
        "total_velocity_data": True
    },
    "data": {
        "one_file_test": False,
        "density": 1,
        "kinematic_viscosity": 1e-5,
        "data_folder_name": 'data',
        "extension": '.csv',
        "startname": 'CFD',
        "output_zip_file": 'output.zip'
    },   
    "chosen_machine": "mac",
    "chosen_optimizer": "adam_optimizer",
    "base_folder_name": '15102023_adam_datalosscontloss_infinite_testing'
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
        base_directory = os.path.join(config["machine"][chosen_machine], config["base_folder_name"])
        main(base_directory, config)