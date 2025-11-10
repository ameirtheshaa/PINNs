from main import *
from config import *
import time 

config["chosen_machine"] = "CREATE"
config["train_test"]["force_device"] = 'cpu'
config["train_test"]["test_size"] = 0.1
config["train_test"]["train"] = True
config["plotting"]["make_logging_plots"] = True
config["train_test"]["test"] = True
config["train_test"]["evaluate"] = True
# # config["plotting"]["make_div_plots"] = True
# # config["plotting"]["make_RANS_plots"] = True
# config["plotting"]["save_csv_predictions"] = True
# config["plotting"]["save_vtk"] = True
config["plotting"]["make_plots"] = True

# config["training"]["use_PCA"] = True

config["training"]["use_fft"] = True
config["training"]["features_factor"] = 2
config["training"]["targets_factor"] = 2

# config["training"]["angles_to_train"] = [0,150]
config["training"]["use_epochs"] = True
config["training"]["num_epochs"] = 1000
config["training"]["print_epochs"] = 1
config["training"]["save_metrics"] = 1

config["training"]["total_model_eval"] = 10

# config["training"]["number_of_hidden_layers"] = 10

config["loss_components"]["data_loss"] = True

neuron_tests = [1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,155,160,165,170,185,190,195,200]
learning_rates = [1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10]
neuron_tests = [128]
learning_rates = [1e-3]

for i in neuron_tests:
    for j in learning_rates:
        config["training"]["neuron_number"] = i
        config["adam_optimizer"]["learning_rate"] = j
        if config["training"]["use_fft"]:
            config["base_folder_name"] = f"fft_test_{i}_{j}_{time.time()}"
        elif config["training"]["use_PCA"]:
            config["base_folder_name"] = f"pca_test_{i}_{j}_{time.time()}"
        else:
            config["base_folder_name"] = f"test_{i}_{j}_{time.time()}"
        print (i,j,config["base_folder_name"])
        if __name__ == "__main__":
            device, data_dict, input_params, output_params, model = initialize_data(config)
            chosen_machine = config["chosen_machine"]
            base_directory = os.path.join(config["machine"][chosen_machine], config["base_folder_name"])
            main(base_directory, config, device, data_dict, input_params, output_params, model)