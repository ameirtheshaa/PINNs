from definitions import *
from training import *
from PINN import *
from plotting import *
from testing import *
from physics import *

def main(base_directory, config, output_zip_file=None):
    overall_start_time = time.time()
    today = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    output_folder = os.path.join('.', base_directory)
    os.makedirs(output_folder, exist_ok=True)
    
    log_folder = os.path.join(output_folder, 'log_output')
    os.makedirs(log_folder, exist_ok=True)

    model_folder = os.path.join(output_folder, 'model_output')
    os.makedirs(model_folder, exist_ok=True)

    model_file_path = os.path.join(model_folder, 'trained_PINN_model.pth')
    
    log_filename = os.path.join(log_folder,f"output_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    sys.stdout = Logger(log_filename)  # Set the logger as the new stdout

    device = print_and_set_available_gpus()

    data_dict = load_data(config, device)

    model = PINN(input_params=config["training"]["input_params"], output_params=config["training"]["output_params"], hidden_layers=config["training"]["number_of_hidden_layers"], neurons_per_layer=[config["training"]["neuron_number"]] * config["training"]["number_of_hidden_layers"], activation=config["training"]["activation_function"], use_batch_norm=config["training"]["batch_normalization"], dropout_rate=config["training"]["dropout_rate"]).to(device)
    
    ###TRAINING###
    if config["train_test"]["train"]:
        model = train_model(model, device, config, data_dict, model_file_path, log_folder)
    ###TRAINING###

    ###TESTING###
    if config["train_test"]["test"]:
        testing(model, device, config, data_dict, model_file_path, output_folder, today, overall_start_time)
    ###TESTING###

    ###EVALUATING###
    if config["train_test"]["evaluate"]:
        evaluation(model, device, config, data_dict, model_file_path, output_folder, today, overall_start_time)
    ###EVALUATING###

    ###NEW ANGLES###
    if config["train_test"]["evaluate_new_angles"]:
        evaluation_new_angles(model, device, config, data_dict, model_file_path, output_folder, today, overall_start_time)
    ###NEW ANGLES###

    ##OTHERS###
    if config["plotting"]["make_logging_plots"]:
        process_logging_statistics(log_folder)
    if config["plotting"]["make_div_plots"]:
        evaluation_physics(model, device, config, data_dict, model_file_path, output_folder, today, overall_start_time, 'Div')     
    if config["plotting"]["make_RANS_plots"]:
        evaluation_physics(model, device, config, data_dict, model_file_path, output_folder, today, overall_start_time, 'RANS')       
    if config["plotting"]["make_data_plots"]:
        make_pure_data_plots(config, output_folder, today, overall_start_time)
    ##OTHERS###

    if output_zip_file is not None:
        shutil.make_archive(output_zip_file[:-4], 'zip', output_folder)

    overall_end_time = time.time()
    overall_elapsed_time = overall_end_time - overall_start_time
    print(f'Overall process completed in {overall_elapsed_time:.2f} seconds')