from definitions import *
from training import *
from PINN import *
from testing import *

def main(config, output_zip_file=None):
    overall_start_time = time.time()
    today = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    strategy = init_tpu()

    chosen_machine = config["chosen_machine"]
    base_directory = os.path.join(config["machine"][chosen_machine], config["base_folder_name"])
        
    print (f"doing {base_directory} now!")

    output_folder = os.path.join('.', base_directory)
    os.makedirs(output_folder, exist_ok=True)
    
    log_folder = os.path.join(output_folder, 'log_output')
    os.makedirs(log_folder, exist_ok=True)

    model_folder = os.path.join(output_folder, 'model_output')
    os.makedirs(model_folder, exist_ok=True)
    
    model_file_path = os.path.join(model_folder, 'trained_PINN_model')
    
    log_filename = os.path.join(log_folder,f"output_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    sys.stdout = Logger(log_filename)  # Set the logger as the new stdout

    if strategy:
        with strategy.scope():
            model = PINN(input_params=config["training"]["input_params"], output_params=config["training"]["output_params"], hidden_layers=config["training"]["number_of_hidden_layers"], neurons_per_layer=[config["training"]["neuron_number"]] * config["training"]["number_of_hidden_layers"], activation=config["training"]["activation_function"], use_batch_norm=config["training"]["batch_normalization"], dropout_rate=config["training"]["dropout_rate"])
        if config["train_test"]["train"]:
            model = train_model(config, model, model_file_path, log_folder, strategy=strategy)
    else:
        model = PINN(input_params=config["training"]["input_params"], output_params=config["training"]["output_params"], hidden_layers=config["training"]["number_of_hidden_layers"], neurons_per_layer=[config["training"]["neuron_number"]] * config["training"]["number_of_hidden_layers"], activation=config["training"]["activation_function"], use_batch_norm=config["training"]["batch_normalization"], dropout_rate=config["training"]["dropout_rate"])
        if config["train_test"]["train"]:
            model = train_model(config, model, model_file_path, log_folder, strategy=None)

    ###TESTING###
    if config["train_test"]["test"]:
        model_file_path_modf = os.path.join(model_folder, 'trained_PINN_model.pth')
        testing(config, model_file_path, model_file_path_modf, output_folder, today, overall_start_time)
    ###TESTING###

    if output_zip_file is not None:
        shutil.make_archive(output_zip_file[:-4], 'zip', output_folder)

    overall_end_time = time.time()
    overall_elapsed_time = overall_end_time - overall_start_time
    print(f'Overall process completed in {overall_elapsed_time:.2f} seconds')