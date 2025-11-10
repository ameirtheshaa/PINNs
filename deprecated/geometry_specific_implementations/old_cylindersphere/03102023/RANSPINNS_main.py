from definitions import *
from nn import *
from plotting import *
import importlib.util
import os

def main(base_directory, config):
    overall_start_time = time.time()
    
    output_folder = os.path.join(base_directory, 'analyses_output')
    os.makedirs(output_folder, exist_ok=True)

    # config_file_path = os.path.join(base_directory, 'config.py')
    # spec = importlib.util.spec_from_file_location("config", config_file_path)
    # config = importlib.util.module_from_spec(spec)
    # spec.loader.exec_module(config)

    log_folder = os.path.join(output_folder, 'log_output')
    os.makedirs(log_folder, exist_ok=True)
    
    log_filename = os.path.join(log_folder,f"output_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    sys.stdout = Logger(log_filename)  # Set the logger as the new stdout
    
    device = print_and_set_available_gpus()
    
    datafolder_path = config["data_folder"]
    filenames = get_filenames_from_folder(datafolder_path, config["extension"], config["startname"])
    
    features, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, idx_train, idx_test = load_data(filenames, base_directory, datafolder_path, device, config)

    model = PINN().to(device)
    model_folder = os.path.join(output_folder, 'model_output')
    os.makedirs(model_folder, exist_ok=True)
    model_file_path = os.path.join(model_folder, 'trained_PINN_model.pth')
    
    ###TRAINING###
    if config["train_test"]["train"]:
        device, batch_size = select_device_and_batch_size(model, X_train_tensor)
        required_memory = estimate_memory(model, X_train_tensor, batch_size)
        print(f"Estimated Memory Requirement for the Batch Size: {required_memory / (1024 ** 3):.2f} GB")
        required_memory = estimate_memory(model, X_train_tensor, batch_size=len(X_train_tensor))
        print(f"Estimated Memory Requirement for the Full Size: {required_memory / (1024 ** 3):.2f} GB")
               
        # train and save the trained model
        model = train_model(model, X_train_tensor, y_train_tensor, config, batch_size, model_file_path, epochs=config["epochs"])
    ###TRAINING###

    ###TESTING###
    if config["train_test"]["test"]:
        # Evaluate the model
        model.load_state_dict(torch.load(model_file_path, map_location=device))
        variables = ['Pressure', 'Velocity:0', 'Velocity:1', 'Velocity:2']
        data_folder = os.path.join(output_folder, 'data_output')
        os.makedirs(data_folder, exist_ok=True)

        predictions, test_predictions_wind_angle = evaluate_model(model, variables, features, idx_test, X_test_tensor, y_test_tensor, data_folder)

        predictions = predictions.cpu()

        plot_folder = os.path.join(output_folder, 'plots_output')
        os.makedirs(plot_folder, exist_ok=True)
        # Plot the test and all predictions for each variable and save the figures
        plot_predictions(y_test_tensor, predictions, plot_folder, variables)
        plot_3d_scatter_comparison(features, y_test_tensor, predictions, plot_folder, variables)
        list_of_directions = ['Points:0','Points:1','Points:2']
        variables_to_plot = get_variables_to_plot(list_of_directions, variables)
        for variable_to_plot in variables_to_plot:
            plot_2d_contour_comparison(features, y_test_tensor, predictions, idx_test, plot_folder, variables, variable_to_plot)
            plot_total_velocity(features, y_test_tensor, predictions, idx_test, plot_folder, variables, variable_to_plot)

        if config["make_individual_plots"]:
            for wind_angle, positions, y_test_tensor, predictions in test_predictions_wind_angle:
                individual_plot_predictions(wind_angle, y_test_tensor, predictions, plot_folder, variables)
                individual_plot_3d_scatter_comparison(wind_angle, positions, y_test_tensor, predictions, plot_folder, variables)
                list_of_directions = ['Points:0','Points:1','Points:2']
                variables_to_plot = get_variables_to_plot(list_of_directions, variables)
                for variable_to_plot in variables_to_plot:
                    individual_plot_2d_contour_comparison(wind_angle, positions, y_test_tensor, predictions, plot_folder, variables, variable_to_plot)
                    individual_plot_total_velocity(wind_angle, positions, y_test_tensor, predictions, plot_folder, variables, variable_to_plot)
    ###TESTING###

    ###EVALUATING###
    if config["train_test"]["evaluate"]:
        features, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, idx_train, idx_test = load_skipped_angle_data(filenames, base_directory, datafolder_path, device, config)
        # Evaluate the model
        model.load_state_dict(torch.load(model_file_path, map_location=device))
        variables = ['Pressure', 'Velocity:0', 'Velocity:1', 'Velocity:2']
        data_folder = os.path.join(output_folder, 'data_output_for_skipped_angle')
        os.makedirs(data_folder, exist_ok=True)
        plot_folder = os.path.join(output_folder, 'plots_output_for_skipped_angle')
        os.makedirs(plot_folder, exist_ok=True)

        predictions, _ = evaluate_model(model, variables, features, idx_test, X_test_tensor, y_test_tensor, data_folder)
        predictions = predictions.cpu()
        # Plot the test and all predictions for each variable and save the figures
        plot_predictions(y_test_tensor, predictions, plot_folder, variables)
        plot_3d_scatter_comparison(features, y_test_tensor, predictions, plot_folder, variables)
        list_of_directions = ['Points:0','Points:1','Points:2']
        variables_to_plot = get_variables_to_plot(list_of_directions, variables)
        for variable_to_plot in variables_to_plot:
            plot_2d_contour_comparison(features, y_test_tensor, predictions, idx_test, plot_folder, variables, variable_to_plot)
            plot_total_velocity(features, y_test_tensor, predictions, idx_test, plot_folder, variables, variable_to_plot)
    ###EVALUATING###

    overall_end_time = time.time()
    overall_elapsed_time = overall_end_time - overall_start_time
    print(f'Overall process completed in {overall_elapsed_time:.2f} seconds')

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        base_directory = sys.argv[1]
        main(base_directory)
    else:
        print("Please provide the base directory as an argument.")