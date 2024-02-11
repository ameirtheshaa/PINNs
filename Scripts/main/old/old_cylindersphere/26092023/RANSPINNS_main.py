from definitions import *
from nn import *
from plotting import *
import importlib.util
import os

def main(base_directory):
    overall_start_time = time.time()

    config_file_path = os.path.join(base_directory, 'config.py')
    spec = importlib.util.spec_from_file_location("config", config_file_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    
    log_filename = os.path.join(base_directory,f"output_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    sys.stdout = Logger(log_filename)  # Set the logger as the new stdout
    
    device = print_and_set_available_gpus()
    
    datafolder_path = config.config["data_folder"]
    filenames = get_filenames_from_folder(datafolder_path, config.config["extension"], config.config["startname"])
    
    for filename in sorted(filenames):
        print (filename)
    
        features, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, idx_train, idx_test, output_folder = load_data(filename, base_directory, datafolder_path, device, config.config)
    
        model = PINN().to(device)
        model_file_path = os.path.join(output_folder, 'trained_PINN_model.pth')
        
        if config.config["train_test"]["train"]:
            device, batch_size = select_device_and_batch_size(model, X_train_tensor)
            required_memory = estimate_memory(model, X_train_tensor, batch_size)
            print(f"Estimated Memory Requirement for the Batch Size: {required_memory / (1024 ** 3):.2f} GB")
            required_memory = estimate_memory(model, X_train_tensor, batch_size=len(X_train_tensor))
            print(f"Estimated Memory Requirement for the Full Size: {required_memory / (1024 ** 3):.2f} GB")
                   
            # train and save the trained model
            model = train_model(model, X_train_tensor, y_train_tensor, config.config, batch_size, epochs=config.config["epochs"])
            torch.save(model.state_dict(), model_file_path)
        
        if config.config["train_test"]["test"]:
            # Evaluate the model
            model.load_state_dict(torch.load(model_file_path, map_location=device))
            variables = ['Pressure', 'Velocity:0', 'Velocity:1', 'Velocity:2']
            metrics_df, predictions = evaluate_model(model, variables, features, idx_test, X_test_tensor, y_test_tensor, output_folder)

            predictions = predictions.cpu()
            
            # Save the MSE results to a CSV file in the output folder
            mse_file_path = os.path.join(output_folder, 'metrics.csv')
            metrics_df.to_csv(mse_file_path, index=False)

            # Plot the test and predictions for each variable and save the figures
            plot_predictions(y_test_tensor, predictions, output_folder, variables)
            plot_3d_scatter_comparison(features, y_test_tensor, predictions, output_folder, variables)
            list_of_directions = ['Points:0','Points:1','Points:2']
            variables_to_plot = get_variables_to_plot(list_of_directions, variables)
            for variable_to_plot in variables_to_plot:
                plot_2d_contour_comparison(features, y_test_tensor, predictions, idx_test, output_folder, variables, variable_to_plot)
                plot_total_velocity(features, y_test_tensor, predictions, idx_test, output_folder, variables, variable_to_plot)

            overall_end_time = time.time()
            overall_elapsed_time = overall_end_time - overall_start_time
            print(f'Overall process completed in {overall_elapsed_time:.2f} seconds')
        
        if config.config["one_file_test"]:
            break

if __name__ == "__main__":
    args = parse_arguments()
    base_directory = args.base_directory
    main(base_directory)