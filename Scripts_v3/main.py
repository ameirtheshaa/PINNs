from definitions import *
from training import *
from PINN import *
from plotting import *

def main(base_directory, config, output_zip_file=None):
    overall_start_time = time.time()
    
    output_folder = os.path.join(base_directory, 'analyses_output')
    os.makedirs(output_folder, exist_ok=True)

    log_folder = os.path.join(output_folder, 'log_output')
    os.makedirs(log_folder, exist_ok=True)
    
    log_filename = os.path.join(log_folder,f"output_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    sys.stdout = Logger(log_filename)  # Set the logger as the new stdout
    
    device = print_and_set_available_gpus()

    chosen_machine_key = config["chosen_machine"]
    
    datafolder_path = os.path.join(config["machine"][chosen_machine_key], config["data"]["data_folder_name"])
    filenames = get_filenames_from_folder(datafolder_path, config["data"]["extension"], config["data"]["startname"])

    normalized_X_train_tensor, normalized_y_train_tensor, normalized_X_test_tensor, normalized_y_test_tensor, normalized_X_train_tensor_skipped, normalized_y_train_tensor_skipped, normalized_X_test_tensor_skipped, normalized_y_test_tensor_skipped, feature_scaler, target_scaler = load_data(filenames, base_directory, datafolder_path, device, config)

    if config["plotting"]["make_pure_data_plots"]:
        print (f'starting to plot pure data, time: {(time.time() - overall_start_time):.2f} seconds')
        plot_folder = os.path.join(output_folder, 'data_plots_output')
        os.makedirs(plot_folder, exist_ok=True)
        data_plot_scatter_2d(datafolder_path,plot_folder)

    model = PINN().to(device)
    model_folder = os.path.join(output_folder, 'model_output')
    os.makedirs(model_folder, exist_ok=True)
    model_file_path = os.path.join(model_folder, 'trained_PINN_model.pth')

    if config["training"]["change_activation_function"]:
        activation_function = config["training"]["activation_function"]
    else:
        activation_function = 'relu'
    
    ###TRAINING###
    if config["train_test"]["train"]:
        device, batch_size = select_device_and_batch_size(model, normalized_X_train_tensor, device)
        required_memory = estimate_memory(model, normalized_X_train_tensor, batch_size)
        print(f"Estimated Memory Requirement for the Batch Size: {required_memory:.2f} GB")
        required_memory = estimate_memory(model, normalized_X_train_tensor, batch_size=len(normalized_X_train_tensor))
        print(f"Estimated Memory Requirement for the Full Size: {required_memory:.2f} GB")
        
        # train and save the trained model
        if config["train_test"]["distributed_training"]:
            world_size = config["distributed_training"]["world_size"]
            epochs=config["training"]["num_epochs"]
            mp.spawn(train_model, args=(world_size, model, activation_function, device, normalized_X_train_tensor, normalized_y_train_tensor, config, batch_size, model_file_path, log_folder, epochs),nprocs=world_size,join=True)
        else:
            model = train_model(config["distributed_training"]["rank"], config["distributed_training"]["world_size"], model, activation_function, device, normalized_X_train_tensor, normalized_y_train_tensor, config, batch_size, model_file_path, log_folder, epochs=config["training"]["num_epochs"])
    ###TRAINING###

    ###TESTING###
    if config["train_test"]["test"]:
        # Evaluate the model
        print (f'starting to test, time: {(time.time() - overall_start_time):.2f} seconds')
        checkpoint = torch.load(model_file_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        data_folder = os.path.join(output_folder, 'data_output')
        os.makedirs(data_folder, exist_ok=True)

        test_predictions, test_predictions_wind_angle = evaluate_model(model, activation_function, normalized_X_test_tensor, normalized_y_test_tensor, feature_scaler, target_scaler, data_folder)
        print (f'model evaluated, time: {(time.time() - overall_start_time):.2f} seconds')

        print (f'starting to plot, time: {(time.time() - overall_start_time):.2f} seconds')
        plot_folder = os.path.join(output_folder, 'plots_output')
        os.makedirs(plot_folder, exist_ok=True)
        for wind_angle, df in test_predictions_wind_angle:
            if wind_angle not in config["training"]["angle_to_leave_out"]:
                plot_prediction_2d(df,wind_angle, plot_folder)
    ###TESTING###

    ###EVALUATING###
    if config["train_test"]["evaluate"]:
        print (f'starting to evaluate, time: {(time.time() - overall_start_time):.2f} seconds')
        # Evaluate the model
        checkpoint = torch.load(model_file_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        data_folder = os.path.join(output_folder, 'data_output_for_skipped_angle')
        os.makedirs(data_folder, exist_ok=True)

        test_predictions_skipped, test_predictions_wind_angle_skipped = evaluate_model_skipped(config, model, activation_function, normalized_X_test_tensor_skipped, normalized_y_test_tensor_skipped, feature_scaler, target_scaler, data_folder)
        print (f'model evaluated, time: {(time.time() - overall_start_time):.2f} seconds')

        print (f'starting to plot, time: {(time.time() - overall_start_time):.2f} seconds')
        plot_folder_skipped = os.path.join(output_folder, 'plots_output_for_skipped_angle(s)')
        os.makedirs(plot_folder_skipped, exist_ok=True)
        for wind_angle, df in test_predictions_wind_angle_skipped:
            if wind_angle in config["training"]["angle_to_leave_out"]:
                plot_prediction_2d(df,wind_angle, plot_folder_skipped)
    ###EVALUATING###

    if output_zip_file is not None:
        shutil.make_archive(output_zip_file[:-4], 'zip', output_folder)

    overall_end_time = time.time()
    overall_elapsed_time = overall_end_time - overall_start_time
    print(f'Overall process completed in {overall_elapsed_time:.2f} seconds')