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

    variables = ['Pressure', 'Velocity:0', 'Velocity:1', 'Velocity:2']
    list_of_directions = ['Points:0','Points:1','Points:2']
    wind_angles = [0, 30, 60, 90, 120, 135, 150, 180]

    if config["plotting"]["make_pure_data_plots"]:
        print (f'starting to plot pure data, time: {(time.time() - overall_start_time):.2f} seconds')
        plot_folder = os.path.join(output_folder, 'data_plots_output')
        os.makedirs(plot_folder, exist_ok=True)
        # if config["plotting"]["2d_scatter_data"]:
        #     variables_to_plot = get_variables_to_plot(list_of_directions, variables)
        #     for wind_angle in wind_angles:
        #         for variable_to_plot in variables_to_plot:
        #             print (f'data plotting {variable_to_plot} for wind_angle = {wind_angle}, time: {(time.time() - overall_start_time):.2f} seconds')
        #             data_plot_scatter_2d(filenames, datafolder_path, wind_angle, plot_folder, variables, variable_to_plot)
        data_plot_scatter_2d_total_velocity(datafolder_path,plot_folder)

    model = PINN().to(device)
    model_folder = os.path.join(output_folder, 'model_output')
    os.makedirs(model_folder, exist_ok=True)
    model_file_path = os.path.join(model_folder, 'trained_PINN_model.pth')
    
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
            mp.spawn(train_model,
                     args=(world_size, model, device, normalized_X_train_tensor, normalized_y_train_tensor, config, batch_size, model_file_path, log_folder, epochs),
                     nprocs=world_size,
                     join=True)
        else:
            model = train_model(config["distributed_training"]["rank"], config["distributed_training"]["world_size"], model, device, normalized_X_train_tensor, normalized_y_train_tensor, config, batch_size, model_file_path, log_folder, epochs=config["training"]["num_epochs"])
    ###TRAINING###

    ###TESTING###
    if config["train_test"]["test"]:
        # Evaluate the model
        print (f'starting to test, time: {(time.time() - overall_start_time):.2f} seconds')
        checkpoint = torch.load(model_file_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        data_folder = os.path.join(output_folder, 'data_output')
        os.makedirs(data_folder, exist_ok=True)

        test_predictions, test_predictions_wind_angle = evaluate_model(model, normalized_X_test_tensor, normalized_y_test_tensor, feature_scaler, target_scaler, data_folder)
        print (f'model evaluated, time: {(time.time() - overall_start_time):.2f} seconds')

        X_test_dataframe = test_predictions[0][0]
        y_test_dataframe =  test_predictions[0][1]
        predictions_dataframe = test_predictions[0][2]

        print (f'starting to plot, time: {(time.time() - overall_start_time):.2f} seconds')
        plot_folder = os.path.join(output_folder, 'plots_output')
        os.makedirs(plot_folder, exist_ok=True)
        plot_scatter_2d_total_velocity(X_test_dataframe,y_test_dataframe,predictions_dataframe, plot_folder)
        # if config["plotting"]["make_total_plots"]:
        #     total_plot_folder = os.path.join(plot_folder, 'total_plots_output')
        #     os.makedirs(total_plot_folder, exist_ok=True)
        #     # Plot the test and all predictions for each variable and save the figures
        #     if config["plotting"]["plot_predictions"]:
        #         print (f'plotting predictions, time: {(time.time() - overall_start_time):.2f} seconds')
        #         plot_predictions(y_test_dataframe, predictions_dataframe, total_plot_folder, variables)
        #     if config["plotting"]["3d_scatter"]:   
        #         print (f'plotting 3d scatter, time: {(time.time() - overall_start_time):.2f} seconds')
        #         plot_scatter_3d(X_test_dataframe, y_test_dataframe, predictions_dataframe, total_plot_folder, variables)
        #     if config["plotting"]["2d_scatter"]:
        #         variables_to_plot = get_variables_to_plot(list_of_directions, variables)
        #         for variable_to_plot in variables_to_plot:
        #             print (f'plotting {variable_to_plot}, time: {(time.time() - overall_start_time):.2f} seconds')
        #             plot_scatter_2d(X_test_dataframe, y_test_dataframe, predictions_dataframe, idx_test, total_plot_folder, variables, variable_to_plot)
        #     if config["plotting"]["total_velocity"]:
        #         plotting_directions = get_list_of_directions(list_of_directions)
        #         for variable_to_plot in plotting_directions:
        #             print (f'plotting {variable_to_plot} for Total Velocity, time: {(time.time() - overall_start_time):.2f} seconds')
        #             plot_scatter_2d_total_velocity(X_test_dataframe, y_test_dataframe, predictions_dataframe, idx_test, total_plot_folder, variables, variable_to_plot)

        # if config["plotting"]["make_individual_plots"]:
        #     for wind_angle, X_test_tensor, y_test_tensor, predictions_tensor in test_predictions_wind_angle:   
        #         # Plot the test and all predictions for each variable and save the figures
        #         if config["plotting"]["plot_predictions_individual"]:
        #             print (f'plotting individual predictions, time: {(time.time() - overall_start_time):.2f} seconds for wind_angle: {wind_angle}')
        #             individual_plot_predictions(wind_angle, y_test_tensor, predictions_tensor, plot_folder, variables)
        #         if config["plotting"]["3d_scatter_individual"]:   
        #             print (f'plotting individual 3d scatter, time: {(time.time() - overall_start_time):.2f} seconds for wind_angle: {wind_angle}')
        #             individual_plot_scatter_3d(wind_angle, X_test_tensor, y_test_tensor, predictions_tensor, plot_folder, variables)
        #         if config["plotting"]["2d_scatter_individual"]:
        #             variables_to_plot = get_variables_to_plot(list_of_directions, variables)
        #             for variable_to_plot in variables_to_plot:
        #                 print (f'plotting individual {variable_to_plot}, time: {(time.time() - overall_start_time):.2f} seconds for wind_angle: {wind_angle}')
        #                 individual_plot_scatter_2d(wind_angle, X_test_tensor, y_test_tensor, predictions_tensor, plot_folder, variables, variable_to_plot)
        #         if config["plotting"]["total_velocity_individual"]:
        #             plotting_directions = get_list_of_directions(list_of_directions)
        #             for variable_to_plot in plotting_directions:
        #                 print (f'plotting individual {variable_to_plot} for Total Velocity, time: {(time.time() - overall_start_time):.2f} seconds for wind_angle: {wind_angle}')
        #                 individual_plot_scatter_2d_total_velocity(wind_angle, X_test_tensor, y_test_tensor, predictions_tensor, plot_folder, variables, variable_to_plot)
    ###TESTING###

    ###EVALUATING###
    if config["train_test"]["evaluate"]:
        print (f'starting to evaluate, time: {(time.time() - overall_start_time):.2f} seconds')
        # Evaluate the model
        checkpoint = torch.load(model_file_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        data_folder = os.path.join(output_folder, 'data_output_for_skipped_angle')
        os.makedirs(data_folder, exist_ok=True)

        test_predictions, test_predictions_wind_angle = evaluate_model(model, normalized_X_test_tensor_skipped, normalized_y_test_tensor_skipped, feature_scaler, target_scaler, data_folder)
        print (f'model evaluated, time: {(time.time() - overall_start_time):.2f} seconds')

        print (f'starting to plot, time: {(time.time() - overall_start_time):.2f} seconds')
        plot_folder = os.path.join(output_folder, 'plots_output')
        os.makedirs(plot_folder, exist_ok=True)
        for i in test_predictions_wind_angle:
            wind_angle = i[0]
            X_test_dataframe = i[1]
            y_test_dataframe =  i[2]
            predictions_dataframe = i[3]   
            print (X_test_dataframe.shape)
            plot_prediction_2d_total_velocity(X_test_dataframe,y_test_dataframe,predictions_dataframe,wind_angle, plot_folder)

        # if config["plotting"]["make_evaluation_plots"]:
        #     plot_folder = os.path.join(output_folder, 'plots_output_for_skipped_angle')
        #     os.makedirs(plot_folder, exist_ok=True)
        #     for wind_angle, positions, normalized_y_test_tensor, normalized_predictions in test_predictions_wind_angle:
        #         predictions = inverse_transform_targets(normalized_predictions.cpu(), target_scaler)
        #         y_test_tensor = inverse_transform_targets(normalized_y_test_tensor.cpu(), target_scaler)    
        #         # Plot the test and all predictions for each variable and save the figures
        #         if config["plotting"]["plot_predictions_evaluation"]:
        #             print (f'plotting individual predictions, time: {(time.time() - overall_start_time):.2f} seconds for wind_angle: {wind_angle}')
        #             individual_plot_predictions(wind_angle, y_test_tensor, predictions, plot_folder, variables)
        #         if config["plotting"]["3d_scatter_evaluation"]:   
        #             print (f'plotting individual 3d scatter, time: {(time.time() - overall_start_time):.2f} seconds for wind_angle: {wind_angle}')
        #             individual_plot_scatter_3d(wind_angle, positions, y_test_tensor, predictions, plot_folder, variables)
        #         if config["plotting"]["2d_scatter_evaluation"]:
        #             variables_to_plot = get_variables_to_plot(list_of_directions, variables)
        #             for variable_to_plot in variables_to_plot:
        #                 print (f'plotting individual {variable_to_plot}, time: {(time.time() - overall_start_time):.2f} seconds for wind_angle: {wind_angle}')
        #                 individual_plot_scatter_2d(wind_angle, positions, y_test_tensor, predictions, plot_folder, variables, variable_to_plot)
        #         if config["plotting"]["total_velocity_evaluation"]:
        #             plotting_directions = get_list_of_directions(list_of_directions)
        #             for variable_to_plot in plotting_directions:
        #                 print (f'plotting individual {variable_to_plot} for Total Velocity, time: {(time.time() - overall_start_time):.2f} seconds for wind_angle: {wind_angle}')
        #                 individual_plot_scatter_2d_total_velocity(wind_angle, positions, y_test_tensor, predictions, plot_folder, variables, variable_to_plot)
    ###EVALUATING###

    if output_zip_file is not None:
        shutil.make_archive(output_zip_file[:-4], 'zip', output_folder)

    overall_end_time = time.time()
    overall_elapsed_time = overall_end_time - overall_start_time
    print(f'Overall process completed in {overall_elapsed_time:.2f} seconds')