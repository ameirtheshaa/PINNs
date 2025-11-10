from definitions import *
from training import *
from PINN import *
from plotting import *
from definitions_new import *

def main(base_directory, config, output_zip_file=None):
    overall_start_time = time.time()
    
    output_folder = os.path.join('.', base_directory)
    os.makedirs(output_folder, exist_ok=True)

    log_folder = os.path.join(output_folder, 'log_output')
    os.makedirs(log_folder, exist_ok=True)
    
    log_filename = os.path.join(log_folder,f"output_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    sys.stdout = Logger(log_filename)  # Set the logger as the new stdout
    
    device = print_and_set_available_gpus()

    chosen_machine_key = config["chosen_machine"]
    
    datafolder_path = os.path.join(config["machine"][chosen_machine_key], config["data"]["data_folder_name"])
    filenames = get_filenames_from_folder(datafolder_path, config["data"]["extension"], config["data"]["startname_data"])
    meteo_filenames = get_filenames_from_folder(datafolder_path, config["data"]["extension"], config["data"]["startname_meteo"])
    geometry_filename = os.path.join(datafolder_path,'scaled_cylinder_sphere.stl')

    normalized_X_train_tensor, normalized_y_train_tensor, normalized_X_test_tensor, normalized_y_test_tensor, normalized_X_train_tensor_skipped, normalized_y_train_tensor_skipped, normalized_X_test_tensor_skipped, normalized_y_test_tensor_skipped, feature_scaler, target_scaler = load_data(filenames, base_directory, datafolder_path, device, config)

    today = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # if config["plotting"]["make_pure_data_plots"]:
    #     print (f'starting to plot pure data, time: {(time.time() - overall_start_time):.2f} seconds')
    #     plot_folder = os.path.join(output_folder, f'data_plots_output_{today}')
    #     df = load_plotting_data(filenames, base_directory, datafolder_path, config)
    #     angles = config["training"]["all_angles"]
    #     plot_data_2d(df,angles,geometry_filename,plot_folder,single=True)
    #     print (f'plot pure data done, time: {(time.time() - overall_start_time):.2f} seconds')

    model = PINN(input_params=config["training"]["input_params"], output_params=config["training"]["output_params"], hidden_layers=config["training"]["number_of_hidden_layers"], neurons_per_layer=[config["training"]["neuron_number"]] * config["training"]["number_of_hidden_layers"], activation=config["training"]["activation_function"], use_batch_norm=config["training"]["batch_normalization"], dropout_rate=config["training"]["dropout_rate"]).to(device)
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
            mp.spawn(train_model, args=(world_size, model, activation_function, device, normalized_X_train_tensor, normalized_y_train_tensor, normalized_X_test_tensor, normalized_y_test_tensor, normalized_X_test_tensor_skipped, normalized_y_test_tensor_skipped, config, feature_scaler, target_scaler, batch_size, model_file_path, log_folder, meteo_filenames, datafolder_path, epochs),nprocs=world_size,join=True)
        else:
            model = train_model(config["distributed_training"]["rank"], config["distributed_training"]["world_size"], model, device, normalized_X_train_tensor, normalized_y_train_tensor, normalized_X_test_tensor, normalized_y_test_tensor, normalized_X_test_tensor_skipped, normalized_y_test_tensor_skipped, config, feature_scaler, target_scaler, batch_size, model_file_path, log_folder, meteo_filenames, datafolder_path, epochs=config["training"]["num_epochs"])
    ###TRAINING###

    ###TESTING###
    if config["train_test"]["test"]:
        # Evaluate the model
        print (f'starting to test, time: {(time.time() - overall_start_time):.2f} seconds')
        checkpoint = torch.load(model_file_path, map_location=device)
        print (f'Model Tested at Epoch = {checkpoint["epoch"]} and Training Completed = {checkpoint["training_completed"]}')
        model.load_state_dict(checkpoint['model_state_dict'])
        data_folder = os.path.join(output_folder, f'data_output_{today}_{checkpoint["epoch"]}')
        os.makedirs(data_folder, exist_ok=True)

        test_predictions, test_predictions_wind_angle = evaluate_model(config, model, normalized_X_test_tensor, normalized_y_test_tensor, feature_scaler, target_scaler, data_folder)
        print (f'model evaluated, time: {(time.time() - overall_start_time):.2f} seconds')

        print (f'starting to plot, time: {(time.time() - overall_start_time):.2f} seconds')
        plot_folder = os.path.join(output_folder, f'plots_output_{today}_{checkpoint["epoch"]}')
        os.makedirs(plot_folder, exist_ok=True)
        for wind_angle, df in test_predictions_wind_angle:
            if wind_angle not in config["training"]["angle_to_leave_out"]:
                plot_prediction_2d(df,wind_angle,config,plot_folder)
    ###TESTING###

    ###EVALUATING###
    if config["train_test"]["evaluate"]:
        print (f'starting to evaluate, time: {(time.time() - overall_start_time):.2f} seconds')
        # Evaluate the model
        checkpoint = torch.load(model_file_path, map_location=device)
        print (f'Model Evaluated at Epoch = {checkpoint["epoch"]} and Training Completed = {checkpoint["training_completed"]}')
        model.load_state_dict(checkpoint['model_state_dict'])
        data_folder = os.path.join(output_folder, f'data_output_for_skipped_angle_{today}_{checkpoint["epoch"]}')
        os.makedirs(data_folder, exist_ok=True)

        test_predictions_skipped, test_predictions_wind_angle_skipped = evaluate_model_skipped(config, model, normalized_X_test_tensor_skipped, normalized_y_test_tensor_skipped, feature_scaler, target_scaler, data_folder)
        print (f'model evaluated, time: {(time.time() - overall_start_time):.2f} seconds')

        print (f'starting to plot, time: {(time.time() - overall_start_time):.2f} seconds')
        plot_folder_skipped = os.path.join(output_folder, f'plots_output_for_skipped_angle(s)_{today}_{checkpoint["epoch"]}')
        os.makedirs(plot_folder_skipped, exist_ok=True)
        for wind_angle, df in test_predictions_wind_angle_skipped:
            if wind_angle in config["training"]["angle_to_leave_out"]:
                plot_prediction_2d(df,wind_angle,config,plot_folder_skipped)
    ###EVALUATING###

    ###NEW ANGLES###
    if config["train_test"]["evaluate_new_angles"]:
        print (f'starting to evaluate new angles, time: {(time.time() - overall_start_time):.2f} seconds')
        # Evaluate the model
        checkpoint = torch.load(model_file_path, map_location=device)
        print (f'Model Evaluated at Epoch = {checkpoint["epoch"]} and Training Completed = {checkpoint["training_completed"]}')
        model.load_state_dict(checkpoint['model_state_dict'])

        wind_angles = config["train_test"]["new_angles"]
        for wind_angle in wind_angles:
            normalized_X_test_tensor = load_data_new_angle(filenames, base_directory, datafolder_path, device, config, feature_scaler, target_scaler, wind_angle)
            df = evaluate_model_new_angles(config, wind_angle, model, normalized_X_test_tensor, feature_scaler, target_scaler)
            print (f'model evaluated for angle = {wind_angle}, time: {(time.time() - overall_start_time):.2f} seconds')
            if config["plotting"]["make_new_angle_plots"]:
                print (f'starting to plot for angle = {wind_angle}, time: {(time.time() - overall_start_time):.2f} seconds')
                plot_folder_new_angle = os.path.join(output_folder, f'plots_output_for_new_angles_{today}_{checkpoint["epoch"]}')
                os.makedirs(plot_folder_new_angle, exist_ok=True)
                plot_new_angles_2d(df,wind_angle,config,plot_folder_new_angle)
    ###NEW ANGLES###

    

    

    print (f'starting to test, time: {(time.time() - overall_start_time):.2f} seconds')
    checkpoint = torch.load(model_file_path, map_location=device)
    print (f'Model Tested at Epoch = {checkpoint["epoch"]} and Training Completed = {checkpoint["training_completed"]}')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # wind_angles = config["training"]["angles_to_train"]
    # data_folder = os.path.join(output_folder, f'data_output_{today}_{checkpoint["epoch"]}')
    # df = evaluate_model(config, model, wind_angles, normalized_X_test_tensor, feature_scaler, target_scaler, normalized_y_test_tensor, data_folder)
    # print (f'model evaluated, time: {(time.time() - overall_start_time):.2f} seconds')

    # print (f'starting to plot, time: {(time.time() - overall_start_time):.2f} seconds')
    # plot_folder = os.path.join(output_folder, f'plots_output_{today}_{checkpoint["epoch"]}')
    # plot_data_2d(df,wind_angles,geometry_filename,plot_folder,single=False)


    # wind_angles = config["training"]["angle_to_leave_out"]
    # data_folder = os.path.join(output_folder, f'data_output_for_skipped_angle_{today}_{checkpoint["epoch"]}')
    # df = evaluate_model(config, model, wind_angles, normalized_X_test_tensor_skipped, feature_scaler, target_scaler, normalized_y_test_tensor_skipped, data_folder)
    # print (f'model evaluated skipped, time: {(time.time() - overall_start_time):.2f} seconds')

    # print (f'starting to plot, time: {(time.time() - overall_start_time):.2f} seconds')
    # plot_folder_skipped = os.path.join(output_folder, f'plots_output_for_skipped_angle(s)_{today}_{checkpoint["epoch"]}')
    # plot_data_2d(df,wind_angles,geometry_filename,plot_folder_skipped,single=False)

    print (f'starting to evaluate new angles, time: {(time.time() - overall_start_time):.2f} seconds')
    wind_angles = config["train_test"]["new_angles"]
    normalized_X_test_tensor_new = load_data_new_angles(filenames, wind_angles, datafolder_path, device, config, feature_scaler, target_scaler)
    df = evaluate_model(config, model, wind_angles, normalized_X_test_tensor_new, feature_scaler, target_scaler)
    print (f'new angles evaluated, starting to plot, time: {(time.time() - overall_start_time):.2f} seconds')
    
    print (f'starting to plot, time: {(time.time() - overall_start_time):.2f} seconds')
    plot_folder_new_angle = os.path.join(output_folder, f'plots_output_for_new_angles_{today}_{checkpoint["epoch"]}')
    plot_data_2d(df,wind_angles,geometry_filename,plot_folder_new_angle,single=True)


    # from newplottingdiv import plot_div_angles_2d


    # ###NEW ANGLES###
    # print (f'starting to evaluate new angles, time: {(time.time() - overall_start_time):.2f} seconds')
    # # Evaluate the model
    # checkpoint = torch.load(model_file_path, map_location=device)
    # print (f'Model Evaluated at Epoch = {checkpoint["epoch"]} and Training Completed = {checkpoint["training_completed"]}')
    # model.load_state_dict(checkpoint['model_state_dict'])

    # wind_angles = config["train_test"]["new_angles"]
    # for wind_angle in wind_angles:
    #     normalized_X_test_tensor = load_data_new_angle(filenames, base_directory, datafolder_path, device, config, feature_scaler, target_scaler, wind_angle)
    #     df = evaluate_model_new_angles(config, wind_angle, model, normalized_X_test_tensor, feature_scaler, target_scaler)
    #     print (f'model evaluated for angle = {wind_angle}, time: {(time.time() - overall_start_time):.2f} seconds')
    #     print (f'starting to plot for angle = {wind_angle}, time: {(time.time() - overall_start_time):.2f} seconds')
    #     plot_folder_new_angle = os.path.join(output_folder, f'plots_output_for_new_angles_withdiv_{today}_{checkpoint["epoch"]}')
    #     os.makedirs(plot_folder_new_angle, exist_ok=True)
    #     plot_div_angles_2d(df,wind_angle,config,plot_folder_new_angle)
    # ###NEW ANGLES###

    # ##OTHERS###
    # df, current_time, total_epochs = filter_info_file(directory, numtoplot=None)
    # make_all_logging_plots(log_folder, df, current_time, total_epochs)
    # ##OTHERS###

    if output_zip_file is not None:
        shutil.make_archive(output_zip_file[:-4], 'zip', output_folder)

    overall_end_time = time.time()
    overall_elapsed_time = overall_end_time - overall_start_time
    print(f'Overall process completed in {overall_elapsed_time:.2f} seconds')