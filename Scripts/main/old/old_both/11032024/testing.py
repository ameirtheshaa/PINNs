from definitions import *
from plotting import *
from testing_definitions import * 

def base_testing(config, device, base_output_folder, model_file_path, model, wind_angles, X_test_tensor, feature_scaler, target_scaler, plot_folder, overall_start_time, testing_type, y_test_tensor=None, data_folder=None):
    chosen_machine_key = config["chosen_machine"]
    datafolder_path = os.path.join(config["machine"][chosen_machine_key], config["data"]["data_folder_name"])
    geometry_filename = os.path.join(datafolder_path, config["data"]["geometry"])
    checkpoint = open_model_file(model_file_path, device)
    print (f'Model {testing_type} at Epoch = {checkpoint["epoch"]} and Training Completed = {checkpoint["training_completed"]}, Time: {(time.time() - overall_start_time):.2f} Seconds')
    model.load_state_dict(checkpoint['model_state_dict'])
    vtk_output_folder = os.path.join(base_output_folder, f'vtk_output_{checkpoint["epoch"]}')
    if y_test_tensor is not None:
        df = evaluate_model(config, model, wind_angles, X_test_tensor, feature_scaler, target_scaler, y_test_tensor, data_folder, testing_type, vtk_output_folder)
        print (f'Model Evaluated and Starting to Plot, Time: {(time.time() - overall_start_time):.2f} Seconds')
        if config["plotting"]["make_plots"]:
            plot_data_2d(config,df,wind_angles,geometry_filename,plot_folder,single=False)
            plot_diff_2d(config,df,wind_angles,geometry_filename,plot_folder)
        print (f'Plotting Done, Time: {(time.time() - overall_start_time):.2f} Seconds')
    else:
        df = evaluate_model(config, model, wind_angles, X_test_tensor, feature_scaler, target_scaler, None, data_folder, testing_type, vtk_output_folder)
        print (f'Model Evaluated and Starting to Plot, Time: {(time.time() - overall_start_time):.2f} Seconds')
        if config["plotting"]["make_plots"]:
            plot_data_2d(config,df,wind_angles,geometry_filename,plot_folder,single=True)
        print (f'Plotting Done, Time: {(time.time() - overall_start_time):.2f} Seconds')

def testing(model, device, config, data_dict, model_file_path, output_folder, today, overall_start_time):
    testing_type = "Testing"
    checkpoint = open_model_file(model_file_path, device)
    training_wind_angles = config["training"]["angles_to_train"]
    testing_data_folder = os.path.join(output_folder, f'data_output_{today}_{checkpoint["epoch"]}')
    testing_plot_folder = os.path.join(output_folder, f'plots_output_{today}_{checkpoint["epoch"]}')
    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, labels_train_tensor, X_train_tensor_skipped, y_train_tensor_skipped, X_test_tensor_skipped, y_test_tensor_skipped, feature_scaler, target_scaler, X_train_divergences_tensor, X_test_divergences_tensor, y_train_divergences_tensor, y_test_divergences_tensor = data_dict.values()
    for training_wind_angle in training_wind_angles:
        X_test_tensor_all, y_test_tensor_all = load_data_new_angles(device, config, feature_scaler, target_scaler, [training_wind_angle])
        base_testing(config, device, output_folder, model_file_path, model, [training_wind_angle], X_test_tensor_all, feature_scaler, target_scaler, testing_plot_folder, overall_start_time, testing_type, y_test_tensor_all, testing_data_folder)

def solid_boundary_testing(model, device, config, data_dict, model_file_path, output_folder, today, overall_start_time):
    testing_type = "Boundary_Testing"
    checkpoint = open_model_file(model_file_path, device)
    all_wind_angles = config["training"]["all_angles"]
    testing_plot_folder = os.path.join(output_folder, f'solid_boundary_plots_output_{today}_{checkpoint["epoch"]}')
    feature_scaler = data_dict["feature_scaler"]
    target_scaler = data_dict["target_scaler"]
    X_test_tensor = load_boundary_testing_data(config, device, feature_scaler, 'geometry_test_points')
    chosen_machine_key = config["chosen_machine"]
    datafolder_path = os.path.join(config["machine"][chosen_machine_key], config["data"]["data_folder_name"])
    geometry_filename = os.path.join(datafolder_path, config["data"]["geometry"])
    checkpoint = open_model_file(model_file_path, device)
    print (f'Model {testing_type} at Epoch = {checkpoint["epoch"]} and Training Completed = {checkpoint["training_completed"]}, Time: {(time.time() - overall_start_time):.2f} Seconds')
    model.load_state_dict(checkpoint['model_state_dict'])
    df = evaluate_model_boundary(config, model, all_wind_angles, X_test_tensor, feature_scaler, target_scaler)
    print (f'Model Evaluated and Starting to Plot, Time: {(time.time() - overall_start_time):.2f} Seconds')
    plot_boundary_2d(config,df,all_wind_angles,geometry_filename,testing_plot_folder)
    print (f'Plotting Done, Time: {(time.time() - overall_start_time):.2f} Seconds')
    evaluation_boundary_physics(model, device, config, data_dict, model_file_path, testing_plot_folder, today, overall_start_time, 'Div')
    evaluation_boundary_physics(model, device, config, data_dict, model_file_path, testing_plot_folder, today, overall_start_time, 'RANS')

def surface_boundary_testing(model, device, config, data_dict, model_file_path, output_folder, today, overall_start_time):
    testing_type = "Surface_Boundary_Testing"
    checkpoint = open_model_file(model_file_path, device)
    all_wind_angles = config["training"]["all_angles"]
    testing_plot_folder = os.path.join(output_folder, f'surface_boundary_plots_output_{today}_{checkpoint["epoch"]}')
    feature_scaler = data_dict["feature_scaler"]
    target_scaler = data_dict["target_scaler"]
    X_test_tensor = load_boundary_testing_data(config, device, feature_scaler, 'surface_geometry_test_points')
    chosen_machine_key = config["chosen_machine"]
    datafolder_path = os.path.join(config["machine"][chosen_machine_key], config["data"]["data_folder_name"])
    geometry_filename = os.path.join(datafolder_path, config["data"]["geometry"])
    checkpoint = open_model_file(model_file_path, device)
    print (f'Model {testing_type} at Epoch = {checkpoint["epoch"]} and Training Completed = {checkpoint["training_completed"]}, Time: {(time.time() - overall_start_time):.2f} Seconds')
    model.load_state_dict(checkpoint['model_state_dict'])
    df = evaluate_model_boundary(config, model, all_wind_angles, X_test_tensor, feature_scaler, target_scaler)
    print (f'Model Evaluated and Starting to Plot, Time: {(time.time() - overall_start_time):.2f} Seconds')
    plot_boundary_2d(config,df,all_wind_angles,geometry_filename,testing_plot_folder)
    print (f'Plotting Done, Time: {(time.time() - overall_start_time):.2f} Seconds')
    evaluation_boundary_physics(model, device, config, data_dict, model_file_path, testing_plot_folder, today, overall_start_time, 'Div')
    evaluation_boundary_physics(model, device, config, data_dict, model_file_path, testing_plot_folder, today, overall_start_time, 'RANS')

def evaluation(model, device, config, data_dict, model_file_path, output_folder, today, overall_start_time):
    testing_type = "Evaluation_Skipped_Angle"
    checkpoint = open_model_file(model_file_path, device)
    skipped_wind_angles = config["training"]["angles_to_leave_out"]
    skipped_data_folder = os.path.join(output_folder, f'data_output_for_skipped_angle_{today}_{checkpoint["epoch"]}')
    skipped_plot_folder = os.path.join(output_folder, f'plots_output_for_skipped_angle(s)_{today}_{checkpoint["epoch"]}')
    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, labels_train_tensor, X_train_tensor_skipped, y_train_tensor_skipped, X_test_tensor_skipped, y_test_tensor_skipped, feature_scaler, target_scaler, X_train_divergences_tensor, X_test_divergences_tensor, y_train_divergences_tensor, y_test_divergences_tensor = data_dict.values()
    X_test_tensor_all, y_test_tensor_all = load_data_new_angles(device, config, feature_scaler, target_scaler, skipped_wind_angles)
    base_testing(config, device, output_folder, model_file_path, model, skipped_wind_angles, X_test_tensor_all, feature_scaler, target_scaler, skipped_plot_folder, overall_start_time, testing_type, y_test_tensor_all, skipped_data_folder)

def evaluation_new_angles(model, device, config, data_dict, model_file_path, output_folder, today, overall_start_time):
    testing_type = "Evaluation_New_Angles"
    checkpoint = open_model_file(model_file_path, device)
    new_wind_angles = config["train_test"]["new_angles"]
    newangles_data_folder = os.path.join(output_folder, f'data_output_for_new_angles_{today}_{checkpoint["epoch"]}')
    newangles_plot_folder = os.path.join(output_folder, f'plots_output_for_new_angles_{today}_{checkpoint["epoch"]}')
    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, labels_train_tensor, X_train_tensor_skipped, y_train_tensor_skipped, X_test_tensor_skipped, y_test_tensor_skipped, feature_scaler, target_scaler, X_train_divergences_tensor, X_test_divergences_tensor, y_train_divergences_tensor, y_test_divergences_tensor = data_dict.values()
    X_test_tensor_new, y_test_tensor_new = load_data_new_angles(device, config, feature_scaler, target_scaler)
    for new_wind_angle in new_wind_angles:
        wind_angle = [new_wind_angle]
        base_testing(config, device, output_folder, model_file_path, model, wind_angle, X_test_tensor_new, feature_scaler, target_scaler, newangles_plot_folder, overall_start_time, testing_type, None, newangles_data_folder)

def process_logging_statistics(log_folder):
    info_path = os.path.join(log_folder, 'info.csv')
    if Path(info_path).exists():
        df, current_time, total_epochs = filter_info_file(log_folder)
        make_all_logging_plots(log_folder, df, current_time, total_epochs)

def make_pure_data_plots(config, output_folder, today, overall_start_time):
    print (f'starting to plot pure data, time: {(time.time() - overall_start_time):.2f} seconds')
    chosen_machine_key = config["chosen_machine"]
    datafolder_path = os.path.join(config["machine"][chosen_machine_key], config["data"]["data_folder_name"])
    geometry_filename = os.path.join(datafolder_path, config["data"]["geometry"])
    plot_folder = os.path.join(output_folder, f'data_plots_output_{today}')
    df = load_plotting_data(config)
    wind_angles = config["training"]["all_angles"]
    plot_data_2d(config,df,wind_angles,geometry_filename,plot_folder,single=True)
    print (f'plot pure data done, time: {(time.time() - overall_start_time):.2f} seconds')