from definitions import *
from plotting import *

def base_testing(config, model, wind_angles, X_test_tensor, feature_scaler, target_scaler, plot_folder, overall_start_time, testing_type, y_test_tensor=None, data_folder=None):
    chosen_machine_key = config["chosen_machine"]
    datafolder_path = os.path.join(config["machine"][chosen_machine_key], config["data"]["data_folder_name"])
    geometry_filename = os.path.join(datafolder_path, config["data"]["geometry"])
    if y_test_tensor is not None:
        df = evaluate_model(config, model, wind_angles, X_test_tensor, feature_scaler, target_scaler, y_test_tensor, data_folder)
        print (f'Model Evaluated and Starting to Plot, Time: {(time.time() - overall_start_time):.2f} Seconds')
        plot_data_2d(df,wind_angles,geometry_filename,plot_folder,single=False)
        print (f'Plotting Done, Time: {(time.time() - overall_start_time):.2f} Seconds')
    else:
        df = evaluate_model(config, model, wind_angles, X_test_tensor, feature_scaler, target_scaler)
        print (f'Model Evaluated and Starting to Plot, Time: {(time.time() - overall_start_time):.2f} Seconds')
        plot_data_2d(df,wind_angles,geometry_filename,plot_folder,single=True)
        print (f'Plotting Done, Time: {(time.time() - overall_start_time):.2f} Seconds')

def testing(model_file_path, config, output_folder, today, overall_start_time):
    testing_type = "Testing"
    training_wind_angles = config["training"]["angles_to_train"]
    with open(f'{model_file_path}.json', 'r') as f:
        additional_state = json.load(f)
    epoch = additional_state['epoch']
    testing_data_folder = os.path.join(output_folder, f'data_output_{today}_{epoch}')
    testing_plot_folder = os.path.join(output_folder, f'plots_output_{today}_{epoch}')
    model = tf.keras.models.load_model(f'{model_file_path}_tf')
    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, labels_train_tensor, X_train_tensor_skipped, y_train_tensor_skipped, X_test_tensor_skipped, y_test_tensor_skipped, feature_scaler, target_scaler = load_data(config)
    base_testing(config, model, training_wind_angles, X_test_tensor, feature_scaler, target_scaler, testing_plot_folder, overall_start_time, testing_type, y_test_tensor, testing_data_folder)