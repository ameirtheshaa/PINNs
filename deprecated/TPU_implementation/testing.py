from definitions import *
from plotting import *
import torch
import torch.nn as nn 
from stl import mesh

class PINN_torch(nn.Module):
    def __init__(self, input_params, output_params, hidden_layers, neurons_per_layer, activation, use_batch_norm, dropout_rate):
        super(PINN_torch, self).__init__()
        input_size = len(input_params)
        output_size = len(output_params)
        layers = []
        layers.append(nn.Linear(input_size, neurons_per_layer[0]))
        for i in range(hidden_layers - 1):
            layers.append(activation())
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(neurons_per_layer[i]))
            if dropout_rate:
                layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.Linear(neurons_per_layer[i], neurons_per_layer[i + 1]))
        layers.append(activation())
        layers.append(nn.Linear(neurons_per_layer[-1], output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def compute_data_loss(self, X, y):
        criterion = nn.MSELoss()
        predictions = self(X)
        loss = criterion(predictions, y)
        return loss

def open_model_file(model_file_path, device):
    checkpoint = torch.load(model_file_path, map_location=device)
    return checkpoint

def load_data(config, device):
    chosen_machine_key = config["chosen_machine"]
    datafolder_path = os.path.join(config["machine"][chosen_machine_key], config["data"]["data_folder_name"])
    filenames = get_filenames_from_folder(datafolder_path, config["data"]["extension"], config["data"]["startname_data"])
    training_wind_angles = config["training"]["angles_to_train"]
    skipped_wind_angles = config["training"]["angles_to_leave_out"]
    angle_to_label = {angle: idx for idx, angle in enumerate(sorted(training_wind_angles))}
        
    data, labels = concatenate_data_files(filenames, datafolder_path, training_wind_angles, angle_to_label)
    features = data[config["training"]["input_params"]]
    targets = data[config["training"]["output_params"]]
    feature_scaler, target_scaler = initialize_and_fit_scalers(features, targets, config)
    normalized_features, normalized_targets = transform_data_with_scalers(features, targets, feature_scaler, target_scaler)
    X_train, X_test, y_train, y_test, labels_train, labels_test = train_test_split(normalized_features, normalized_targets, labels, test_size=config["train_test"]["test_size"], random_state=config["train_test"]["random_state"])
    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = convert_to_tensor(X_train, X_test, y_train, y_test, device=device)
    labels_train_tensor = torch.tensor(np.array(labels_train), dtype=torch.long)
    
    data_skipped = concatenate_data_files(filenames, datafolder_path, skipped_wind_angles)
    features_skipped = data_skipped[config["training"]["input_params"]]
    targets_skipped = data_skipped[config["training"]["output_params"]]
    normalized_features_skipped, normalized_targets_skipped = transform_data_with_scalers(features_skipped, targets_skipped, feature_scaler, target_scaler)
    X_train_skipped, X_test_skipped, y_train_skipped, y_test_skipped = train_test_split(normalized_features_skipped, normalized_targets_skipped,test_size=len(data_skipped)-1, random_state=config["train_test"]["random_state"])    
    X_train_tensor_skipped, y_train_tensor_skipped, X_test_tensor_skipped, y_test_tensor_skipped = convert_to_tensor(X_train_skipped, X_test_skipped, y_train_skipped, y_test_skipped, device=device)
    
    data_dict = {
        "X_train_tensor": X_train_tensor,
        "y_train_tensor": y_train_tensor,
        "X_test_tensor": X_test_tensor,
        "y_test_tensor": y_test_tensor,
        "labels_train_tensor": labels_train_tensor,
        "X_train_tensor_skipped": X_train_tensor_skipped,
        "y_train_tensor_skipped": y_train_tensor_skipped,
        "X_test_tensor_skipped": X_test_tensor_skipped,
        "y_test_tensor_skipped": y_test_tensor_skipped,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler
    }

    return data_dict

def get_available_device_memory(device):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        available_gpu_memory = (info.free/(1024**3))
        total_gpu_memory = torch.cuda.get_device_properties(device).total_memory/(1024**3)
        free_memory = available_gpu_memory
        total_memory = total_gpu_memory
        pynvml.nvmlShutdown()
    else:
        cpu_memory = psutil.virtual_memory()
        total_cpu_memory = cpu_memory.total/(1024**3)
        available_cpu_memory = cpu_memory.available/(1024**3)
        free_memory = available_cpu_memory
        total_memory = total_cpu_memory
    return free_memory, total_memory

def print_and_set_available_gpus():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"{num_gpus} GPUs available:")
        free_memories = []
        for i in range(num_gpus):
            free_memory, _ = get_available_device_memory(i)
            free_memories.append(free_memory)
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}, Free Memory: {free_memory:.2f} GB")
        selected_gpu = free_memories.index(max(free_memories))
        device = torch.device(f'cuda:{selected_gpu}')
        print(f"Using device: {device}, with {free_memories[selected_gpu]:.2f} GB free memory")
    else:
        print("No GPUs available.")
        device = torch.device('cpu')
        print(f"Using device: {device}")
    return device

def transform_data_with_scalers(features, targets, feature_scaler, target_scaler):
    normalized_features = feature_scaler.transform(features)
    normalized_targets = target_scaler.transform(targets)
    return normalized_features, normalized_targets

def inverse_transform_features(features_normalized, feature_scaler):
    return feature_scaler.inverse_transform(features_normalized)

def inverse_transform_targets(targets_normalized, target_scaler):
    return target_scaler.inverse_transform(targets_normalized)

def concatenate_data_files(filenames, datafolder_path, wind_angles=None, angle_labels=None):
    dfs = []
    labels = []
    for filename in sorted(filenames):
        df = pd.read_csv(os.path.join(datafolder_path, filename))
        wind_angle = int(filename.split('_')[-1].split('.')[0])
        df['cos(WindAngle)'] = (np.cos(np.deg2rad(wind_angle)))
        df['sin(WindAngle)'] = (np.sin(np.deg2rad(wind_angle)))
        if wind_angles is None:
            df['WindAngle'] = (wind_angle)
            dfs.append(df)
        else:
            if wind_angle in wind_angles:
                df['WindAngle'] = (wind_angle)
                dfs.append(df)
                if angle_labels is not None:
                    label = angle_labels[wind_angle]
                    labels.extend([label] * len(df))
                    df['WindAngle'] = (wind_angle)
    data = pd.concat(dfs)
    if angle_labels is not None:
        labels = np.array(labels)
        return data, labels
    else:
        return data 

def convert_to_tensor(X_train, X_test, y_train, y_test, device=None):
    X_train_tensor = torch.tensor(np.array(X_train), dtype=torch.float32)
    y_train_tensor = torch.tensor(np.array(y_train), dtype=torch.float32)
    X_test_tensor = torch.tensor(np.array(X_test), dtype=torch.float32)
    y_test_tensor = torch.tensor(np.array(y_test), dtype=torch.float32)
    if device is not None:
        X_train_tensor = X_train_tensor.to(device)
        y_train_tensor = y_train_tensor.to(device)
        X_test_tensor = X_test_tensor.to(device)
        y_test_tensor = y_test_tensor.to(device)
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor

def load_json(model_file_path):
    with open(f'{model_file_path}.json', 'r') as f:
        additional_state = json.load(f)
    epoch = additional_state['epoch']
    # current_loss = additional_state['current_loss']
    training_completed = additional_state['training_completed']
    return epoch, training_completed

def evaluate_model(config, model, wind_angles, X_test_tensor, feature_scaler, target_scaler, y_test_tensor = None, output_folder = None):
    model.eval()
    with torch.no_grad():
        dfs = []
        predictions_tensor = model(X_test_tensor)
        X_test_tensor_cpu = X_test_tensor.cpu()
        X_test_tensor_cpu = inverse_transform_features(X_test_tensor_cpu, feature_scaler)
        X_test_column_names = config["training"]["input_params_modf"]
        X_test_dataframe = pd.DataFrame(X_test_tensor_cpu, columns=X_test_column_names)
        if y_test_tensor is None:
            predictions_tensor_cpu = predictions_tensor.cpu()
            predictions_tensor_cpu = inverse_transform_targets(predictions_tensor_cpu, target_scaler)
            predictions_column_names = config["training"]["output_params_modf"]
            predictions_dataframe = pd.DataFrame(predictions_tensor_cpu, columns=predictions_column_names)
            predictions_dataframe['Velocity_Magnitude'] = np.sqrt(predictions_dataframe['Velocity_X']**2 + predictions_dataframe['Velocity_Y']**2 + predictions_dataframe['Velocity_Z']**2)
            for wind_angle in wind_angles:
                lower_bound = wind_angle - 2
                upper_bound = wind_angle + 2
                X_test_dataframe['WindAngle_rad'] = np.arctan2(X_test_dataframe['sin(WindAngle)'], X_test_dataframe['cos(WindAngle)'])
                X_test_dataframe['WindAngle'] = X_test_dataframe['WindAngle_rad'].apply(lambda x: int(np.ceil(np.degrees(x))))
                mask = X_test_dataframe['WindAngle'].between(lower_bound, upper_bound)
                filtered_X_test_dataframe = X_test_dataframe.loc[mask]
                filtered_predictions = predictions_dataframe.loc[mask]
                if len(filtered_predictions)!= 0:
                    combined_df = pd.concat([filtered_X_test_dataframe, filtered_predictions], axis=1)
                    combined_df['WindAngle'] = (wind_angle)
                    dfs.append(combined_df)
        else:
            os.makedirs(output_folder, exist_ok=True)
            y_test_tensor_cpu = y_test_tensor.cpu()
            y_test_tensor_cpu = inverse_transform_targets(y_test_tensor_cpu, target_scaler)
            output_column_names = config["training"]["output_params_modf"]
            y_test_column_names = [item + "_Actual" for item in output_column_names]
            y_test_dataframe = pd.DataFrame(y_test_tensor_cpu, columns=y_test_column_names)
            y_test_dataframe['Velocity_Magnitude_Actual'] = np.sqrt(y_test_dataframe['Velocity_X_Actual']**2 + y_test_dataframe['Velocity_Y_Actual']**2 + y_test_dataframe['Velocity_Z_Actual']**2)
            predictions_tensor_cpu = predictions_tensor.cpu()
            predictions_tensor_cpu = inverse_transform_targets(predictions_tensor_cpu, target_scaler)
            predictions_column_names = [item + "_Predicted" for item in output_column_names]
            predictions_dataframe = pd.DataFrame(predictions_tensor_cpu, columns=predictions_column_names)
            predictions_dataframe['Velocity_Magnitude_Predicted'] = np.sqrt(predictions_dataframe['Velocity_X_Predicted']**2 + predictions_dataframe['Velocity_Y_Predicted']**2 + predictions_dataframe['Velocity_Z_Predicted']**2)
            for wind_angle in wind_angles:
                lower_bound = wind_angle - 2
                upper_bound = wind_angle + 2
                X_test_dataframe['WindAngle_rad'] = np.arctan2(X_test_dataframe['sin(WindAngle)'], X_test_dataframe['cos(WindAngle)'])
                X_test_dataframe['WindAngle'] = X_test_dataframe['WindAngle_rad'].apply(lambda x: int(np.ceil(np.degrees(x))))
                mask = X_test_dataframe['WindAngle'].between(lower_bound, upper_bound)
                filtered_X_test_dataframe = X_test_dataframe.loc[mask]
                filtered_y_test = y_test_dataframe.loc[mask]
                filtered_predictions = predictions_dataframe.loc[mask]
                if len(filtered_predictions)!= 0 and len(filtered_y_test)!=0:
                    rows_list = []
                    for i, var in enumerate(y_test_column_names):
                        var_cleaned = var.replace('_Actual', '')
                        actuals = filtered_y_test.iloc[:, i]
                        preds = filtered_predictions.iloc[:, i]
                        mse = sklearn.metrics.mean_squared_error(actuals, preds)
                        rmse = np.sqrt(mse)
                        mae = sklearn.metrics.mean_absolute_error(actuals, preds)
                        r2 = sklearn.metrics.r2_score(actuals, preds)
                        rows_list.append({
                            'Variable': var_cleaned, 
                            'MSE': mse,
                            'RMSE': rmse,
                            'MAE': mae,
                            'R2': r2
                        })
                    data_folder = os.path.join(output_folder, f'data_output_for_wind_angle_{wind_angle}')
                    os.makedirs(data_folder, exist_ok=True)
                    combined_df = pd.concat([filtered_X_test_dataframe, filtered_y_test, filtered_predictions], axis=1)
                    combined_file_path = os.path.join(data_folder, f'combined_actuals_and_predictions_for_wind_angle_{wind_angle}.csv')
                    combined_df.to_csv(combined_file_path, index=False)
                    metrics_df = pd.DataFrame(rows_list)   
                    metrics_file_path = os.path.join(data_folder, f'metrics_for_wind_angle_{wind_angle}.csv')
                    metrics_df.to_csv(metrics_file_path, index=False)
                    dfs.append(combined_df)
        data = pd.concat(dfs)
        return data

def base_testing(config, device, model_file_path, model_file_path_modf, model, wind_angles, X_test_tensor, feature_scaler, target_scaler, plot_folder, overall_start_time, testing_type, y_test_tensor=None, data_folder=None):
    chosen_machine_key = config["chosen_machine"]
    datafolder_path = os.path.join(config["machine"][chosen_machine_key], config["data"]["data_folder_name"])
    geometry_filename = os.path.join(datafolder_path, config["data"]["geometry"])
    checkpoint = open_model_file(model_file_path_modf, device)
    epoch, training_completed = load_json(model_file_path)
    print (f'Model {testing_type} at Epoch = {epoch} and Training Completed = {training_completed}, Time: {(time.time() - overall_start_time):.2f} Seconds')
    model.eval()
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

def testing(config, model_file_path, model_file_path_modf, output_folder, today, overall_start_time):
    testing_type = "Testing"
    device = print_and_set_available_gpus()
    data_dict = load_data(config, device)
    model = PINN_torch(input_params=config["training"]["input_params"], output_params=config["training"]["output_params"], hidden_layers=config["training"]["number_of_hidden_layers"], neurons_per_layer=[config["training"]["neuron_number"]] * config["training"]["number_of_hidden_layers"], activation=nn.ELU, use_batch_norm=config["training"]["batch_normalization"], dropout_rate=config["training"]["dropout_rate"]).to(device)  
    checkpoint = open_model_file(model_file_path_modf, device)
    epoch, training_completed = load_json(model_file_path)
    training_wind_angles = config["training"]["angles_to_train"]
    testing_data_folder = os.path.join(output_folder, f'data_output_{today}_{epoch}')
    testing_plot_folder = os.path.join(output_folder, f'plots_output_{today}_{epoch}')
    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, labels_train_tensor, X_train_tensor_skipped, y_train_tensor_skipped, X_test_tensor_skipped, y_test_tensor_skipped, feature_scaler, target_scaler = data_dict.values()
    base_testing(config, device, model_file_path, model_file_path_modf, model, training_wind_angles, X_test_tensor, feature_scaler, target_scaler, testing_plot_folder, overall_start_time, testing_type, y_test_tensor, testing_data_folder)

def evaluation(model, device, config, data_dict, model_file_path, output_folder, today, overall_start_time):
    testing_type = "Evaluation Skipped Angle"
    checkpoint = open_model_file(model_file_path, device)
    skipped_wind_angles = config["training"]["angles_to_leave_out"]
    skipped_data_folder = os.path.join(output_folder, f'data_output_for_skipped_angle_{today}_{epoch}')
    skipped_plot_folder = os.path.join(output_folder, f'plots_output_for_skipped_angle(s)_{today}_{epoch}')
    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, labels_train_tensor, X_train_tensor_skipped, y_train_tensor_skipped, X_test_tensor_skipped, y_test_tensor_skipped, feature_scaler, target_scaler = data_dict.values()
    base_testing(config, device, model_file_path, model, skipped_wind_angles, X_test_tensor_skipped, feature_scaler, target_scaler, skipped_plot_folder, overall_start_time, testing_type, y_test_tensor_skipped, skipped_data_folder)

def evaluation_new_angles(model, device, config, data_dict, model_file_path, output_folder, today, overall_start_time):
    testing_type = "Evaluation New Angles"
    checkpoint = open_model_file(model_file_path, device)
    new_wind_angles = config["train_test"]["new_angles"]
    newangles_plot_folder = os.path.join(output_folder, f'plots_output_for_new_angles_{today}_{epoch}')
    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, labels_train_tensor, X_train_tensor_skipped, y_train_tensor_skipped, X_test_tensor_skipped, y_test_tensor_skipped, feature_scaler, target_scaler = data_dict.values()
    X_test_tensor_new = load_data_new_angles(device, config, feature_scaler, target_scaler)
    for new_wind_angle in new_wind_angles:
        wind_angle = [new_wind_angle]
        base_testing(config, device, model_file_path, model, wind_angle, X_test_tensor_new, feature_scaler, target_scaler, newangles_plot_folder, overall_start_time, testing_type)

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
    plot_data_2d(df,wind_angles,geometry_filename,plot_folder,single=True)
    print (f'plot pure data done, time: {(time.time() - overall_start_time):.2f} seconds')