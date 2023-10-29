import os
import numpy as np
import subprocess
import sys
import time
import datetime
import psutil
import argparse
import shutil
import pynvml
import itertools
import GPUtil
import socket
import numpy as np
import pandas as pd
from pathlib import Path
import importlib.util
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from scipy.interpolate import griddata
import torch.multiprocessing as mp

class Logger(object):
    def __init__(self, filename='Default.log'):
        self.terminal = sys.stdout  # Save the original stdout
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        pass

def parse_arguments():
    parser = argparse.ArgumentParser(description='RANSPINNS Main Script')
    parser.add_argument('--base_directory', type=str, required=True,
                        help='The base directory where the config.py file is located')
    args = parser.parse_args()
    return args

def get_available_device_memory(device):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        pynvml.nvmlInit()  # Initialize the NVML library
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        available_gpu_memory = (info.free / (1024 ** 3))  # Free memory in GB
        total_gpu_memory = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        free_memory = available_gpu_memory
        total_memory = total_gpu_memory
        pynvml.nvmlShutdown()  # Close the NVML library
    else:
        cpu_memory = psutil.virtual_memory()
        total_cpu_memory = cpu_memory.total / (1024 ** 3)
        available_cpu_memory = cpu_memory.available / (1024 ** 3)
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

def estimate_memory(model, input_tensor, batch_size):
    # Memory occupied by model parameters and gradients
    param_memory = sum(p.numel() * 4 for p in model.parameters())  # 4 bytes per parameter (float32)
    grad_memory = param_memory  # gradients occupy the same amount of memory as parameters
    
    # Memory occupied by the input tensor and other intermediate tensors
    # Assuming memory occupied by other tensors is approximately equal to the input tensor memory
    input_memory = input_tensor.numel() * 4 * batch_size  # 4 bytes per element (float32)
    intermediate_memory = input_memory  # Rough approximation
    
    # Total Memory
    total_memory = param_memory + grad_memory + input_memory + intermediate_memory
    total_memory = total_memory / (1024 ** 3)

    return total_memory  # in GB

def select_device_and_batch_size(model, input_tensor , device):
    single_sample_memory = estimate_memory(model, input_tensor, batch_size=1)
    if device != 'cpu':
        free_memory, _ = get_available_device_memory(device.index)
        max_batch_size_gpu = free_memory // single_sample_memory
        optimal_batch_size = max_batch_size_gpu
        print(f"{device} with free memory: {free_memory:.2f} GB and with optimal batch size: {max_batch_size_gpu}")
    else:
        free_memory, _ = get_available_device_memory(device)
        max_batch_size_cpu = free_memory // single_sample_memory
        optimal_batch_size = max_batch_size_cpu
        print(f"{device} with free memory: {free_memory:.2f} GB and with optimal batch size: {max_batch_size_cpu}")
    optimal_batch_size = int(optimal_batch_size)
    print(f"Now using device: {device} with optimal batch size: {optimal_batch_size}")
    return device, optimal_batch_size

def get_filenames_from_folder(path, extension, startname):
    """Get a list of filenames from the specified folder that end with a given extension."""
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith(extension) and f.startswith(startname)]

def get_optimizer(model, optimizer_config):
    optimizer_type = optimizer_config["type"]
    if optimizer_type == "both_optimizers":
        optimizer_lbfgs = torch.optim.LBFGS(model.parameters(), lr=optimizer_config["learning_rate_lbfgs"], 
                                    max_iter=optimizer_config["max_iter_lbfgs"], 
                                    max_eval=optimizer_config["max_eval_lbfgs"], 
                                    tolerance_grad=optimizer_config["tolerance_grad_lbfgs"], 
                                    tolerance_change=optimizer_config["tolerance_change_lbfgs"], 
                                    history_size=optimizer_config["history_size_lbfgs"], 
                                    line_search_fn=optimizer_config["line_search_fn_lbfgs"])
        optimizer_adam = torch.optim.Adam(model.parameters(), lr=optimizer_config["learning_rate_adam"])
        return optimizer_adam, optimizer_lbfgs
    else:
        if optimizer_type == "LBFGS":
            optimizer = torch.optim.LBFGS(model.parameters(), lr=optimizer_config["learning_rate"], 
                                    max_iter=optimizer_config["max_iter"], 
                                    max_eval=optimizer_config["max_eval"], 
                                    tolerance_grad=optimizer_config["tolerance_grad"], 
                                    tolerance_change=optimizer_config["tolerance_change"], 
                                    history_size=optimizer_config["history_size"], 
                                    line_search_fn=optimizer_config["line_search_fn"])
        elif optimizer_type == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_config["learning_rate"])
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        return optimizer

def initialize_and_fit_scalers(features, targets, config):

    if config["training"]["change_scaler"]:
        if config["training"]["scaler"] == "min_max":
            feature_scaler = MinMaxScaler(feature_range=config["training"]["min_max_scaler_range"])
            target_scaler = MinMaxScaler(feature_range=config["training"]["min_max_scaler_range"])

            # Fit the scalers
            feature_scaler.fit(features)
            target_scaler.fit(targets)
        else:
            print ("scaler undefined")
    else:
        # Initialize Standard Scalers for features and targets
        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()

        # Fit the scalers
        feature_scaler.fit(features)
        target_scaler.fit(targets)
    
    return feature_scaler, target_scaler

def transform_data_with_scalers(features, targets, feature_scaler, target_scaler):
    # Transform the features and targets using the provided scalers
    normalized_features = feature_scaler.transform(features)
    normalized_targets = target_scaler.transform(targets)
    
    return normalized_features, normalized_targets

def inverse_transform_features(features_normalized, feature_scaler):
    """Inverse transform the features using the provided scaler."""
    features_original = feature_scaler.inverse_transform(features_normalized)
    return features_original

def inverse_transform_targets(targets_normalized, target_scaler):
    """Inverse transform the targets using the provided scaler."""
    targets_original = target_scaler.inverse_transform(targets_normalized)
    return targets_original

def load_data(filenames, base_directory, datafolder_path, device, config):

    rho = config["data"]["density"]
    nu = config["data"]["kinematic_viscosity"]
    angle_to_leave_out = config["training"]["angle_to_leave_out"]

    dfs = []
    dfs_skipped = []

    if angle_to_leave_out is None:
        for filename in sorted(filenames):
            df = pd.read_csv(os.path.join(datafolder_path, filename))

            index_str = filename.split('_')[-1].split('.')[0]  # Extract the index part of the filename
            index = int(index_str)  # Convert the index to integer

            meteo_data = pd.read_csv(os.path.join(datafolder_path,'meteo.csv'))

            # Look up the corresponding row in the meteo.csv file
            meteo_row = meteo_data[meteo_data['index'] == index]

            # Extract the wind angle from the found row
            wind_angle = meteo_row['cs degree'].values[0]  
            
            # Add new columns with unique values for each file
            df['cos(WindAngle)'] = (np.cos(np.deg2rad(wind_angle)))
            df['sin(WindAngle)'] = (np.sin(np.deg2rad(wind_angle)))
            
            # Append the modified DataFrame to the list
            dfs.append(df)
    else:
        for filename in sorted(filenames):
            df = pd.read_csv(os.path.join(datafolder_path, filename))

            index_str = filename.split('_')[-1].split('.')[0]  # Extract the index part of the filename
            index = int(index_str)  # Convert the index to integer

            meteo_data = pd.read_csv(os.path.join(datafolder_path,'meteo.csv'))

            # Look up the corresponding row in the meteo.csv file
            meteo_row = meteo_data[meteo_data['index'] == index]

            # Extract the wind angle from the found row
            wind_angle = meteo_row['cs degree'].values[0]

            if wind_angle in angle_to_leave_out:
                print (f'Skipping Angle = {wind_angle} degrees')
            else:
                # Add new columns with unique values for each file
                df['cos(WindAngle)'] = (np.cos(np.deg2rad(wind_angle)))
                df['sin(WindAngle)'] = (np.sin(np.deg2rad(wind_angle)))
                
                # Append the modified DataFrame to the list
                dfs.append(df)

    # Concatenate the list of DataFrames
    data = pd.concat(dfs)

    # Extract features from the dataframe
    features = data[['Points:0', 'Points:1', 'Points:2', 'cos(WindAngle)', 'sin(WindAngle)']]
    targets = data[['Pressure', 'Velocity:0', 'Velocity:1', 'Velocity:2', 'TurbVisc']]

    feature_scaler, target_scaler = initialize_and_fit_scalers(features, targets, config)
    normalized_features, normalized_targets = transform_data_with_scalers(features, targets, feature_scaler, target_scaler)

    # Perform the train-test split and get the indices
    X_train, X_test, y_train, y_test = train_test_split(normalized_features, normalized_targets,test_size=config["train_test"]["test_size"], random_state=config["train_test"]["random_state"])

    # Convert to PyTorch Tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    X_train_tensor = X_train_tensor.to(device)
    y_train_tensor = y_train_tensor.to(device)
    X_test_tensor = X_test_tensor.to(device)
    y_test_tensor = y_test_tensor.to(device)

    for filename in sorted(filenames):
        df = pd.read_csv(os.path.join(datafolder_path, filename))

        index_str = filename.split('_')[-1].split('.')[0]  # Extract the index part of the filename
        index = int(index_str)  # Convert the index to integer

        meteo_data = pd.read_csv(os.path.join(datafolder_path,'meteo.csv'))

        # Look up the corresponding row in the meteo.csv file
        meteo_row = meteo_data[meteo_data['index'] == index]

        # Extract the wind angle from the found row
        wind_angle = meteo_row['cs degree'].values[0]

        if wind_angle not in angle_to_leave_out:
            print (f'Angle = {wind_angle} degrees part of dataset!')
        else:
            # Add new columns with unique values for each file
            df['cos(WindAngle)'] = (np.cos(np.deg2rad(wind_angle)))
            df['sin(WindAngle)'] = (np.sin(np.deg2rad(wind_angle)))
        
            # Append the modified DataFrame to the list
            dfs_skipped.append(df)

    data_skipped = pd.concat(dfs_skipped)

    # Extract features from the dataframe
    features_skipped = data_skipped[['Points:0', 'Points:1', 'Points:2', 'cos(WindAngle)', 'sin(WindAngle)']]
    targets_skipped = data_skipped[['Pressure', 'Velocity:0', 'Velocity:1', 'Velocity:2', 'TurbVisc']]

    normalized_features_skipped, normalized_targets_skipped = transform_data_with_scalers(features_skipped, targets_skipped, feature_scaler, target_scaler)

    X_train_skipped, X_test_skipped, y_train_skipped, y_test_skipped = train_test_split(normalized_features_skipped, normalized_targets_skipped,test_size=len(data_skipped)-1, random_state=config["train_test"]["random_state"])

    X_train_tensor_skipped = torch.tensor(X_train_skipped, dtype=torch.float32)
    y_train_tensor_skipped = torch.tensor(y_train_skipped, dtype=torch.float32)
    X_test_tensor_skipped = torch.tensor(X_test_skipped, dtype=torch.float32)
    y_test_tensor_skipped = torch.tensor(y_test_skipped, dtype=torch.float32)

    X_train_tensor_skipped = X_train_tensor_skipped.to(device)
    y_train_tensor_skipped = y_train_tensor_skipped.to(device)
    X_test_tensor_skipped = X_test_tensor_skipped.to(device)
    y_test_tensor_skipped = y_test_tensor_skipped.to(device)

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, X_train_tensor_skipped, y_train_tensor_skipped, X_test_tensor_skipped, y_test_tensor_skipped, feature_scaler, target_scaler

def extract_unique_wind_angles_from_X(X):
    # Extract unique pairs of sin and cos values
    unique_pairs, _ = torch.unique(X[:, 3:5], dim=0, return_inverse=True)
    cos_theta = unique_pairs[:, 0]
    sin_theta = unique_pairs[:, 1]

    # Compute unique wind angles in radians
    wind_angle_rad = torch.atan2(sin_theta, cos_theta)
    wind_angle_deg = torch.rad2deg(wind_angle_rad)

    return wind_angle_deg

def extract_wind_angle_from_X_closest_match(X, wind_angles, tolerance=5):
    # Extract the unique wind angles in degrees from X
    unique_wind_angles_deg = extract_unique_wind_angles_from_X(X)
    
    # Find the closest match from the given wind_angles list within the tolerance
    closest_matches = []
    for angle in wind_angles:
        diff = torch.abs(unique_wind_angles_deg - angle)
        mask = diff <= tolerance
        closest_matches.append(torch.where(mask, angle, unique_wind_angles_deg))
    
    # Stack the closest matches and take the minimum absolute difference for each element
    stacked_matches = torch.stack(closest_matches, dim=-1)
    min_diffs, _ = torch.min(torch.abs(stacked_matches - unique_wind_angles_deg.unsqueeze(-1)), dim=-1)
    
    # Replace the values in unique_wind_angles_deg with the closest match where the difference is within the tolerance
    unique_wind_angles_deg = torch.where(min_diffs <= tolerance, stacked_matches[torch.arange(stacked_matches.size(0)), min_diffs.argmin(dim=-1)], unique_wind_angles_deg)
    
    return unique_wind_angles_deg.tolist()

def generate_points_from_X(X, n, device):
    """
    Generate a tensor of points based on the min, max of x, y, z in X and the unique wind angles.

    Parameters:
    - X: Input tensor containing x, y, z, cos(wind_angle), sin(wind_angle).
    - n: Number of points to divide the range of x, y, z.

    Returns:
    - points_tensor: Tensor containing the generated points.
    """
    
    # Extract min and max of x, y, z
    x_min, x_max = torch.min(X[:, 0]), torch.max(X[:, 0])
    y_min, y_max = torch.min(X[:, 1]), torch.max(X[:, 1])
    z_min, z_max = torch.min(X[:, 2]), torch.max(X[:, 2])

    # Extract unique wind angles
    wind_angles_deg = extract_unique_wind_angles_from_X(X)

    # Generate points
    x_values = torch.linspace(x_min, x_max, n)
    y_values = torch.linspace(y_min, y_max, n)
    z_values = torch.linspace(z_min, z_max, n)

    all_points = []
    for angle in wind_angles_deg:
        cos_theta = torch.cos(torch.deg2rad(angle))
        sin_theta = torch.sin(torch.deg2rad(angle))
        
        for x in x_values:
            for y in y_values:
                for z in z_values:
                    point = [x, y, z, cos_theta, sin_theta]
                    all_points.append(point)

    points_tensor = torch.stack(all_points).to(device)

    return points_tensor

def evaluate_model(model, activation_function, X_test_tensor, y_test_tensor, feature_scaler, target_scaler, output_folder):
    model.eval()
    test_predictions = []
    test_predictions_wind_angle = []
    with torch.no_grad():
        predictions_tensor = model(X_test_tensor, activation_function)

        wind_angles = [0, 30, 60, 90, 120, 135, 150, 180]

        X_test_tensor_cpu = X_test_tensor.cpu()
        X_test_tensor_cpu = inverse_transform_features(X_test_tensor_cpu, feature_scaler)
        X_test_column_names = ["X", "Y", "Z", "cos(WindAngle)", "sin(WindAngle)"]
        X_test_dataframe = pd.DataFrame(X_test_tensor_cpu, columns=X_test_column_names)

        y_test_tensor_cpu = y_test_tensor.cpu()
        y_test_tensor_cpu = inverse_transform_targets(y_test_tensor_cpu, target_scaler)
        y_test_column_names = ['Pressure_Actual', 'Velocity_X_Actual', 'Velocity_Y_Actual', 'Velocity_Z_Actual', 'TurbVisc_Actual']
        y_test_dataframe = pd.DataFrame(y_test_tensor_cpu, columns=y_test_column_names)
        y_test_dataframe['Velocity_Magnitude_Actual'] = np.sqrt(y_test_dataframe['Velocity_X_Actual']**2 + 
                                                y_test_dataframe['Velocity_Y_Actual']**2 + 
                                                y_test_dataframe['Velocity_Z_Actual']**2)


        predictions_tensor_cpu = predictions_tensor.cpu()
        predictions_tensor_cpu = inverse_transform_targets(predictions_tensor_cpu, target_scaler)
        predictions_column_names = ['Pressure_Predicted', 'Velocity_X_Predicted', 'Velocity_Y_Predicted', 'Velocity_Z_Predicted', 'TurbVisc_Predicted']
        predictions_dataframe = pd.DataFrame(predictions_tensor_cpu, columns=predictions_column_names)
        predictions_dataframe['Velocity_Magnitude_Predicted'] = np.sqrt(predictions_dataframe['Velocity_X_Predicted']**2 + 
                                                predictions_dataframe['Velocity_X_Predicted']**2 + 
                                                predictions_dataframe['Velocity_X_Predicted']**2)

        rows_list = []
        for i, var in enumerate(y_test_column_names):
            var_cleaned = var.replace('_Actual', '')
            actuals = y_test_dataframe.iloc[:, i]
            preds = predictions_dataframe.iloc[:, i]

            mse = sklearn.metrics.mean_squared_error(actuals, preds)
            rmse = np.sqrt(mse)
            mae = sklearn.metrics.mean_absolute_error(actuals, preds)
            r2 = sklearn.metrics.r2_score(actuals, preds)
            
            # Append the new row as a dictionary to the list
            rows_list.append({
                'Variable': var_cleaned, 
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            })
        
        data_folder = os.path.join(output_folder, f'data_output_for_wind_angle_all')
        os.makedirs(data_folder, exist_ok=True)

        combined_df = pd.concat([X_test_dataframe, y_test_dataframe, predictions_dataframe], axis=1)
        test_predictions.append([combined_df])
        combined_file_path = os.path.join(data_folder, f'combined_actuals_and_predictions_for_wind_angle_all.csv')
        combined_df.to_csv(combined_file_path, index=False)

        metrics_df = pd.DataFrame(rows_list)   
        metrics_file_path = os.path.join(data_folder, f'metrics_for_wind_angle_all.csv')
        metrics_df.to_csv(metrics_file_path, index=False)

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
                    
                    # Append the new row as a dictionary to the list
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
                test_predictions_wind_angle.append([wind_angle, combined_df])
                combined_file_path = os.path.join(data_folder, f'combined_actuals_and_predictions_for_wind_angle_{wind_angle}.csv')
                combined_df.to_csv(combined_file_path, index=False)

                metrics_df = pd.DataFrame(rows_list)   
                metrics_file_path = os.path.join(data_folder, f'metrics_for_wind_angle_{wind_angle}.csv')
                metrics_df.to_csv(metrics_file_path, index=False)

    return test_predictions, test_predictions_wind_angle

def evaluate_model_skipped(config, model, activation_function, X_test_tensor, y_test_tensor, feature_scaler, target_scaler, output_folder):
    model.eval()
    test_predictions = []
    test_predictions_wind_angle = []
    with torch.no_grad():
        predictions_tensor = model(X_test_tensor, activation_function)

        wind_angles = config["training"]["angle_to_leave_out"]

        X_test_tensor_cpu = X_test_tensor.cpu()
        X_test_tensor_cpu = inverse_transform_features(X_test_tensor_cpu, feature_scaler)
        X_test_column_names = ["X", "Y", "Z", "cos(WindAngle)", "sin(WindAngle)"]
        X_test_dataframe = pd.DataFrame(X_test_tensor_cpu, columns=X_test_column_names)

        y_test_tensor_cpu = y_test_tensor.cpu()
        y_test_tensor_cpu = inverse_transform_targets(y_test_tensor_cpu, target_scaler)
        y_test_column_names = ['Pressure_Actual', 'Velocity_X_Actual', 'Velocity_Y_Actual', 'Velocity_Z_Actual', 'TurbVisc_Actual']
        y_test_dataframe = pd.DataFrame(y_test_tensor_cpu, columns=y_test_column_names)
        y_test_dataframe['Velocity_Magnitude_Actual'] = np.sqrt(y_test_dataframe['Velocity_X_Actual']**2 + 
                                                y_test_dataframe['Velocity_Y_Actual']**2 + 
                                                y_test_dataframe['Velocity_Z_Actual']**2)


        predictions_tensor_cpu = predictions_tensor.cpu()
        predictions_tensor_cpu = inverse_transform_targets(predictions_tensor_cpu, target_scaler)
        predictions_column_names = ['Pressure_Predicted', 'Velocity_X_Predicted', 'Velocity_Y_Predicted', 'Velocity_Z_Predicted', 'TurbVisc_Predicted']
        predictions_dataframe = pd.DataFrame(predictions_tensor_cpu, columns=predictions_column_names)
        predictions_dataframe['Velocity_Magnitude_Predicted'] = np.sqrt(predictions_dataframe['Velocity_X_Predicted']**2 + 
                                                predictions_dataframe['Velocity_X_Predicted']**2 + 
                                                predictions_dataframe['Velocity_X_Predicted']**2)

        rows_list = []
        for i, var in enumerate(y_test_column_names):
            var_cleaned = var.replace('_Actual', '')
            actuals = y_test_dataframe.iloc[:, i]
            preds = predictions_dataframe.iloc[:, i]

            mse = sklearn.metrics.mean_squared_error(actuals, preds)
            rmse = np.sqrt(mse)
            mae = sklearn.metrics.mean_absolute_error(actuals, preds)
            r2 = sklearn.metrics.r2_score(actuals, preds)
            
            # Append the new row as a dictionary to the list
            rows_list.append({
                'Variable': var_cleaned, 
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            })
        
        data_folder = os.path.join(output_folder, f'data_output_for_wind_angle_all')
        os.makedirs(data_folder, exist_ok=True)

        combined_df = pd.concat([X_test_dataframe, y_test_dataframe, predictions_dataframe], axis=1)
        test_predictions.append([combined_df])
        combined_file_path = os.path.join(data_folder, f'combined_actuals_and_predictions_for_wind_angle_all.csv')
        combined_df.to_csv(combined_file_path, index=False)

        metrics_df = pd.DataFrame(rows_list)   
        metrics_file_path = os.path.join(data_folder, f'metrics_for_wind_angle_all.csv')
        metrics_df.to_csv(metrics_file_path, index=False)

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
                    
                    # Append the new row as a dictionary to the list
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
                test_predictions_wind_angle.append([wind_angle, combined_df])
                combined_file_path = os.path.join(data_folder, f'combined_actuals_and_predictions_for_wind_angle_{wind_angle}.csv')
                combined_df.to_csv(combined_file_path, index=False)

                metrics_df = pd.DataFrame(rows_list)   
                metrics_file_path = os.path.join(data_folder, f'metrics_for_wind_angle_{wind_angle}.csv')
                metrics_df.to_csv(metrics_file_path, index=False)

    return test_predictions, test_predictions_wind_angle