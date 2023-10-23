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
        print(f"GPU with free memory: {free_memory:.2f} GB and with optimal batch size: {max_batch_size_gpu}")
    else:
        cpu_memory = psutil.virtual_memory()
        free_memory = cpu_memory.available / (1024 ** 3)
        max_batch_size_cpu = free_memory // single_sample_memory
        print(f"CPU with free memory: {free_memory:.2f} and with optimal batch size: {max_batch_size_cpu}")
    optimal_batch_size = int(optimal_batch_size)
    print(f"Now using device: {device} with optimal batch size: {optimal_batch_size}")
    return device, optimal_batch_size

def get_filenames_from_folder(path, extension, startname):
    """Get a list of filenames from the specified folder that end with a given extension."""
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith(extension) and f.startswith(startname)]

def get_optimizer(model, optimizer_config):
    optimizer_type = optimizer_config["type"]
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
            df['cos(WindAngle)'] = np.abs(np.cos(np.deg2rad(wind_angle)))
            df['sin(WindAngle)'] = np.abs(np.sin(np.deg2rad(wind_angle)))
            
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

            if wind_angle == angle_to_leave_out:
                print (f'Skipping Angle = {wind_angle} degrees')
            else:
                # Add new columns with unique values for each file
                df['cos(WindAngle)'] = np.cos(np.deg2rad(wind_angle))
                df['sin(WindAngle)'] = np.sin(np.deg2rad(wind_angle))
                
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

        if wind_angle != angle_to_leave_out:
            print (f'Angle = {wind_angle} degrees part of dataset!')
        else:
            # Add new columns with unique values for each file
            df['cos(WindAngle)'] = np.cos(np.deg2rad(wind_angle))
            df['sin(WindAngle)'] = np.sin(np.deg2rad(wind_angle))
            
            # Append the modified DataFrame to the list
            dfs_skipped.append(df)

    data_skipped = pd.concat(dfs_skipped)

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

def get_pure_data(filenames, datafolder_path, wind_angle, variables, variable_to_plot, x_feature, y_feature):
    for filename in sorted(filenames):
        
        df = pd.read_csv(os.path.join(datafolder_path, filename))

        index_str = filename.split('_')[-1].split('.')[0]  # Extract the index part of the filename
        index = int(index_str)  # Convert the index to integer

        meteo_data = pd.read_csv(os.path.join(datafolder_path,'meteo.csv'))

        # Look up the corresponding row in the meteo.csv file
        meteo_row = meteo_data[meteo_data['index'] == index]

        # Extract the wind angle from the found row
        wind_angle_data = meteo_row['cs degree'].values[0]

        u_index = variables.index('Velocity:0')
        v_index = variables.index('Velocity:1')
        w_index = variables.index('Velocity:2')

        u_actual = df.iloc[:, u_index]
        v_actual = df.iloc[:, v_index]
        w_actual = df.iloc[:, w_index]

        velocity_actual = np.sqrt(u_actual**2 + v_actual**2 + w_actual**2)

        df['Total Velocity'] = velocity_actual

        if wind_angle_data == wind_angle:

            x_plot, y_plot = get_x_y_for_plot(x_feature, y_feature, df)

            z_actual = df.iloc[:, variables.index(variable_to_plot)]

            return x_plot, y_plot, z_actual

def get_pure_data_velocity(filenames, datafolder_path, wind_angle, variables, variable_to_plot, x_feature, y_feature):
    for filename in sorted(filenames):
        
        df = pd.read_csv(os.path.join(datafolder_path, filename))

        index_str = filename.split('_')[-1].split('.')[0]  # Extract the index part of the filename
        index = int(index_str)  # Convert the index to integer

        meteo_data = pd.read_csv(os.path.join(datafolder_path,'meteo.csv'))

        # Look up the corresponding row in the meteo.csv file
        meteo_row = meteo_data[meteo_data['index'] == index]

        # Extract the wind angle from the found row
        wind_angle_data = meteo_row['cs degree'].values[0]

        if wind_angle_data == wind_angle:

            x_plot, y_plot = get_x_y_for_plot(x_feature, y_feature, df)

            u_index = variables.index('Velocity:0')
            v_index = variables.index('Velocity:1')
            w_index = variables.index('Velocity:2')

            u_actual = df.iloc[:, u_index]
            v_actual = df.iloc[:, v_index]
            w_actual = df.iloc[:, w_index]

            z_actual = np.sqrt(u_actual**2 + v_actual**2 + w_actual**2)

            return x_plot, y_plot, z_actual

def load_skipped_angle_data(filenames, base_directory, datafolder_path, device, config, feature_scaler, target_scaler):

    rho = config["data"]["density"]
    nu = config["data"]["kinematic_viscosity"]
    angle_to_leave_out = config["training"]["angle_to_leave_out"]

    dfs = []

    if angle_to_leave_out is None:
        print ('No angle was left out during training session')
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

            if wind_angle == angle_to_leave_out:
                 # Add new columns with unique values for each file
                df['cos(WindAngle)'] = np.abs(np.cos(np.deg2rad(wind_angle)))
                df['sin(WindAngle)'] = np.abs(np.sin(np.deg2rad(wind_angle)))
                dfs.append(df)
            else:
                print (f'Skipping Angle = {wind_angle} degrees that was trained earlier')

    data = pd.concat(dfs)
    # Extract features from the dataframe
    features = data[['Points:0', 'Points:1', 'Points:2', 'cos(WindAngle)', 'sin(WindAngle)']]
    targets = data[['Pressure', 'Velocity:0', 'Velocity:1', 'Velocity:2', 'TurbVisc']]

    normalized_features, normalized_targets = transform_data_with_scalers(features, targets, feature_scaler, target_scaler)
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(normalized_features, normalized_targets, range(len(normalized_features)),test_size=len(data)-1, random_state=42)

    # Convert to PyTorch Tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    X_train_tensor = X_train_tensor.to(device)
    y_train_tensor = y_train_tensor.to(device)
    X_test_tensor = X_test_tensor.to(device)
    y_test_tensor = y_test_tensor.to(device)

    return features, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, idx_train, idx_test

def calculate_alpha(available_gpu_memory, total_gpu_memory):
    alpha = 1 - available_gpu_memory / total_gpu_memory
    return alpha

def get_variables_to_plot(list_of_directions, variables):

    variables_to_plot = []

    for variable in variables:
        for direction1 in list_of_directions:
            for direction2 in list_of_directions:
                if direction2 != direction1:
                    x = [[direction1,direction2],variable]
                    y = [[direction2,direction1],variable]
                    if x not in variables_to_plot:
                        if y not in variables_to_plot:
                            variables_to_plot.append(x)

    return variables_to_plot

def get_list_of_directions(list_of_directions):

    variables_to_plot = []

    for direction1 in list_of_directions:
        for direction2 in list_of_directions:
            if direction2 != direction1:
                x = [direction1,direction2]
                y = [direction2,direction1]
                if x not in variables_to_plot:
                    if y not in variables_to_plot:
                        variables_to_plot.append(x)

    return variables_to_plot

def evaluate_model(model, X_test_tensor, y_test_tensor, feature_scaler, target_scaler, output_folder):
    model.eval()
    test_predictions = []
    test_predictions_wind_angle = []
    with torch.no_grad():
        predictions_tensor = model(X_test_tensor)
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

        test_predictions.append([X_test_dataframe, y_test_dataframe, predictions_dataframe])

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

        X_test_dataframe['WindAngle_rad'] = np.arctan2(X_test_dataframe['sin(WindAngle)'], X_test_dataframe['cos(WindAngle)'])
        X_test_dataframe['WindAngle_deg'] = X_test_dataframe['WindAngle_rad'].apply(lambda x: int(np.ceil(np.degrees(x))))
        combined_df = pd.concat([X_test_dataframe, y_test_dataframe, predictions_dataframe], axis=1)
        combined_file_path = os.path.join(data_folder, f'combined_actuals_and_predictions_for_wind_angle_all.csv')
        combined_df.to_csv(combined_file_path, index=False)

        metrics_df = pd.DataFrame(rows_list)   
        metrics_file_path = os.path.join(data_folder, f'metrics_for_wind_angle_all.csv')
        metrics_df.to_csv(metrics_file_path, index=False)

        # Now, if you want to filter the dataframe based on a specific angle (in degrees), you can do:
        for wind_angle in wind_angles:
            mask = (X_test_dataframe['WindAngle_deg'] == wind_angle)
            print (wind_angle, X_test_dataframe['WindAngle_deg'])
            filtered_X_test_dataframe = X_test_dataframe.loc[mask]

            filtered_y_test = y_test_dataframe.loc[mask]
            filtered_predictions = predictions_dataframe.loc[mask]

            print (X_test_dataframe.shape, filtered_X_test_dataframe.shape)

            test_predictions_wind_angle.append([wind_angle, filtered_X_test_dataframe, filtered_y_test, filtered_predictions])

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
                combined_file_path = os.path.join(data_folder, f'combined_actuals_and_predictions_for_wind_angle_{wind_angle}.csv')
                combined_df.to_csv(combined_file_path, index=False)

                metrics_df = pd.DataFrame(rows_list)   
                metrics_file_path = os.path.join(data_folder, f'metrics_for_wind_angle_{wind_angle}.csv')
                metrics_df.to_csv(metrics_file_path, index=False)

    return test_predictions, test_predictions_wind_angle

def get_x_y_for_plot(x_feature, y_feature, features, idx_test=[]):

    x = features['Points:0'].to_numpy()
    y = features['Points:1'].to_numpy()
    z = features['Points:2'].to_numpy()

    if len(idx_test)!=0:
        x = x[idx_test]
        y = y[idx_test]
        z = z[idx_test]

    # Scaling
    # x, y to [-1, 1]
    # z to [0, 1]
    x = (x - 500.0) / 300.0
    y = (y - 500.0) / 300.0
    z = z / 300.0

    if x_feature == 'Points:0':
        x_plot = x
    elif x_feature == 'Points:1':
        x_plot = y
    elif x_feature == 'Points:2':
        x_plot = z
    if y_feature == 'Points:0':
        y_plot = x
    elif y_feature == 'Points:1':
        y_plot = y
    elif y_feature == 'Points:2':
        y_plot = z
    return x_plot, y_plot

def prepare_2d_subplots(x_feature, y_feature, z_actual, variable_to_plot, wind_angle=None):
    fig, axs = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    if wind_angle is not None:
        fig.suptitle(f'Comparison of Actual vs. Predicted {variable_to_plot} values with Wind Angle = {wind_angle}, Max {variable_to_plot} = {max(z_actual):.2f}, num = {len(z_actual)}')
    else:
        fig.suptitle(f'Comparison of Actual vs. Predicted {variable_to_plot} values, Max {variable_to_plot} = {max(z_actual):.2f}, num = {len(z_actual)}')
    plt.tight_layout(pad=2)
    return fig, axs

def prepare_2d_plots(x_feature, y_feature, z_actual, variable_to_plot, wind_angle=None):
    fig, ax = plt.subplots(figsize=(16, 16))
    if wind_angle is not None:
        title = f'Comparison of Actual vs. Predicted {variable_to_plot} values with Wind Angle = {wind_angle}, Max {variable_to_plot} = {max(z_actual):.2f}, num = {len(z_actual)}'
    else:
        title = f'Comparison of Actual vs. Predicted {variable_to_plot} values, Max {variable_to_plot} = {max(z_actual):.2f}, num = {len(z_actual)}'
    
    ax.set_title(title)
    plt.tight_layout(pad=2)
    return fig, ax

def geometry_coordinates():
    # Given values
    x_sphere, y_sphere, z_sphere = 500, 500, 50
    r_sphere = 5

    x_cylinder, y_cylinder, z_cylinder = 500, 570, 0
    r_cylinder_bottom = 7.5
    r_cylinder_top = 6.5
    h_cylinder = 65

    # Scaling
    x_sphere_scaled = (x_sphere - 500.0) / 300.0
    y_sphere_scaled = (y_sphere - 500.0) / 300.0
    z_sphere_scaled = z_sphere / 300.0
    r_sphere_scaled = r_sphere / 300.0

    x_cylinder_scaled = (x_cylinder - 500.0) / 300.0
    y_cylinder_scaled = (y_cylinder - 500.0) / 300.0
    z_cylinder_scaled = z_cylinder / 300.0
    r_cylinder_bottom_scaled = r_cylinder_bottom / 300.0
    r_cylinder_top_scaled = r_cylinder_top / 300.0
    h_cylinder_scaled = h_cylinder / 300.0

def axes_limits_and_labels(axs, x_feature, y_feature, variable_to_plot, wind_angle=None):

    if x_feature == 'Points:0':
        axs[0].set_xlim([-1, 1])
        axs[0].set_xlabel("X")
        axs[1].set_xlim([-1, 1])
        axs[1].set_xlabel("X")
    elif x_feature == 'Points:1':
        axs[0].set_xlim([-1, 1])
        axs[0].set_xlabel("Y")
        axs[1].set_xlim([-1, 1])
        axs[1].set_xlabel("Y")
    elif x_feature == 'Points:2':
        axs[0].set_xlim([0, 1])
        axs[0].set_xlabel("Z")
        axs[1].set_xlim([0, 1])
        axs[1].set_xlabel("Z")
    if y_feature == 'Points:0':
        axs[0].set_ylim([-1, 1])
        axs[0].set_ylabel("X")
        axs[1].set_ylim([-1, 1])
        axs[1].set_ylabel("X")
    elif y_feature == 'Points:1':
        axs[0].set_ylim([-1, 1])
        axs[0].set_ylabel("Y")
        axs[1].set_ylim([-1, 1])
        axs[1].set_ylabel("Y")
    elif y_feature == 'Points:2':
        axs[0].set_ylim([0, 1])
        axs[0].set_ylabel("Z")
        axs[1].set_ylim([0, 1])
        axs[1].set_ylabel("Z")

    if x_feature == "Points:0" and y_feature == "Points:1":
        if wind_angle is not None:
            axs[0].set_title(f'XY Plot for Actual {variable_to_plot} with Wind Angle = {wind_angle}')
            axs[1].set_title(f'XY Plot for Predicted {variable_to_plot} with Wind Angle = {wind_angle}')
        else:
            axs[0].set_title(f'XY Plot for Actual {variable_to_plot} with all Wind Angles')
            axs[1].set_title(f'XY Plot for Predicted {variable_to_plot} with Wind Angles')

    if x_feature == "Points:0" and y_feature == "Points:2":
        if wind_angle is not None:
            axs[0].set_title(f'XZ Plot for Actual {variable_to_plot} with Wind Angle = {wind_angle}')
            axs[1].set_title(f'XZ Plot for Predicted {variable_to_plot} with Wind Angle = {wind_angle}')
        else:
            axs[0].set_title(f'XZ Plot for Actual {variable_to_plot} with all Wind Angles')
            axs[1].set_title(f'XZ Plot for Predicted {variable_to_plot} with all Wind Angles')

    if x_feature == "Points:1" and y_feature == "Points:2":
        if wind_angle is not None:
            axs[0].set_title(f'YZ Plot for Actual {variable_to_plot} with Wind Angle = {wind_angle}')
            axs[1].set_title(f'YZ Plot for Predicted {variable_to_plot} with Wind Angle = {wind_angle}')
        else:
            axs[0].set_title(f'YZ Plot for Actual {variable_to_plot} with all Wind Angles')
            axs[1].set_title(f'YZ Plot for Predicted {variable_to_plot} with all Wind Angles')

def axis_limits_and_labels(ax, x_feature, y_feature, variable_to_plot, wind_angle=None):

    if x_feature == 'Points:0':
        ax.set_xlim([-1, 1])
        ax.set_xlabel("X")
    elif x_feature == 'Points:1':
        ax.set_xlim([-1, 1])
        ax.set_xlabel("Y")
    elif x_feature == 'Points:2':
        ax.set_xlim([0, 1])
        ax.set_xlabel("Z")
    if y_feature == 'Points:0':
        ax.set_ylim([-1, 1])
        ax.set_ylabel("X")
    elif y_feature == 'Points:1':
        ax.set_ylim([-1, 1])
        ax.set_ylabel("Y")
    elif y_feature == 'Points:2':
        ax.set_ylim([0, 1])
        ax.set_ylabel("Z")

    if x_feature == "Points:0" and y_feature == "Points:1":
        if wind_angle is not None:
            ax.set_title(f'XY Plot for Actual {variable_to_plot} with Wind Angle = {wind_angle}')
        else:
            ax.set_title(f'XY Plot for Actual {variable_to_plot} with all Wind Angles')

    if x_feature == "Points:0" and y_feature == "Points:2":
        if wind_angle is not None:
            ax.set_title(f'XZ Plot for Actual {variable_to_plot} with Wind Angle = {wind_angle}')
        else:
            ax.set_title(f'XZ Plot for Actual {variable_to_plot} with all Wind Angles')

    if x_feature == "Points:1" and y_feature == "Points:2":
        if wind_angle is not None:
            ax.set_title(f'YZ Plot for Actual {variable_to_plot} with Wind Angle = {wind_angle}')
        else:
            ax.set_title(f'YZ Plot for Actual {variable_to_plot} with all Wind Angles')

def save_2d_plot(output_folder, x_feature, y_feature, variable_to_plot, wind_angle=None):
    safe_x_feature = x_feature.replace(':', '')
    safe_y_feature = y_feature.replace(':', '')
    safe_var_name = variable_to_plot.replace(':', '_')
    if wind_angle is not None:
        plot_folder = os.path.join(output_folder, f'plots_{wind_angle}')
        os.makedirs(plot_folder, exist_ok=True)
        plt.savefig(os.path.join(plot_folder, f"{safe_x_feature}_{safe_y_feature}_{safe_var_name}_for_wind_angle_{wind_angle}.png"))
    else:
        plt.savefig(os.path.join(output_folder, f"{safe_x_feature}_{safe_y_feature}_{safe_var_name}_contour_comparison.png"))
    plt.close()

def compute_total_velocity(actual, predicted, variables):

        u_index = variables.index('Velocity:0')
        v_index = variables.index('Velocity:1')
        w_index = variables.index('Velocity:2')

        u_actual = actual[:, u_index]
        u_predicted = predicted[:, u_index]
        v_actual = actual[:, v_index]
        v_predicted = predicted[:, v_index]
        w_actual = actual[:, w_index]
        w_predicted = predicted[:, w_index]

        # Compute the magnitude of the velocity vector
        velocity_actual = np.sqrt(u_actual**2 + v_actual**2 + w_actual**2)
        velocity_predicted = np.sqrt(u_predicted**2 + v_predicted**2 + w_predicted**2)


        return velocity_actual, velocity_predicted

def plotting_details(z_actual):
    mean_z = np.mean(z_actual)
    std_z = np.std(z_actual)
    vmin = mean_z - 2 * std_z
    vmax = mean_z + 2 * std_z

    vmin = int(vmin)
    vmax = int(vmax)

    cmap = plt.cm.RdBu_r
    scatter_size = 5
    ticks = 5

    return vmin, vmax, cmap, scatter_size, ticks

def normalize_grids(plane, points1, points2):
    if plane == 'X-Z':
        points1 = (points1 - 500)/300
        points2 = (points2)/300
    if plane == 'Y-Z':
        points1 = (points1 - 500)/300
        points2 = (points2)/300
    if plane == 'X-Y':
        points1 = (points1 - 500)/300
        points2 = (points2 - 500)/300

    return points1, points2


def normalize_cut_tolerance(plane, cut, tolerance):
    if plane == 'X-Z':
        cut = (cut - 500)/300
        tolerance = (tolerance - 500)/300
    if plane == 'Y-Z':
        cut = (cut - 500)/300
        tolerance = (tolerance - 500)/300
    if plane == 'X-Y':
        cut = (cut)/300
        tolerance = (tolerance)/300

    return cut, tolerance


def geometry_coordinates(plane):
    # Generate scatter points within the sphere's volume
    phi = np.linspace(0, 2*np.pi, 100)
    theta = np.linspace(0, np.pi, 100)
    theta, phi = np.meshgrid(theta, phi)
    sphere_center_x = 500
    sphere_radius_x = 5
    sphere_center_y = 500
    sphere_radius_y = 5
    sphere_center_z = 50
    sphere_radius_z = 5
    theta_min = 0
    theta_max = np.pi
    phi_min = 0
    phi_max = 2 * np.pi
    r_min = 0  # Minimum radius
    r_max = 5  # Maximum radius (sphere's radius)
    num_points = 1000  # Number of scatter points
    theta_values = np.random.uniform(theta_min, theta_max, num_points)
    phi_values = np.random.uniform(phi_min, phi_max, num_points)
    r_values = np.random.uniform(r_min, r_max, num_points)
    x_sphere_fill = sphere_center_x + r_values * np.sin(theta_values) * np.cos(phi_values)
    y_sphere_fill = sphere_center_y + r_values * np.sin(theta_values) * np.sin(phi_values)
    z_sphere_fill = sphere_center_z + r_values * np.cos(theta_values)

    # Generate scatter points for the cylinder body
    cylinder_center_x = 500
    cylinder_radius_x = 7.5
    cylinder_center_y = 570
    cylinder_radius_y = 7.5
    cylinder_height = 65

    # Generate scatter points for the cylinder's rounded cap
    z_cap_corrected = np.linspace(cylinder_height - 1, cylinder_height, 100)
    
    if plane == 'Y-Z':
        y_cylinder_body_fill, z_cylinder_body_fill = np.meshgrid(np.linspace(cylinder_center_y - cylinder_radius_y, cylinder_center_y + cylinder_radius_y, 100), np.linspace(0, cylinder_height, 100))
        y_cylinder_cap, z_cylinder_cap_corrected = np.meshgrid(np.linspace(cylinder_center_y - cylinder_radius_y, cylinder_center_y + cylinder_radius_y, 100),  z_cap_corrected)
        return y_sphere_fill, z_sphere_fill, y_cylinder_body_fill, z_cylinder_body_fill, y_cylinder_cap, z_cylinder_cap_corrected
    if plane == 'X-Z':
        x_cylinder_body_fill, z_cylinder_body_fill = np.meshgrid(np.linspace(cylinder_center_x - cylinder_radius_x, cylinder_center_x + cylinder_radius_x, 100), np.linspace(0, cylinder_height, 100))
        x_cylinder_cap, z_cylinder_cap_corrected = np.meshgrid(np.linspace(cylinder_center_x - cylinder_radius_x, cylinder_center_x + cylinder_radius_x, 100),  z_cap_corrected)
        return x_sphere_fill, z_sphere_fill, x_cylinder_body_fill, z_cylinder_body_fill, x_cylinder_cap, z_cylinder_cap_corrected
    if plane == 'X-Y':
        x_cylinder_body_fill, y_cylinder_body_fill = np.meshgrid(np.linspace(cylinder_center_x - cylinder_radius_x, cylinder_center_x + cylinder_radius_x, 100), np.linspace(cylinder_center_y - cylinder_radius_y, cylinder_center_y + cylinder_radius_y, 100))
        x_cylinder_cap, y_cylinder_cap = np.meshgrid(np.linspace(cylinder_center_x - cylinder_radius_x, cylinder_center_x + cylinder_radius_x, 100),  np.linspace(cylinder_center_y - cylinder_radius_y, cylinder_center_y + cylinder_radius_y, 100))
        return x_sphere_fill, y_sphere_fill, x_cylinder_body_fill, y_cylinder_body_fill, x_cylinder_cap, y_cylinder_cap


def plot_data(filename,angle,savename,plane,cut,tolerance,cmap):
    
    df = pd.read_csv(filename)

    if plane == 'X-Z':
        points = ['Points:0', 'Points:2']
        noplotdata = 'Points:1'
        coordinate3 = 'Y'
        lim_min1, lim_max1 = (-1,1)
        lim_min2, lim_max2 = (0,1)
    if plane == 'Y-Z':
        points = ['Points:1', 'Points:2']
        noplotdata = 'Points:0'
        coordinate3 = 'X'
        lim_min1, lim_max1 = (-1,1)
        lim_min2, lim_max2 = (0,1)
    if plane == 'X-Y':
        points = ['Points:0', 'Points:1']
        noplotdata = 'Points:2'
        coordinate3 = 'Z'
        lim_min1, lim_max1 = (-1,1)
        lim_min2, lim_max2 = (-1,1)

    points1 = points[0]
    points2 = points[1]

    coordinate1 = plane.split('-')[0]
    coordinate2 = plane.split('-')[1]

    # Filter the data to focus on the y-z plane
    filtered_df = df[(df[noplotdata] >= cut - tolerance) & (df[noplotdata] <= cut + tolerance)]

    # Define a regular grid covering the range of y and z coordinates
    grid1, grid2 = np.mgrid[filtered_df[points1].min():filtered_df[points1].max():1000j, filtered_df[points2].min():filtered_df[points2].max():1000j]

    # Interpolate all velocity components onto the grid
    grid_vx = griddata(filtered_df[[points1, points2]].values, filtered_df['Velocity:0'].values, (grid1, grid2), method='linear', fill_value=0)
    grid_vy = griddata(filtered_df[[points1, points2]].values, filtered_df['Velocity:1'].values, (grid1, grid2), method='linear', fill_value=0)
    grid_vz = griddata(filtered_df[[points1, points2]].values, filtered_df['Velocity:2'].values, (grid1, grid2), method='linear', fill_value=0)

    # Calculate the total magnitude of the velocity on the grid
    grid_magnitude = np.sqrt(grid_vx**2 + grid_vy**2 + grid_vz**2)

    spherefill1, spherefill2, cylinderfill1, cylinderfill2, cylindercapfill1, cylindercapfill2 = geometry_coordinates(plane)


    grid1, grid2 = normalize_grids(plane, grid1, grid2)
    spherefill1, spherefill2 = normalize_grids(plane, spherefill1, spherefill2)
    cylinderfill1, cylinderfill2 = normalize_grids(plane, cylinderfill1, cylinderfill2)
    cylindercapfill1, cylindercapfill2 = normalize_grids(plane, cylindercapfill1, cylindercapfill2)
    cut, tolerance = normalize_cut_tolerance(plane, cut, tolerance)

    # Visualize using matplotlib
    fig, ax = plt.subplots(figsize=(12, 8))
    contour = ax.contourf(grid1, grid2, grid_magnitude, levels=128, cmap=cmap)
    ax.scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
    ax.scatter(cylinderfill1.ravel(),  cylinderfill2.ravel(), c='black', s=1, label='Cylinder Body Fill')
    ax.scatter(cylindercapfill1.ravel(), cylindercapfill2, c='black', s=1, label='Cylinder Cap Fill')
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label('Velocity Magnitude', rotation=270, labelpad=15)
    ax.set_xlabel(f'{coordinate1} Coordinate')
    ax.set_ylabel(f'{coordinate2} Coordinate')
    ax.set_xlim(lim_min1, lim_max1) 
    ax.set_ylim(lim_min2, lim_max2)
    ax.set_title(f'Total Velocity in the {plane} Plane for Wind Angle = {angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')
    plt.tight_layout()
    plt.savefig(savename)
    plt.close()

def plot_data_predictions(X_test_dataframe,y_test_dataframe,predictions_dataframe,angle,savename,plane,cut,tolerance,cmap):
    wind_angle = angle
    if plane == 'X-Z':
        points = ['X', 'Z']
        noplotdata = 'Y'
        coordinate3 = 'Y'
        lim_min1, lim_max1 = (-1,1)
        lim_min2, lim_max2 = (0,1)
    if plane == 'Y-Z':
        points = ['Y', 'Z']
        noplotdata = 'X'
        coordinate3 = 'X'
        lim_min1, lim_max1 = (-1,1)
        lim_min2, lim_max2 = (0,1)
    if plane == 'X-Y':
        points = ['X', 'Y']
        noplotdata = 'Z'
        coordinate3 = 'Z'
        lim_min1, lim_max1 = (-1,1)
        lim_min2, lim_max2 = (-1,1)

    points1 = points[0]
    points2 = points[1]

    coordinate1 = plane.split('-')[0]
    coordinate2 = plane.split('-')[1]

    cos_val = round(np.cos(np.radians(wind_angle)), 3)
    sin_val = round(np.sin(np.radians(wind_angle)), 3)
    
    mask = (X_test_dataframe.iloc[:, 3].round(3) == cos_val) & (X_test_dataframe.iloc[:, 4].round(3) == sin_val)
    X_test_dataframe = X_test_dataframe.loc[mask]
    y_test_dataframe = y_test_dataframe.loc[mask]
    predictions_dataframe = predictions_dataframe.loc[mask]

    df = pd.concat([X_test_dataframe, y_test_dataframe, predictions_dataframe], axis=1)

    # Filter the data to focus on the y-z plane/...
    filtered_df = df[(df[noplotdata] >= cut - tolerance) & (df[noplotdata] <= cut + tolerance)]

    print (wind_angle, filtered_df.shape)

    # Define a regular grid covering the range of y and z coordinates/...
    grid1, grid2 = np.mgrid[filtered_df[points1].min():filtered_df[points1].max():1000j, filtered_df[points2].min():filtered_df[points2].max():1000j]

    # Interpolate all velocity components onto the grid
    try:
        grid_vx_actual = griddata(filtered_df[[points1, points2]].values, filtered_df['Velocity_X_Actual'].values, (grid1, grid2), method='linear', fill_value=0)
        grid_vy_actual = griddata(filtered_df[[points1, points2]].values, filtered_df['Velocity_Y_Actual'].values, (grid1, grid2), method='linear', fill_value=0)
        grid_vz_actual = griddata(filtered_df[[points1, points2]].values, filtered_df['Velocity_Z_Actual'].values, (grid1, grid2), method='linear', fill_value=0)

        grid_vx_pred = griddata(filtered_df[[points1, points2]].values, filtered_df['Velocity_X_Predicted'].values, (grid1, grid2), method='linear', fill_value=0)
        grid_vy_pred = griddata(filtered_df[[points1, points2]].values, filtered_df['Velocity_Y_Predicted'].values, (grid1, grid2), method='linear', fill_value=0)
        grid_vz_pred = griddata(filtered_df[[points1, points2]].values, filtered_df['Velocity_Z_Predicted'].values, (grid1, grid2), method='linear', fill_value=0)
    except:
        print(f"not enough points")
        return

    # Calculate the total magnitude of the velocity on the grid
    grid_magnitude_actual = np.sqrt(grid_vx_actual**2 + grid_vy_actual**2 + grid_vz_actual**2)
    grid_magnitude_pred = np.sqrt(grid_vx_pred**2 + grid_vy_pred**2 + grid_vz_pred**2)

    spherefill1, spherefill2, cylinderfill1, cylinderfill2, cylindercapfill1, cylindercapfill2 = geometry_coordinates(plane)

    grid1, grid2 = normalize_grids(plane, grid1, grid2)
    spherefill1, spherefill2 = normalize_grids(plane, spherefill1, spherefill2)
    cylinderfill1, cylinderfill2 = normalize_grids(plane, cylinderfill1, cylinderfill2)
    cylindercapfill1, cylindercapfill2 = normalize_grids(plane, cylindercapfill1, cylindercapfill2)
    cut, tolerance = normalize_cut_tolerance(plane, cut, tolerance)

    fig, axs = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    fig.suptitle(f'Comparison of Actual vs. Predicted Total Velocity values with Wind Angle = {wind_angle}')
    contour_actual = axs[0].contourf(grid1, grid2, grid_magnitude_actual, levels=128, cmap=cmap)
    axs[0].scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
    axs[0].scatter(cylinderfill1.ravel(),  cylinderfill2.ravel(), c='black', s=1, label='Cylinder Body Fill')
    axs[0].scatter(cylindercapfill1.ravel(), cylindercapfill2, c='black', s=1, label='Cylinder Cap Fill')
    cbar = fig.colorbar(contour_actual, ax=axs[0])
    cbar.set_label('Velocity Magnitude - Actual', rotation=270, labelpad=15)
    axs[0].set_xlabel(f'{coordinate1} Coordinate')
    axs[0].set_ylabel(f'{coordinate2} Coordinate')
    axs[0].set_xlim(lim_min1, lim_max1) 
    axs[0].set_ylim(lim_min2, lim_max2)
    axs[0].set_title(f'Total Actual Velocity in the {plane} Plane for Wind Angle = {angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')
    contour_pred = axs[1].contourf(grid1, grid2, grid_magnitude_pred, levels=128, cmap=cmap)
    axs[1].scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
    axs[1].scatter(cylinderfill1.ravel(),  cylinderfill2.ravel(), c='black', s=1, label='Cylinder Body Fill')
    axs[1].scatter(cylindercapfill1.ravel(), cylindercapfill2, c='black', s=1, label='Cylinder Cap Fill')
    cbar = fig.colorbar(contour_pred, ax=axs[1])
    cbar.set_label('Velocity Magnitude - Predicted', rotation=270, labelpad=15)
    axs[1].set_xlabel(f'{coordinate1} Coordinate')
    axs[1].set_ylabel(f'{coordinate2} Coordinate')
    axs[1].set_xlim(lim_min1, lim_max1) 
    axs[1].set_ylim(lim_min2, lim_max2)
    axs[1].set_title(f'Total Predicted Velocity in the {plane} Plane for Wind Angle = {angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')
    plt.tight_layout()
    plt.savefig(savename)
    plt.close()