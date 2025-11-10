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
import re
import csv
import pandas as pd
import random
from pathlib import Path
import importlib.util
import sklearn
import collections
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.tri as tri
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from scipy.interpolate import griddata
from scipy.stats import linregress
import torch.multiprocessing as mp
from stl import mesh
# import FreeCAD
# import Mesh

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

def save_to_csv(epoch, epochs, use_epoch, current_loss, current_elapsed_time_hours, free_memory, data_loss, cont_loss, momentum_loss, avg_no_slip_loss, avg_inlet_loss, total_loss, weighted_total_loss, file_path):
    data = {
        'Epoch': f'{epoch}',
        'Loss': current_loss.cpu().detach().numpy(),
        'Total Time Elapsed (hours)': f'{current_elapsed_time_hours:.2f}',
        'Free Memory (GB)': f'{free_memory:.2f}',
        'Data Loss': data_loss if data_loss == 0 else data_loss.cpu().detach().numpy(),
        'Continuity Loss': cont_loss if cont_loss == 0 else cont_loss.cpu().detach().numpy(),
        'Momentum Loss': momentum_loss if momentum_loss == 0 else momentum_loss.cpu().detach().numpy(),
        'Averaged No Slip Loss': avg_no_slip_loss if avg_no_slip_loss == 0 else avg_no_slip_loss.cpu().detach().numpy(),
        'Averaged Inlet Loss': avg_inlet_loss if avg_inlet_loss == 0 else avg_inlet_loss.cpu().detach().numpy(),
        'Total Loss': total_loss if total_loss == 0 else total_loss.cpu().detach().numpy(),
        'Total Loss Weighted': weighted_total_loss if weighted_total_loss == 0 else weighted_total_loss.cpu().detach().numpy()
    }

    # Check if file exists and has content
    file_exists = os.path.isfile(file_path) and os.path.getsize(file_path) > 0

    # Writing to CSV
    with open(file_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data.keys())
        # Write the header only if the file is new or empty
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

def save_evaluation_to_csv(epoch, epochs, use_epoch, output_params, mses, r2s, file_path):
    # Prepare the header and data
    headers = ['Epoch'] + [f'MSE_{param}' for param in output_params] + [f'R2_{param}' for param in output_params]
    epoch_label = f'{epoch}'
    data = [epoch_label] + mses + r2s

    # Check if file exists
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write the header if the file is new
        if not file_exists:
            writer.writerow(headers)

        # Write the data
        writer.writerow(data)

def process_epoch_lines(file_path):
    epoch_lines = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('Epoch'):
                epoch_lines.append(line.strip())  # Remove trailing newline character
    return epoch_lines

def concatenate_files(directory, new_file_path):

    # List all files and directories in the specified directory
    file_paths = os.listdir(directory)

    # Open the new file in write mode
    with open(os.path.join(directory, new_file_path), 'w') as new_file:
        for path in file_paths:
            if path.endswith('.txt') and path.startswith('output_log'):
                print (f'now processing {path}')
                epoch_lines = process_epoch_lines(path)
                for line in epoch_lines:
                    new_file.write(line)
                
    print("Files concatenated successfully into:", os.path.join(directory, new_file_path))

def filter_info_file(directory):
    df = pd.read_csv(os.path.join(directory, 'info.csv'))
    
    previous_time = 0
    current_time = 0
    numtoplot = 200

    for i in df.values:
        time = float(i[2])
        if time < previous_time:
            current_time += previous_time
        previous_time = time
        written_time = current_time + time

    total_epochs = df['Epoch'].iloc[-1]

    df = df.loc[:, (df != 0).any(axis=0)]

    # numtoskip = int(len(df)/numtoplot)
    # df = df[(df['Epoch'] - 5) % numtoskip == 0]

    return df, written_time, total_epochs

def filter_trainingloss_file(directory, filename):
    df = pd.read_csv(os.path.join(directory, filename))
    
    numtoplot = 200

    # numtoskip = int(len(df)/numtoplot)
    # df = df[(df['Epoch'] - 5) % numtoskip == 0]

    return df

def get_log_analysis(directory):
    # new_file_path = 'all_output_log.txt'

    # # concatenate_files(directory, new_file_path)
    # # numtoskip = int(len(df)/numtoplot)
    # # df = df[(df['Epoch'] - 5) % numtoskip == 0]

    df, current_time, total_epochs = filter_info_file(directory)
    
    return df, current_time, total_epochs

def convert_state_dict(config, state_dict):

    if config["train_test"]["distributed_training"]:
        if any(key.startswith("module.module.") for key in state_dict):
            modified_state_dict = {k.replace("module.module.", "module."): v for k, v in state_dict.items()}
        elif not any(key.startswith("module.") for key in state_dict):
            modified_state_dict = {"module." + k: v for k, v in state_dict.items()}
        else:
            modified_state_dict = state_dict
    else:
        if any(key.startswith("module.") for key in state_dict):
            modified_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        else:
            modified_state_dict = state_dict

    new_state_dict = {}
    for old_key, value in modified_state_dict.items():
        new_key = old_key
        new_key = new_key.replace('fc1', 'layers.0')
        new_key = new_key.replace('fc2', 'layers.2')
        new_key = new_key.replace('fc3', 'layers.4')
        new_key = new_key.replace('fc4', 'layers.6')
        new_key = new_key.replace('output_layer', 'layers.8')
        new_state_dict[new_key] = value

    return new_state_dict

def restructure_data(faces_data):
    # Reshaping into faces
    faces = [tuple(faces_data[i:i+3]) for i in range(0, len(faces_data), 3)]
    return faces

def compute_normal(v1, v2, v3):
    # Convert to numpy arrays
    v1, v2, v3 = np.array(v1), np.array(v2), np.array(v3)
    # Compute edges
    edge1 = v2 - v1
    edge2 = v3 - v1
    # Compute normal
    normal = np.cross(edge1, edge2)
    # Normalize
    normal = normal / np.linalg.norm(normal)
    return normal

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

def load_plotting_data(filenames, base_directory, datafolder_path, config):

    dfs = []

    for filename in sorted(filenames):
        df = pd.read_csv(os.path.join(datafolder_path, filename))
        wind_angle = int(filename.split('_')[-1].split('.')[0])  # Extract the index part of the filename
        
        # Add new columns with unique values for each file
        df['WindAngle'] = (wind_angle)
        df['cos(WindAngle)'] = (np.cos(np.deg2rad(wind_angle)))
        df['sin(WindAngle)'] = (np.sin(np.deg2rad(wind_angle)))
        
        # Append the modified DataFrame to the list
        dfs.append(df)
    

    # Concatenate the list of DataFrames
    data = pd.concat(dfs)

    data.rename(columns={'Points:0': 'X', 'Points:1': 'Y', 'Points:2': 'Z', 'Velocity:0': 'Velocity_X', 'Velocity:1': 'Velocity_Y', 'Velocity:2': 'Velocity_Z'}, inplace=True)

    data['Velocity_Magnitude'] = np.sqrt(data['Velocity_X']**2 + 
                                                data['Velocity_Y']**2 + 
                                                data['Velocity_Z']**2)

    return data

def load_data(filenames, base_directory, datafolder_path, device, config):

    rho = config["data"]["density"]
    nu = config["data"]["kinematic_viscosity"]
    angle_to_leave_out = config["training"]["angle_to_leave_out"]

    dfs = []
    dfs_skipped = []

    if angle_to_leave_out is None:
        for filename in sorted(filenames):
            df = pd.read_csv(os.path.join(datafolder_path, filename))
            wind_angle = int(filename.split('_')[-2])  # Extract the index part of the filename
            
            # Add new columns with unique values for each file
            df['cos(WindAngle)'] = (np.cos(np.deg2rad(wind_angle)))
            df['sin(WindAngle)'] = (np.sin(np.deg2rad(wind_angle)))
            
            # Append the modified DataFrame to the list
            dfs.append(df)
    else:
        for filename in sorted(filenames):
            df = pd.read_csv(os.path.join(datafolder_path, filename))
            wind_angle = int(filename.split('_')[-1].split('.')[0])  # Extract the index part of the filename

            if wind_angle in angle_to_leave_out:
                pass
            else:
                # Add new columns with unique values for each file
                df['cos(WindAngle)'] = (np.cos(np.deg2rad(wind_angle)))
                df['sin(WindAngle)'] = (np.sin(np.deg2rad(wind_angle)))
                
                # Append the modified DataFrame to the list
                dfs.append(df)

    # Concatenate the list of DataFrames
    data = pd.concat(dfs)

    # Extract features from the dataframe
    features = data[config["training"]["input_params"]]
    targets = data[config["training"]["output_params"]]

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
        wind_angle = int(filename.split('_')[-1].split('.')[0])  # Extract the index part of the filename

        if wind_angle not in angle_to_leave_out:
            pass
        else:
            # Add new columns with unique values for each file
            df['cos(WindAngle)'] = (np.cos(np.deg2rad(wind_angle)))
            df['sin(WindAngle)'] = (np.sin(np.deg2rad(wind_angle)))
        
            # Append the modified DataFrame to the list
            dfs_skipped.append(df)

    data_skipped = pd.concat(dfs_skipped)

    # Extract features from the dataframe
    features_skipped = data_skipped[config["training"]["input_params"]]
    targets_skipped = data_skipped[config["training"]["output_params"]]

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

def load_data_new_angle(filenames, base_directory, datafolder_path, device, config, feature_scaler, target_scaler, wind_angle):

    dfs = []

    for filename in sorted(filenames):
        df = pd.read_csv(os.path.join(datafolder_path, filename))
        
        # Add new columns with unique values for each file
        df['cos(WindAngle)'] = (np.cos(np.deg2rad(wind_angle)))
        df['sin(WindAngle)'] = (np.sin(np.deg2rad(wind_angle)))
        
        # Append the modified DataFrame to the list
        dfs.append(df)

    # Concatenate the list of DataFrames
    data = pd.concat(dfs)

    # Extract features from the dataframe
    features = data[config["training"]["input_params"]]
    targets = data[config["training"]["output_params"]]

    normalized_features, normalized_targets = transform_data_with_scalers(features, targets, feature_scaler, target_scaler)

    # Perform the train-test split and get the indices
    X_train, X_test, y_train, y_test = train_test_split(normalized_features, normalized_targets,test_size=config["train_test"]["test_size_new_angle"], random_state=config["train_test"]["random_state"])

    # Convert to PyTorch Tensors
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    X_test_tensor = X_test_tensor.to(device)

    return X_test_tensor

def extract_unique_wind_angles_from_X(X):
    # Extract unique pairs of sin and cos values
    unique_pairs, _ = torch.unique(X[:, 3:5], dim=0, return_inverse=True)
    cos_theta = unique_pairs[:, 0]
    sin_theta = unique_pairs[:, 1]

    # Compute unique wind angles in radians
    wind_angle_rad = torch.atan2(sin_theta, cos_theta)
    wind_angle_deg = torch.rad2deg(wind_angle_rad)

    wind_angle_deg[wind_angle_deg < 0] += 360

    wind_angle_deg = torch.where(wind_angle_deg > 180, 360 - wind_angle_deg, wind_angle_deg)

    return wind_angle_deg

def extract_wind_angle_from_X_closest_match(X, wind_angles, tolerance):
    unique_wind_angles_deg = extract_unique_wind_angles_from_X(X)
    
    # Initialize an array to store the closest match for each unique angle
    closest_matches = torch.full_like(unique_wind_angles_deg, float('inf'))

    # Iterate over each angle in wind_angles
    for angle in wind_angles:
        diff = torch.abs(unique_wind_angles_deg - angle)
        mask = diff <= tolerance

        # Update the closest match only if the current angle is a closer match
        closer_match = diff < torch.abs(closest_matches - unique_wind_angles_deg)
        combined_mask = mask & closer_match
        closest_matches = torch.where(combined_mask, angle, closest_matches)

    # Replace infinities with original angles (where no close match was found)
    no_match = closest_matches == float('inf')
    closest_matches = torch.where(no_match, unique_wind_angles_deg, closest_matches)

    return closest_matches.tolist()

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

    points_tensor = torch.tensor(all_points, dtype=torch.float32)
    points_tensor = points_tensor.to(device)

    return points_tensor

def evaluate_model(config, model, X_test_tensor, y_test_tensor, feature_scaler, target_scaler, output_folder):
    model.eval()
    test_predictions = []
    test_predictions_wind_angle = []
    with torch.no_grad():
        predictions_tensor = model(X_test_tensor)

        wind_angles = config["training"]["all_angles"]

        X_test_tensor_cpu = X_test_tensor.cpu()
        X_test_tensor_cpu = inverse_transform_features(X_test_tensor_cpu, feature_scaler)
        X_test_column_names = config["training"]["input_params_modf"]
        X_test_dataframe = pd.DataFrame(X_test_tensor_cpu, columns=X_test_column_names)

        y_test_tensor_cpu = y_test_tensor.cpu()
        y_test_tensor_cpu = inverse_transform_targets(y_test_tensor_cpu, target_scaler)
        output_column_names = config["training"]["output_params_modf"]
        y_test_column_names = [item + "_Actual" for item in output_column_names]
        y_test_dataframe = pd.DataFrame(y_test_tensor_cpu, columns=y_test_column_names)
        y_test_dataframe['Velocity_Magnitude_Actual'] = np.sqrt(y_test_dataframe['Velocity_X_Actual']**2 + 
                                                y_test_dataframe['Velocity_Y_Actual']**2 + 
                                                y_test_dataframe['Velocity_Z_Actual']**2)


        predictions_tensor_cpu = predictions_tensor.cpu()
        predictions_tensor_cpu = inverse_transform_targets(predictions_tensor_cpu, target_scaler)
        predictions_column_names = [item + "_Predicted" for item in output_column_names]
        predictions_dataframe = pd.DataFrame(predictions_tensor_cpu, columns=predictions_column_names)
        predictions_dataframe['Velocity_Magnitude_Predicted'] = np.sqrt(predictions_dataframe['Velocity_X_Predicted']**2 + 
                                                predictions_dataframe['Velocity_Y_Predicted']**2 + 
                                                predictions_dataframe['Velocity_Z_Predicted']**2)

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

        dfs = []

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

                dfs.append(combined_df)

        data = pd.concat(dfs)

    return test_predictions, test_predictions_wind_angle, data

def evaluate_model_skipped(config, model, X_test_tensor, y_test_tensor, feature_scaler, target_scaler, output_folder):
    model.eval()
    test_predictions = []
    test_predictions_wind_angle = []
    with torch.no_grad():
        predictions_tensor = model(X_test_tensor)

        wind_angles = config["training"]["angle_to_leave_out"]

        X_test_tensor_cpu = X_test_tensor.cpu()
        X_test_tensor_cpu = inverse_transform_features(X_test_tensor_cpu, feature_scaler)
        X_test_column_names = config["training"]["input_params_modf"]
        X_test_dataframe = pd.DataFrame(X_test_tensor_cpu, columns=X_test_column_names)

        y_test_tensor_cpu = y_test_tensor.cpu()
        y_test_tensor_cpu = inverse_transform_targets(y_test_tensor_cpu, target_scaler)
        output_column_names = config["training"]["output_params_modf"]
        y_test_column_names = [item + "_Actual" for item in output_column_names]
        y_test_dataframe = pd.DataFrame(y_test_tensor_cpu, columns=y_test_column_names)
        y_test_dataframe['Velocity_Magnitude_Actual'] = np.sqrt(y_test_dataframe['Velocity_X_Actual']**2 + 
                                                y_test_dataframe['Velocity_Y_Actual']**2 + 
                                                y_test_dataframe['Velocity_Z_Actual']**2)


        predictions_tensor_cpu = predictions_tensor.cpu()
        predictions_tensor_cpu = inverse_transform_targets(predictions_tensor_cpu, target_scaler)
        predictions_column_names = [item + "_Predicted" for item in output_column_names]
        predictions_dataframe = pd.DataFrame(predictions_tensor_cpu, columns=predictions_column_names)
        predictions_dataframe['Velocity_Magnitude_Predicted'] = np.sqrt(predictions_dataframe['Velocity_X_Predicted']**2 + 
                                                predictions_dataframe['Velocity_Y_Predicted']**2 + 
                                                predictions_dataframe['Velocity_Z_Predicted']**2)

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

def evaluate_model_new_angles(config, wind_angle, model, X_test_tensor, feature_scaler, target_scaler):
    model.eval()
    with torch.no_grad():
        predictions_tensor = model(X_test_tensor)

        X_test_tensor_cpu = X_test_tensor.cpu()
        X_test_tensor_cpu = inverse_transform_features(X_test_tensor_cpu, feature_scaler)
        X_test_column_names = config["training"]["input_params_modf"]
        X_test_dataframe = pd.DataFrame(X_test_tensor_cpu, columns=X_test_column_names)

        predictions_tensor_cpu = predictions_tensor.cpu()
        predictions_tensor_cpu = inverse_transform_targets(predictions_tensor_cpu, target_scaler)
        predictions_column_names = config["training"]["output_params_modf"]
        predictions_dataframe = pd.DataFrame(predictions_tensor_cpu, columns=predictions_column_names)
        predictions_dataframe['Velocity_Magnitude'] = np.sqrt(predictions_dataframe['Velocity_X']**2 + 
                                                predictions_dataframe['Velocity_Y']**2 + 
                                                predictions_dataframe['Velocity_Z']**2)

        combined_df = pd.concat([X_test_dataframe, predictions_dataframe], axis=1)
        combined_df['WindAngle'] = (wind_angle)

    return combined_df

def evaluate_model_training(device, model, model_file_path, X_test_tensor, y_test_tensor):
    checkpoint = torch.load(model_file_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    with torch.no_grad():
        predictions_tensor = model(X_test_tensor)

        # Move tensors to CPU for compatibility with sklearn
        predictions_tensor_cpu = predictions_tensor.cpu()
        y_test_tensor_cpu = y_test_tensor.cpu()

        mses = []
        r2s = []

        # Iterate over each column (variable) if they are multi-dimensional
        for i in range(predictions_tensor_cpu.shape[1]):  # Assuming the second dimension contains the variables
            predictions_flat = predictions_tensor_cpu[:, i].flatten()
            y_test_flat = y_test_tensor_cpu[:, i].flatten()

            # Calculate MSE for each variable
            mse = sklearn.metrics.mean_squared_error(y_test_flat, predictions_flat)
            r2 = sklearn.metrics.r2_score(y_test_flat, predictions_flat)

            mses.append(mse)
            r2s.append(r2)

    model.train()

    return mses, r2s

def evaluate_model_div(config, model, X_test_tensor, y_test_tensor, feature_scaler, target_scaler):
    model.eval()

    print ('lets do this')

    predictions_tensor = model(X_test_tensor)

    wind_angles = config["training"]["angle_to_leave_out"]

    X_test_tensor_cpu = X_test_tensor.cpu()
    X_test_tensor_cpu_inv = inverse_transform_features(X_test_tensor_cpu, feature_scaler)

    x_tensor, y_tensor, z_tensor = X_test_tensor_cpu_inv[:, 0], X_test_tensor_cpu_inv[:, 1], X_test_tensor_cpu_inv[:, 2]
    
    y_test_tensor_cpu = y_test_tensor.cpu()
    y_test_tensor_cpu_inv = inverse_transform_targets(y_test_tensor_cpu, target_scaler)

    vx, vy, vz = y_test_tensor_cpu_inv[:, 0], y_test_tensor_cpu_inv[:, 1], y_test_tensor_cpu_inv[:, 2]

    # Example data (replace these with your actual data)
    vx = np.array(vx)  # Replace with your actual data
    vy = np.array(vy)
    vz = np.array(vz)
    x = np.array(x_tensor)
    y = np.array(y_tensor)
    z = np.array(z_tensor)

    unique_x = np.unique(x)
    unique_y = np.unique(y)
    unique_z = np.unique(z)

    grid_dimensions = (len(unique_x), len(unique_y), len(unique_z))
    print("Grid dimensions:", grid_dimensions)


    vx = y_test_tensor_cpu_inv[:, 0].reshape(grid_dimensions[0], grid_dimensions[1], grid_dimensions[2])
    vy = y_test_tensor_cpu_inv[:, 1].reshape(grid_dimensions[0], grid_dimensions[1], grid_dimensions[2])
    vz = y_test_tensor_cpu_inv[:, 2].reshape(grid_dimensions[0], grid_dimensions[1], grid_dimensions[2])

    # Assuming uniform grid spacing
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]

    # Compute partial derivatives using central differences
    dvx_dx = (np.roll(vx, -1, axis=0) - np.roll(vx, 1, axis=0)) / (2 * dx)
    dvy_dy = (np.roll(vy, -1, axis=1) - np.roll(vy, 1, axis=1)) / (2 * dy)
    dvz_dz = (np.roll(vz, -1, axis=2) - np.roll(vz, 1, axis=2)) / (2 * dz)

    # Calculate divergence
    divergence = dvx_dx + dvy_dy + dvz_dz

    print (divergence)

    # # Handling the boundaries (optional, based on how you want to treat the edges)
    # divergence[0, :, :], divergence[-1, :, :] = 0, 0  # For x boundaries
    # divergence[:, 0, :], divergence[:, -1, :] = 0, 0  # For y boundaries
    # divergence[:, :, 0], divergence[:, :, -1] = 0, 0  # For z boundaries

    # # Convert x_tensor, y_tensor, z_tensor to PyTorch tensors and enable gradient tracking
    # x_tensor = torch.tensor(x_tensor, dtype=torch.float32, requires_grad=True)
    # y_tensor = torch.tensor(y_tensor, dtype=torch.float32, requires_grad=True)
    # z_tensor = torch.tensor(z_tensor, dtype=torch.float32, requires_grad=True)



    # vx, vy, vz = y_test_tensor_cpu_inv[:, 0], y_test_tensor_cpu_inv[:, 1], y_test_tensor_cpu_inv[:, 2]

    # # Ensure vx, vy, vz are also PyTorch tensors with requires_grad set to True
    # vx = torch.tensor(vx, dtype=torch.float32, requires_grad=True)
    # vy = torch.tensor(vy, dtype=torch.float32, requires_grad=True)
    # vz = torch.tensor(vz, dtype=torch.float32, requires_grad=True)

    # # Compute gradients for each velocity component
    # grad_vx_x = torch.autograd.grad(outputs=vx, inputs=x_tensor, grad_outputs=torch.ones_like(vx), only_inputs=True, retain_graph=True, allow_unused=True)[0]
    # grad_vy_y = torch.autograd.grad(outputs=vy, inputs=y_tensor, grad_outputs=torch.ones_like(vy), only_inputs=True, retain_graph=True, allow_unused=True)[0]
    # grad_vz_z = torch.autograd.grad(outputs=vz, inputs=z_tensor, grad_outputs=torch.ones_like(vz), only_inputs=True, retain_graph=True, allow_unused=True)[0]

    # # Compute the divergence as the sum of the partial derivatives
    # divergence_actual = grad_vx_x + grad_vy_y + grad_vz_z

    # vx_pred, vy_pred, vz_pred = predictions_tensor[:, 0], predictions_tensor[:, 1], predictions_tensor[:, 2]


    
    # X_test_column_names = config["training"]["input_params_modf"]
    # X_test_dataframe = pd.DataFrame(X_test_tensor_cpu, columns=X_test_column_names)

    # y_test_tensor_cpu = y_test_tensor.cpu()
    # y_test_tensor_cpu = inverse_transform_targets(y_test_tensor_cpu, target_scaler)
    # output_column_names = config["training"]["output_params_modf"]
    # y_test_column_names = [item + "_Actual" for item in output_column_names]
    # y_test_dataframe = pd.DataFrame(y_test_tensor_cpu, columns=y_test_column_names)
    # y_test_dataframe['Velocity_Magnitude_Actual'] = np.sqrt(y_test_dataframe['Velocity_X_Actual']**2 + 
    #                                         y_test_dataframe['Velocity_Y_Actual']**2 + 
    #                                         y_test_dataframe['Velocity_Z_Actual']**2)


    # predictions_tensor_cpu = predictions_tensor.cpu()
    # predictions_tensor_cpu = inverse_transform_targets(predictions_tensor_cpu, target_scaler)
    # predictions_column_names = [item + "_Predicted" for item in output_column_names]
    # predictions_dataframe = pd.DataFrame(predictions_tensor_cpu, columns=predictions_column_names)
    # predictions_dataframe['Velocity_Magnitude_Predicted'] = np.sqrt(predictions_dataframe['Velocity_X_Predicted']**2 + 
    #                                         predictions_dataframe['Velocity_Y_Predicted']**2 + 
    #                                         predictions_dataframe['Velocity_Z_Predicted']**2)

    #     rows_list = []
    #     for i, var in enumerate(y_test_column_names):
    #         var_cleaned = var.replace('_Actual', '')
    #         actuals = y_test_dataframe.iloc[:, i]
    #         preds = predictions_dataframe.iloc[:, i]

    #         mse = sklearn.metrics.mean_squared_error(actuals, preds)
    #         rmse = np.sqrt(mse)
    #         mae = sklearn.metrics.mean_absolute_error(actuals, preds)
    #         r2 = sklearn.metrics.r2_score(actuals, preds)
            
    #         # Append the new row as a dictionary to the list
    #         rows_list.append({
    #             'Variable': var_cleaned, 
    #             'MSE': mse,
    #             'RMSE': rmse,
    #             'MAE': mae,
    #             'R2': r2
    #         })
        
    #     data_folder = os.path.join(output_folder, f'data_output_for_wind_angle_all')
    #     os.makedirs(data_folder, exist_ok=True)

    #     combined_df = pd.concat([X_test_dataframe, y_test_dataframe, predictions_dataframe], axis=1)
    #     test_predictions.append([combined_df])
    #     combined_file_path = os.path.join(data_folder, f'combined_actuals_and_predictions_for_wind_angle_all.csv')
    #     combined_df.to_csv(combined_file_path, index=False)

    #     metrics_df = pd.DataFrame(rows_list)   
    #     metrics_file_path = os.path.join(data_folder, f'metrics_for_wind_angle_all.csv')
    #     metrics_df.to_csv(metrics_file_path, index=False)

    #     for wind_angle in wind_angles:
    #         lower_bound = wind_angle - 2
    #         upper_bound = wind_angle + 2

    #         X_test_dataframe['WindAngle_rad'] = np.arctan2(X_test_dataframe['sin(WindAngle)'], X_test_dataframe['cos(WindAngle)'])
    #         X_test_dataframe['WindAngle'] = X_test_dataframe['WindAngle_rad'].apply(lambda x: int(np.ceil(np.degrees(x))))
    #         mask = X_test_dataframe['WindAngle'].between(lower_bound, upper_bound)
    #         filtered_X_test_dataframe = X_test_dataframe.loc[mask]

    #         filtered_y_test = y_test_dataframe.loc[mask]
    #         filtered_predictions = predictions_dataframe.loc[mask]

    #         if len(filtered_predictions)!= 0 and len(filtered_y_test)!=0:
    #             rows_list = []
    #             for i, var in enumerate(y_test_column_names):
    #                 var_cleaned = var.replace('_Actual', '')
    #                 actuals = filtered_y_test.iloc[:, i]
    #                 preds = filtered_predictions.iloc[:, i]

    #                 mse = sklearn.metrics.mean_squared_error(actuals, preds)
    #                 rmse = np.sqrt(mse)
    #                 mae = sklearn.metrics.mean_absolute_error(actuals, preds)
    #                 r2 = sklearn.metrics.r2_score(actuals, preds)
                    
    #                 # Append the new row as a dictionary to the list
    #                 rows_list.append({
    #                     'Variable': var_cleaned, 
    #                     'MSE': mse,
    #                     'RMSE': rmse,
    #                     'MAE': mae,
    #                     'R2': r2
    #                 })
                
    #             data_folder = os.path.join(output_folder, f'data_output_for_wind_angle_{wind_angle}')
    #             os.makedirs(data_folder, exist_ok=True)

    #             combined_df = pd.concat([filtered_X_test_dataframe, filtered_y_test, filtered_predictions], axis=1)
    #             test_predictions_wind_angle.append([wind_angle, combined_df])
    #             combined_file_path = os.path.join(data_folder, f'combined_actuals_and_predictions_for_wind_angle_{wind_angle}.csv')
    #             combined_df.to_csv(combined_file_path, index=False)

    #             metrics_df = pd.DataFrame(rows_list)   
    #             metrics_file_path = os.path.join(data_folder, f'metrics_for_wind_angle_{wind_angle}.csv')
    #             metrics_df.to_csv(metrics_file_path, index=False)

    # return test_predictions, test_predictions_wind_angle