import os
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
import importlib.util
import sklearn
import collections
import random
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.utils.data import Sampler
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.tri as tri
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from scipy.interpolate import griddata
from scipy.stats import linregress
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from stl import mesh
from pathlib import Path
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkIOEnSight import vtkEnSightGoldBinaryReader
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridWriter
from scipy.spatial import cKDTree

def generate_machine_paths(dataset_name):
    return {
        "mac": os.path.join(os.path.expanduser("~"), "Dropbox", "School", "Graduate", "CERN", "Temp_Files", "nonlineardynamics", "ameir_PINNs", dataset_name),
        "CREATE": os.path.join('Z:\\', dataset_name),
        "google": f"/content/drive/Othercomputers/My Mac mini/{dataset_name}",
    }

class Logger(object):
    def __init__(self, filename='Default.log'):
        self.terminal = sys.stdout  # Save the original stdout
        self.log = open(filename, 'a')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass

class WindAngleDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class BalancedWindAngleSampler(Sampler):
    def __init__(self, dataset, wind_angles):
        self.dataset = dataset
        self.wind_angles = wind_angles
        self.indices = self._create_indices()
    def _create_indices(self):
        angle_indices = {angle: np.where(self.dataset.labels == angle)[0] for angle in self.wind_angles}
        balanced_indices = []
        while True:
            batch = []
            for angle in self.wind_angles:
                batch.extend(np.random.choice(angle_indices[angle], size=1))
            balanced_indices.extend(batch)
            if len(balanced_indices) > len(self.dataset):
                break
        return balanced_indices
    def __iter__(self):
        return iter(self.indices[:len(self.dataset)])
    def __len__(self):
        return len(self.dataset)

def get_time_elapsed(start_time, time_passed=None):
    if time_passed is None:
        return (time.time() - start_time)
    else:
        return (time.time() - start_time + time_passed)

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

def get_filenames_from_folder(path, extension, startname):
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith(extension) and f.startswith(startname)]

def print_statement(epoch, epochs, use_epoch, current_loss, loss_dict, free_memory, current_elapsed_time_hours):
    print(f'Epoch [{epoch}/{epochs if use_epoch else "infinity"}], '
          f'Loss: {current_loss}, '
          f'Total Time Elapsed: {current_elapsed_time_hours:.2f} hours, '
          f'with free memory: {free_memory:.2f} GB; '
          f'data_loss = {loss_dict.get("data_loss", 0)}, '
          f'cont_loss = {loss_dict.get("cont_loss", 0)}, '
          f'momentum_loss = {loss_dict.get("momentum_loss", 0)}, '
          f'no_slip_loss = {loss_dict.get("no_slip_loss", 0)}, '
          f'inlet_loss = {loss_dict.get("inlet_loss", 0)}, '
          f'total_loss = {loss_dict["total_loss"]}, '
          f'total_loss_weighted = {loss_dict.get("total_loss_weighted", 0)} - '
          f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

def save_to_csv(epoch, epochs, use_epoch, current_elapsed_time_hours, free_memory, loss_dict, file_path):
    data = {
        'Epoch': f'{epoch}',
        'Total Time Elapsed (hours)': f'{current_elapsed_time_hours:.2f}',
        'Free Memory (GB)': f'{free_memory:.2f}',
        'Data Loss': loss_dict.get('data_loss', 0),
        'Continuity Loss': loss_dict.get('cont_loss', 0),
        'Momentum Loss': loss_dict.get('momentum_loss', 0),
        'Averaged No Slip Loss': loss_dict.get('no_slip_loss', 0),
        'Averaged Inlet Loss': loss_dict.get('inlet_loss', 0),
        'Total Loss': loss_dict.get('total_loss', 0),
        'Total Loss Weighted': loss_dict.get('total_loss_weighted', 0)
    }
    for key in data:
        if key != 'Epoch' and key != 'Total Time Elapsed (hours)' and key != 'Free Memory (GB)' and data[key] != 0:
            data[key] = data[key].cpu().detach().numpy()
    file_exists = os.path.isfile(file_path) and os.path.getsize(file_path) > 0
    with open(file_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

def save_evaluation_to_csv(epoch, epochs, use_epoch, output_params, mses, r2s, file_path):
    headers = ['Epoch'] + [f'MSE_{param}' for param in output_params] + [f'R2_{param}' for param in output_params]
    epoch_label = f'{epoch}'
    data = [epoch_label] + mses + r2s
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(data)

def save_losses_to_json(recent_losses, sma, consecutive_sma_count, current_elapsed_time, file_path):
    recent_losses_list = [loss.tolist() if isinstance(loss, np.ndarray) else loss for loss in recent_losses]
    data = {
        'recent_losses': recent_losses_list,
        'sma': sma.tolist() if isinstance(sma, np.ndarray) else sma,
        'consecutive_sma_count': consecutive_sma_count,
        'current_elapsed_time': current_elapsed_time
    }
    with open(file_path, 'w') as file:
        json.dump(data, file)

def extract_input_parameters(X, input_params):
    extracted_params = []
    for param in input_params:
        if param == 'Points:0':
            extracted_params.append(X[:, 0:1])
        elif param == 'Points:1':
            extracted_params.append(X[:, 1:2])
        elif param == 'Points:2':
            extracted_params.append(X[:, 2:3])
        elif param == 'cos(WindAngle)':
            extracted_params.append(X[:, 3:4])
        elif param == 'sin(WindAngle)':
            extracted_params.append(X[:, 4:5])
        else:
            raise ValueError(f"Unknown parameter: {param}")
    return extracted_params

def extract_output_parameters(Y, output_params):
    extracted_output_params = []
    for param in output_params:
        if param == 'Pressure':
            extracted_output_params.append(Y[:, 0:1])
        elif param == 'Velocity:0':
            extracted_output_params.append(Y[:, 1:2])
        elif param == 'Velocity:1':
            extracted_output_params.append(Y[:, 2:3])
        elif param == 'Velocity:2':
            extracted_output_params.append(Y[:, 3:4])
        elif param == 'TurbVisc':
            extracted_output_params.append(Y[:, 4:5])
        else:
            raise ValueError(f"Unknown output parameter: {param}")
    return extracted_output_params

def load_losses_from_json(file_path, max_length):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            recent_losses = collections.deque(data.get('recent_losses', []), maxlen=max_length)
            sma = data.get('sma', None)
            consecutive_sma_count = data.get('consecutive_sma_count', [])
            current_elapsed_time = data.get('current_elapsed_time', None)
            return recent_losses, sma, consecutive_sma_count, current_elapsed_time
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return collections.deque(maxlen=max_length), 0, 0,  None
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {file_path}")
        return collections.deque(maxlen=max_length), 0, 0,  None

def filter_info_file(directory, numtoplot=None):
    df = pd.read_csv(os.path.join(directory, 'info.csv'))    
    time = df['Total Time Elapsed (hours)'].iloc[-1]
    total_epochs = df['Epoch'].iloc[-1]
    df = df.loc[:, (df != 0).any(axis=0)]
    if numtoplot is not None:
        numtoskip = int(len(df)/numtoplot)
        df = df[(df['Epoch'] - 5) % numtoskip == 0]
    return df, time, total_epochs

def filter_trainingloss_file(directory, filename, numtoplot=None):
    df = pd.read_csv(os.path.join(directory, filename))
    if numtoplot is not None:
        numtoskip = int(len(df)/numtoplot)
        df = df[(df['Epoch'] - 5) % numtoskip == 0]
    return df

def restructure_data(faces_data):
    return [tuple(faces_data[i:i+3]) for i in range(0, len(faces_data), 3)]

def compute_normal(v1, v2, v3):
    v1, v2, v3 = np.array(v1), np.array(v2), np.array(v3)
    edge1 = v2 - v1
    edge2 = v3 - v1
    normal = np.cross(edge1, edge2)
    return normal / np.linalg.norm(normal)

def get_optimizer(model, chosen_optimizer_key, optimizer_config):
    if chosen_optimizer_key == "lbfgs_optimizer":
        optimizer = torch.optim.LBFGS(model.parameters(), lr=optimizer_config["learning_rate"], 
            max_iter=optimizer_config["max_iter"], 
            max_eval=optimizer_config["max_eval"], 
            tolerance_grad=optimizer_config["tolerance_grad"], 
            tolerance_change=optimizer_config["tolerance_change"], 
            history_size=optimizer_config["history_size"], 
            line_search_fn=optimizer_config["line_search_fn"])
    elif chosen_optimizer_key == "adam_optimizer":
        optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_config["learning_rate"])
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    return optimizer

def initialize_and_fit_scalers(features, targets, config):
    feature_scaler = config["training"]["feature_scaler"]
    target_scaler = config["training"]["target_scaler"]
    feature_scaler.fit(features)
    target_scaler.fit(targets)
    return feature_scaler, target_scaler

def open_model_file(model_file_path, device):
    checkpoint = torch.load(model_file_path, map_location=device)
    return checkpoint

def transform_data_with_scalers(features, targets, feature_scaler, target_scaler):
    normalized_features = feature_scaler.transform(features)
    normalized_targets = target_scaler.transform(targets)
    return normalized_features, normalized_targets

def inverse_transform_features(features_normalized, feature_scaler):
    return feature_scaler.inverse_transform(features_normalized)

def inverse_transform_targets(targets_normalized, target_scaler):
    return target_scaler.inverse_transform(targets_normalized)

# def evaluate_model_training_PCA(config, device, model, model_file_path, X_test_tensor, y_test_tensor, pca, original_shape):
#     checkpoint = torch.load(model_file_path, map_location=device)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.eval()
#     with torch.no_grad():
#         predictions_tensor = model(X_test_tensor)
#         predictions_tensor_cpu = predictions_tensor.cpu()
#         y_test_tensor_cpu = y_test_tensor.cpu()
#         X_test_tensor_cpu = X_test_tensor.cpu()
#         features_final, targets_final, targets_pred_final = invert_data_PCA(config, X_test_tensor_cpu, y_test_tensor_cpu, predictions_tensor_cpu, pca, original_shape)

#         mses = []
#         r2s = []

#         for i in range(targets_pred_final.shape[1]):
#             predictions_flat = targets_pred_final[:, i].flatten()
#             y_test_flat = targets_final[:, i].flatten()
#             mse = sklearn.metrics.mean_squared_error(y_test_flat, predictions_flat)
#             r2 = sklearn.metrics.r2_score(y_test_flat, predictions_flat)
#             mses.append(mse)
#             r2s.append(r2)
#     model.train()
#     return mses, r2s

def evaluate_model_training_PCA(config, device, model, model_file_path, X_test_tensor, y_test_tensor, pca_rows, pca_columns, original_shape):
    checkpoint = torch.load(model_file_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    with torch.no_grad():
        predictions_tensor = model(X_test_tensor)
        predictions_tensor_cpu = predictions_tensor.cpu()
        y_test_tensor_cpu = y_test_tensor.cpu()
        X_test_tensor_cpu = X_test_tensor.cpu()
        features_final, targets_final, targets_pred_final = invert_data_PCA(config, X_test_tensor_cpu, y_test_tensor_cpu, predictions_tensor_cpu, pca_rows, pca_columns, original_shape)

        mses = []
        r2s = []

        for i in range(targets_pred_final.shape[1]):
            predictions_flat = targets_pred_final[:, i].flatten()
            y_test_flat = targets_final[:, i].flatten()
            mse = sklearn.metrics.mean_squared_error(y_test_flat, predictions_flat)
            r2 = sklearn.metrics.r2_score(y_test_flat, predictions_flat)
            mses.append(mse)
            r2s.append(r2)
    model.train()
    return mses, r2s

def evaluate_model_training(device, model, model_file_path, X_test_tensor, y_test_tensor):
    checkpoint = torch.load(model_file_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    with torch.no_grad():
        predictions_tensor = model(X_test_tensor)
        predictions_tensor_cpu = predictions_tensor.cpu()
        y_test_tensor_cpu = y_test_tensor.cpu()
        mses = []
        r2s = []
        for i in range(predictions_tensor_cpu.shape[1]):
            predictions_flat = predictions_tensor_cpu[:, i].flatten()
            y_test_flat = y_test_tensor_cpu[:, i].flatten()
            mse = sklearn.metrics.mean_squared_error(y_test_flat, predictions_flat)
            r2 = sklearn.metrics.r2_score(y_test_flat, predictions_flat)
            mses.append(mse)
            r2s.append(r2)
    model.train()
    return mses, r2s

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

def load_plotting_data(config, wind_angles=None):
    chosen_machine_key = config["chosen_machine"]
    datafolder_path = os.path.join(config["machine"][chosen_machine_key], config["data"]["data_folder_name"])
    filenames = get_filenames_from_folder(datafolder_path, config["data"]["extension"], config["data"]["startname_data"])
    if wind_angles is None:
        data = concatenate_data_files(filenames, datafolder_path)
    else:
        data = concatenate_data_files(filenames, datafolder_path, wind_angles)
    data.rename(columns={'Points:0': 'X', 'Points:1': 'Y', 'Points:2': 'Z', 'Velocity:0': 'Velocity_X', 'Velocity:1': 'Velocity_Y', 'Velocity:2': 'Velocity_Z'}, inplace=True)
    data['Velocity_Magnitude'] = np.sqrt(data['Velocity_X']**2 + data['Velocity_Y']**2 + data['Velocity_Z']**2)
    return data

def load_derivative_data(config, wind_angles):
    chosen_machine_key = config["chosen_machine"]
    datafolder_path = os.path.join(config["machine"][chosen_machine_key], config["data"]["data_folder_name"])
    filenames = get_filenames_from_folder(datafolder_path, config["data"]["extension"], "all_data")
    data = concatenate_data_files(filenames, datafolder_path, wind_angles)
    data.rename(columns={'Points:0': 'X', 'Points:1': 'Y', 'Points:2': 'Z', 
    'GradientVelocity:0': 'u_x', 'GradientVelocity:1': 'u_y', 'GradientVelocity:2': 'u_z', 
    'GradientVelocity:3': 'v_x', 'GradientVelocity:4': 'v_y', 'GradientVelocity:5': 'v_z', 
    'GradientVelocity:6': 'w_x', 'GradientVelocity:7': 'w_y', 'GradientVelocity:8': 'w_z',
    'SecondGradientVelocity:0': 'u_xx','SecondGradientVelocity:1': 'u_xy','SecondGradientVelocity:2': 'u_xz',
    'SecondGradientVelocity:3': 'u_yx','SecondGradientVelocity:4': 'u_yy','SecondGradientVelocity:5': 'u_yz',
    'SecondGradientVelocity:6': 'u_zx','SecondGradientVelocity:7': 'u_zy','SecondGradientVelocity:8': 'u_zz',
    'SecondGradientVelocity:9': 'v_xx','SecondGradientVelocity:10': 'v_xy','SecondGradientVelocity:11': 'v_xz',
    'SecondGradientVelocity:12': 'v_yx','SecondGradientVelocity:13': 'v_yy','SecondGradientVelocity:14': 'v_yz',
    'SecondGradientVelocity:15': 'v_zx','SecondGradientVelocity:16': 'v_zy','SecondGradientVelocity:17': 'v_zz',
    'SecondGradientVelocity:18': 'w_xx','SecondGradientVelocity:19': 'w_xy','SecondGradientVelocity:20': 'w_xz',
    'SecondGradientVelocity:21': 'w_yx','SecondGradientVelocity:22': 'w_yy','SecondGradientVelocity:23': 'w_yz',
    'SecondGradientVelocity:24': 'w_zx','SecondGradientVelocity:25': 'w_zy','SecondGradientVelocity:26': 'w_zz',
    'Velocity:0': 'u', 'Velocity:1': 'v', 'Velocity:2': 'w', 
    'GradientPressure:0': 'p_x', 'GradientPressure:1': 'p_y', 'GradientPressure:2': 'p_z',
    },inplace=True)
    return data

def extract_stds_means(scaler, params):
    stds = scaler.scale_  
    means = scaler.mean_  
    stds_dict = {param + "_std": std for param, std in zip(params, stds)}
    means_dict = {param + "_mean": mean for param, mean in zip(params, means)}
    stds_means_dict = {**stds_dict, **means_dict}
    return stds_means_dict

# def pad_with_zeros(array, pad_width, mode='constant', **kwargs):
#     """
#     Pads a NumPy array with zeros on all sides according to the specified pad_width.
    
#     Parameters:
#     - array: np.array, the original array to be padded.
#     - pad_width: int or sequence of ints, specifying the number of elements to pad 
#       on each side. For a 1D array, this can be a single integer or a tuple of two 
#       integers. For a 2D array, it can be a tuple of two tuples, where each inner 
#       tuple has two integers (before, after) for each dimension (row, column).
#     - mode: str, the mode used for padding. Default is 'constant'.
#     - **kwargs: additional keyword arguments passed to np.pad. For mode='constant',
#       constant_values determines the values to set the pad elements. Default is 0.
    
#     Returns:
#     - Padded array: np.array, the array after padding with zeros.
#     """
    
#     if 'constant_values' not in kwargs:
#         kwargs['constant_values'] = 0
    
#     return np.pad(array, pad_width=pad_width, mode=mode, **kwargs)

# def drop_array_indices(data, exclude_indices):
#     col_indices = np.arange(data.shape[1])
#     keep_indices = np.isin(col_indices, exclude_indices, invert=True)
#     sub_data = data[:, keep_indices]
#     return sub_data

# def extend_array_PCA(array, required_length):
#     current_length = array.shape[0]
#     if current_length == required_length:
#         return array
#     repeat_factor = -(-required_length // current_length)  # Ceiling division
#     extended_array = np.tile(array, (repeat_factor, 1))
#     if extended_array.shape[0] > required_length:
#         extended_array = extended_array[:required_length, :]
#     return extended_array

# def initialize_and_fit_PCA(array, n_components):
#     pca = PCA(n_components=n_components)
#     pca.fit(array)
#     explained_variance_ratio = pca.explained_variance_ratio_
#     print(f"Explained variance ratio of the first {len(explained_variance_ratio)} principal components:", explained_variance_ratio)
#     print(f"Explained variance sum of the first {len(explained_variance_ratio)} principal components:", np.sum(explained_variance_ratio))
#     return pca

# def transform_data_with_PCA(array, pca):
#     array = pca.transform(array)
#     return array

# def compute_PCA(config, df, wind_angles, input_params, output_params, pca, required_length, angle_labels=None):
#     dfs = []
#     labels = []
#     for angle in wind_angles:
#         data = df[df[:, 5] == float(angle)]
#         exclude_indices = [3, 4, 5]
#         sub_data = drop_array_indices(data, exclude_indices)
#         # sub_data = extend_array_PCA(sub_data, required_length)
#         sub_data = sub_data.T
#         sub_data = transform_data_with_PCA(sub_data, pca)
#         sub_data = sub_data.T
#         columns = input_params+output_params
#         sub_data = pd.DataFrame(sub_data, columns=columns)
#         sub_data['cos(WindAngle)'] = data[:, exclude_indices][0][0]
#         sub_data['sin(WindAngle)'] = data[:, exclude_indices][0][1]
#         if angle_labels is not None:
#             label = angle_labels[angle]
#             labels.extend([label] * len(sub_data))
#         dfs.append(sub_data)
#     all_data = pd.concat(dfs, ignore_index=True)
#     if angle_labels is not None:
#         return all_data, labels
#     else:
#         return all_data

# def compute_PCA_inverse(config, X, y, pca, wind_angles, input_params_points, input_params, output_params, device):
#     dfs = []
#     X = X.cpu()
#     y = y.cpu()
#     X = np.array(X)
#     y = np.array(y)
#     exclude_indices = [3, 4]
#     for i in wind_angles:
#         # matching_rows = np.all(X[:, exclude_indices] == i, axis=1)
#         tolerance = 1e-3
#         matching_rows = np.all(np.isclose(X[:, exclude_indices], i, atol=tolerance), axis=1)
#         X_ = X[matching_rows, :]
#         y_ = y[matching_rows, :]
#         sub_X = drop_array_indices(X_, exclude_indices)
#         sub_data = np.concatenate([sub_X, y_], axis=1)
#         data = np.concatenate([X_, y_], axis=1)
#         sub_data = pca.inverse_transform(sub_data.T)
#         sub_data = sub_data.T
#         columns = input_params_points+output_params
#         sub_data = pd.DataFrame(sub_data, columns=columns)
#         sub_data['cos(WindAngle)'] = data[:, exclude_indices][0][0]
#         sub_data['sin(WindAngle)'] = data[:, exclude_indices][0][1]
#         dfs.append(sub_data)
#     all_data = pd.concat(dfs, ignore_index=True)
#     X_new = all_data[input_params]
#     y_new = all_data[output_params]
#     X_new = torch.tensor(np.array(X_new), dtype=torch.float32)
#     y_new = torch.tensor(np.array(y_new), dtype=torch.float32)
#     X_new = X_new.to(device)
#     y_new = y_new.to(device)
#     return all_data, X_new, y_new

# def fit_PCA(normalized_features, normalized_targets, exclude_indices, n_components):
#     concatenated_normalized = (np.concatenate([normalized_features, normalized_targets], axis=1))
#     sub_concatenated_normalized = drop_array_indices(concatenated_normalized, exclude_indices)
#     required_length = sub_concatenated_normalized.shape[0]
#     pca = initialize_and_fit_PCA(sub_concatenated_normalized.T, n_components)
#     return pca, required_length

# def compute_mse_PCA(orig_data, data_inv, indices):
#     data_inv = np.array(data_inv)
#     orig_data = np.array(orig_data)
#     indices1 = np.lexsort((data_inv[:, indices[1]], data_inv[:, indices[0]]))
#     data_inv = data_inv[indices1]
#     indices2 = np.lexsort((orig_data[:, indices[1]], orig_data[:, indices[0]]))
#     orig_data = orig_data[indices2]
#     squared_diff = (np.array(orig_data) - np.array(data_inv)) ** 2
#     mse = list(np.mean(squared_diff, axis=0))
#     print("MSE column-wise:", mse)
#     return mse

# def invert_data_PCA(config, features, targets, predictions, pca, original_shape):

#     output_params = config["training"]["output_params"]
#     input_params = config["training"]["input_params"]

#     def get_features_targets(X_pca, n, m):
#             first_n_cols = []
#             last_m_cols = []
#             for i in range(0, X_pca.shape[1], n+m):
#                 first_n = X_pca[:, i:i+n]
#                 last_m = X_pca[:, i+n:i+(n+m)]
#                 first_n_cols.append(first_n)
#                 last_m_cols.append(last_m)
#             array_first_n = np.concatenate(first_n_cols, axis=1)
#             array_last_m = np.concatenate(last_m_cols, axis=1)
#             return array_first_n, array_last_m

#     def invert_features_targets(array_first_n, array_last_m, n_, m):
#             n = array_first_n.shape[1] // n_
#             reconstructed_cols = []
#             for i in range(n):
#                 cols_first_n = array_first_n[:, i*n_:(i+1)*n_]
#                 cols_last_m = array_last_m[:, i*m:(i+1)*m]
#                 for col in cols_first_n.T:
#                     reconstructed_cols.append(col)
#                 for col in cols_last_m.T:
#                     reconstructed_cols.append(col)
#             original_array_reconstructed = np.column_stack(reconstructed_cols)
#             return original_array_reconstructed

#     def invert_pca(pca, X_pca, original_shape):
#             X_inverted = (pca.inverse_transform(X_pca.T)).T 
#             X_3d_reconstructed = X_inverted.reshape(original_shape)

#             # print("Shape after PCA and inverse transformation:", X_inverted.shape)
#             # print("Reconstructed 3D shape:", X_3d_reconstructed.shape)

#             return X_inverted, X_3d_reconstructed

#     X_pca = invert_features_targets(features, targets, len(input_params), len(output_params))
#     X_pca_pred = invert_features_targets(features, predictions, len(input_params), len(output_params))

#     X_inverted, X_3d_reconstructed = invert_pca(pca, X_pca, original_shape)
#     X_inverted_pred, X_3d_reconstructed_pred = invert_pca(pca, X_pca_pred, original_shape)

#     mse = mean_squared_error(X_inverted, X_inverted_pred)
#     mse_3d = mean_squared_error(X_3d_reconstructed.flatten(), X_3d_reconstructed_pred.flatten())

#     print ("MSE inverse transformation:", mse, "MSE reconstructed 3D:", mse_3d)

#     features, targets = get_features_targets(X_inverted, len(input_params), len(output_params))
#     features_pred, targets_pred = get_features_targets(X_inverted_pred, len(input_params), len(output_params))

#     targets_reshaped = targets.reshape(targets.shape[0], int(targets.shape[1]/len(output_params)), len(output_params))
#     targets_final = targets_reshaped.reshape(targets.shape[0]*(int(targets.shape[1]/len(output_params))), len(output_params))

#     targets_pred_reshaped = targets_pred.reshape(targets_pred.shape[0], int(targets_pred.shape[1]/len(output_params)), len(output_params))
#     targets_pred_final = targets_pred_reshaped.reshape(targets_pred.shape[0]*(int(targets_pred.shape[1]/len(output_params))), len(output_params))

#     features_reshaped = features.reshape(features.shape[0], int(features.shape[1]/len(input_params)), len(input_params))
#     features_final = features_reshaped.reshape(features.shape[0]*(int(features.shape[1]/len(input_params))), len(input_params))

#     return features_final, targets_final, targets_pred_final

def invert_data_PCA(config, features, targets, predictions, pca_rows, pca_columns, original_shape):

    output_params = config["training"]["output_params"]
    input_params = config["training"]["input_params"]

    def get_features_targets(X_pca, n, m):
            first_n_cols = []
            last_m_cols = []
            for i in range(0, X_pca.shape[1], n+m):
                first_n = X_pca[:, i:i+n]
                last_m = X_pca[:, i+n:i+(n+m)]
                first_n_cols.append(first_n)
                last_m_cols.append(last_m)
            array_first_n = np.concatenate(first_n_cols, axis=1)
            array_last_m = np.concatenate(last_m_cols, axis=1)
            return array_first_n, array_last_m

    def invert_features_targets(array_first_n, array_last_m, n_, m):
            n = array_first_n.shape[1] // n_
            reconstructed_cols = []
            for i in range(n):
                cols_first_n = array_first_n[:, i*n_:(i+1)*n_]
                cols_last_m = array_last_m[:, i*m:(i+1)*m]
                for col in cols_first_n.T:
                    reconstructed_cols.append(col)
                for col in cols_last_m.T:
                    reconstructed_cols.append(col)
            original_array_reconstructed = np.column_stack(reconstructed_cols)
            return original_array_reconstructed

    def invert_pca(pca_rows, pca_columns, X_pca):
        X_inverted = (pca_rows.inverse_transform((pca_columns.inverse_transform(X_pca)).T)).T
        # print("Shape after PCA and inverse transformation:", X_inverted.shape)

        return X_inverted

    X_pca = invert_features_targets(features, targets, len(input_params), len(output_params))
    X_pca_pred = invert_features_targets(features, predictions, len(input_params), len(output_params))

    X_inverted = invert_pca(pca_rows, pca_columns, X_pca)
    X_inverted_pred = invert_pca(pca_rows, pca_columns, X_pca_pred)

    mse = mean_squared_error(X_inverted, X_inverted_pred)

    print ("MSE inverse transformation:", mse)

    features, targets = get_features_targets(X_inverted, len(input_params), len(output_params))
    features_pred, targets_pred = get_features_targets(X_inverted_pred, len(input_params), len(output_params))

    targets_reshaped = targets.reshape(targets.shape[0], int(targets.shape[1]/len(output_params)), len(output_params))
    targets_final = targets_reshaped.reshape(targets.shape[0]*(int(targets.shape[1]/len(output_params))), len(output_params))

    targets_pred_reshaped = targets_pred.reshape(targets_pred.shape[0], int(targets_pred.shape[1]/len(output_params)), len(output_params))
    targets_pred_final = targets_pred_reshaped.reshape(targets_pred.shape[0]*(int(targets_pred.shape[1]/len(output_params))), len(output_params))

    features_reshaped = features.reshape(features.shape[0], int(features.shape[1]/len(input_params)), len(input_params))
    features_final = features_reshaped.reshape(features.shape[0]*(int(features.shape[1]/len(input_params))), len(input_params))

    return features_final, targets_final, targets_pred_final

config = {
    "lbfgs_optimizer": {
        "type": "LBFGS",
        "learning_rate": 0.00001,
        "max_iter": 200000,
        "max_eval": 50000,
        "history_size": 50,
        "tolerance_grad": 1e-05,
        "tolerance_change": 0.5 * np.finfo(float).eps,
        "line_search_fn": "strong_wolfe"
    },
    "adam_optimizer": {
        "type": "Adam",
        "learning_rate": 0.001,
    },
    "training": {
        "number_of_hidden_layers": 4,
        "neuron_number": 128,
        "input_params": ['Points:0', 'Points:1', 'Points:2', 'cos(WindAngle)', 'sin(WindAngle)'], 
        "input_params_modf": ["X", "Y", "Z", "cos(WindAngle)", "sin(WindAngle)"], 
        "output_params": ['Velocity:0', 'Velocity:1', 'Velocity:2'], #['Pressure', 'Velocity:0', 'Velocity:1', 'Velocity:2', 'TurbVisc']
        "output_params_modf": ['Velocity_X', 'Velocity_Y', 'Velocity_Z'], #['Pressure', 'Velocity_X', 'Velocity_Y', 'Velocity_Z', 'TurbVisc']
        "activation_function": nn.ELU, #nn.ReLU, nn.Tanh,
        "batch_normalization": False,
        "dropout_rate": None,
        "use_epochs": False,
        "num_epochs": 1,
        "print_epochs": 10,
        "use_batches": False,
        "batch_size": 2**15,
        "use_data_for_cont_loss": False,
        "angles_to_leave_out": [135],
        "angles_to_train": [0,15,30,45,60,75,90,105,120,150,165,180],
        "all_angles": [0,15,30,45,60,75,90,105,120,135,150,165,180],
        "boundary": [[0,1000,0,1000,0,1000],100], #[[-2520,2520,-2520,2520,0,1000],100]
        "loss_diff_threshold": 1e-5,
        "consecutive_count_threshold": 10,
        "feature_scaler": sklearn.preprocessing.StandardScaler(), #sklearn.preprocessing.MinMaxScaler(feature_range=(-1,1))
        "target_scaler": sklearn.preprocessing.StandardScaler(), #sklearn.preprocessing.MinMaxScaler(feature_range=(-1,1))
    },
    "machine": {
        "mac": os.path.join(os.path.expanduser("~"), "Dropbox", "School", "Graduate", "CERN", "Temp_Files", "nonlineardynamics", "ameir_PINNs", "cylinder_cell"),
        "CREATE": os.path.join('Z:\\', "cylinder_cell"),
        "google": f"/content/drive/Othercomputers/MacMini/cylinder_cell",
    },
    "data": {
        "density": 1,
        "kinematic_viscosity": 1e-5,
        "data_folder_name": 'data',
        "extension": '.csv',
        "startname_data": 'CFD',
        "startname_meteo": 'meteo_',
        "output_zip_file": 'output.zip',
        "geometry": 'scaled_cylinder_sphere.stl' #'ladefense.stl'
    },
    "loss_components": {
        "data_loss": False,
        "cont_loss": False,
        "momentum_loss": False,
        "no_slip_loss":False, 
        "inlet_loss": False,
        "use_weighting": False,
        "weighting_scheme": 'adaptive_weighting', #'gradient_magnitude'
        "adaptive_weighting_initial_weight": 0.9,
        "adaptive_weighting_final_weight": 0.1,
    },
    "train_test": {
        "train": False,
        "test": False,
        "boundary_test": False,
        "evaluate": False,
        "evaluate_new_angles": False,
        "test_size": 0.1,
        "data_loss_test_size": None, #10000 
        "new_angles_test_size": 0.99999,
        "random_state": 42,
        "new_angles": [0,15,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,75,90,105,120,135,150,165,180]
    },
    "plotting": {
        "lim_min_max": [(-1, 1),(-1, 1),(0, 1)], #[(-0.3, 0.3),(-0.3, 0.3),(0, 1)]
        "arrow": [False, [[500,500],[500,570]]],
        "plotting_params": [['X-Y',50,5]], # [['X-Z',570,5],['Y-Z',500,5],['X-Y',50,5]] # ['X-Y',5,5] # [['X-Z',-300,5],['Y-Z',-300,5],['X-Y',5,5]]
        "plot_geometry": False,
        "make_logging_plots": False,
        "save_csv_predictions": False,
        "make_plots": False,
        "make_data_plots": False,
        "make_div_plots": False,
        "make_RANS_plots": False,
    },
    "chosen_machine": "mac",
    "chosen_optimizer": "adam_optimizer",
    "base_folder_names": "test",
    "base_folder_names": ["test"]
}

# def load_data_PCA(config, device):
#     chosen_machine_key = config["chosen_machine"]
#     datafolder_path = os.path.join(config["machine"][chosen_machine_key], config["data"]["data_folder_name"])
#     filenames = get_filenames_from_folder(datafolder_path, config["data"]["extension"], config["data"]["startname_data"])
#     training_wind_angles = config["training"]["angles_to_train"]
#     skipped_wind_angles = config["training"]["angles_to_leave_out"]
#     output_params = config["training"]["output_params"]
#     input_params = config["training"]["input_params"]
#     input_params_points = ['Points:0', 'Points:1', 'Points:2']
#     angle_to_label = {angle: idx for idx, angle in enumerate(sorted(training_wind_angles))}

#     def get_train_test_tensor(filenames, datafolder_path, training_wind_angles, angle_to_label):

#         print ("LOADING TRAINING DATASET")

#         def get_array(filenames, datafolder_path, training_wind_angles, angle_to_label):
#             all_data, labels = concatenate_data_files(filenames, datafolder_path, training_wind_angles, angle_to_label)
#             all_features = all_data[input_params]
#             all_targets = all_data[output_params]
#             feature_scaler, target_scaler = initialize_and_fit_scalers(all_features, all_targets, config)
#             normalized_features, normalized_targets = transform_data_with_scalers(all_features, all_targets, feature_scaler, target_scaler)

#             unique_labels = np.unique(np.array(labels))
#             n = len(unique_labels)  # Number of unique labels
#             m = normalized_features.shape[0] // n  # Number of data points per label
#             p_combined = normalized_features.shape[1] + normalized_targets.shape[1]  # Combined number of features and targets

#             label_column_indices = {}
#             for i, label in enumerate(unique_labels):
#                 start_index = i * p_combined
#                 end_index = start_index + p_combined
#                 label_column_indices[label] = (start_index, end_index)

#             # label_column_indices now maps each label to its column range in X_flattened
#             # specific_label = unique_labels[0]  # Example: Getting the first label
#             # start_index, end_index = label_column_indices[specific_label]
#             # # You can now use start_index and end_index to slice X_flattened for operations specific to this label
#             # specific_label_data = X_flattened[:, start_index:end_index]

#             combined_3d = np.zeros((n, m, p_combined))
#             for i, label in enumerate(unique_labels):
#                 indices = np.where(np.array(labels) == label)[0]
#                 combined = np.hstack((normalized_features[indices, :], normalized_targets[indices, :]))
#                 combined_3d[i, :, :] = combined
#             X_flattened = combined_3d.reshape(combined_3d.shape[1], -1)
            
#             return combined_3d, X_flattened, label_column_indices, feature_scaler, target_scaler

#         def apply_PCA(X_3d, X_flattened, n_components=None):
#             pca = PCA(n_components)
#             X_pca = (pca.fit_transform(X_flattened.T)).T

#             explained_variance_ratio = pca.explained_variance_ratio_
#             print(f"Explained variance ratio of the first {len(explained_variance_ratio)} principal components:", explained_variance_ratio)
#             print(f"Explained variance sum of the first {len(explained_variance_ratio)} principal components:", np.sum(explained_variance_ratio))

#             print("Original shape:", X_3d.shape)
#             print("Flattened shape:", X_flattened.shape)
#             print("After PCA shape:", X_pca.shape)

#             return pca, X_pca

#         def invert_pca(pca, X_pca):
#             X_inverted = (pca.inverse_transform(X_pca.T)).T
#             original_shape = X_3d.shape 
#             X_3d_reconstructed = X_inverted.reshape(original_shape)
#             mse = mean_squared_error(X_flattened, X_inverted)
#             mse_3d = mean_squared_error(X_3d.flatten(), X_3d_reconstructed.flatten())

#             print("Shape after PCA and inverse transformation:", X_inverted.shape)
#             print("Reconstructed 3D shape:", X_3d_reconstructed.shape)
#             print ("MSE inverse transformation:", mse, "MSE reconstructed 3D:", mse_3d)

#             return X_inverted, X_3d_reconstructed
    
#         def get_features_targets(X_pca, n, m):
#             first_n_cols = []
#             last_m_cols = []
#             for i in range(0, X_pca.shape[1], n+m):
#                 first_n = X_pca[:, i:i+n]
#                 last_m = X_pca[:, i+n:i+(n+m)]
#                 first_n_cols.append(first_n)
#                 last_m_cols.append(last_m)
#             array_first_n = np.concatenate(first_n_cols, axis=1)
#             array_last_m = np.concatenate(last_m_cols, axis=1)
#             return array_first_n, array_last_m

#         def invert_features_targets(array_first_n, array_last_m, n_, m):
#             n = array_first_n.shape[1] // n_
#             reconstructed_cols = []
#             for i in range(n):
#                 cols_first_n = array_first_n[:, i*n_:(i+1)*n_]
#                 cols_last_m = array_last_m[:, i*m:(i+1)*m]
#                 for col in cols_first_n.T:
#                     reconstructed_cols.append(col)
#                 for col in cols_last_m.T:
#                     reconstructed_cols.append(col)
#             original_array_reconstructed = np.column_stack(reconstructed_cols)
#             return original_array_reconstructed

#         X_3d, X_flattened, label_column_indices, feature_scaler, target_scaler = get_array(filenames, datafolder_path, training_wind_angles, angle_to_label)
#         pca, X_pca = apply_PCA(X_3d, X_flattened, n_components=None)
#         X_inverted, X_3d_reconstructed = invert_pca(pca, X_pca)
#         features, targets = get_features_targets(X_pca, len(input_params), len(output_params))
#         original_array_reconstructed = invert_features_targets(features, targets, len(input_params), len(output_params))
#         X_train = features
#         y_train = targets
#         X_train_tensor = torch.tensor(np.array(X_train), dtype=torch.float32)
#         y_train_tensor = torch.tensor(np.array(y_train), dtype=torch.float32)
#         X_train_tensor = X_train_tensor.to(device)
#         y_train_tensor = y_train_tensor.to(device)

#         print (f"Features Size: {features.shape}, Targets Size: {targets.shape}")
#         print (f"Reconstructed Array: {original_array_reconstructed.shape}, Original Array: {X_pca.shape}")

#         print ("LOADED TRAINING DATASET")

#         return X_train_tensor, y_train_tensor, pca, feature_scaler, target_scaler, X_3d.shape

#     X_train_tensor, y_train_tensor, pca, feature_scaler, target_scaler, original_shape = get_train_test_tensor(filenames, datafolder_path, training_wind_angles, angle_to_label)

#     def get_new_skipped_values(list1, list2):
#         len1, len2 = len(list1), len(list2)
#         def pad_with_random_values(shorter, longer):
#             pad_length = len(longer) - len(shorter)
#             unique_values_for_padding = list(set(longer) - set(shorter))
#             if pad_length > len(unique_values_for_padding):
#                 raise ValueError("Not enough unique values in the longer list to pad without repetition.")
#             padding_values = random.sample(unique_values_for_padding, pad_length)
#             shorter.extend(padding_values)
#         if len1 < len2:
#             pad_with_random_values(list1, list2)
#         elif len2 < len1:
#             pad_with_random_values(list2, list1)
#         return list1, list2

#     for i in training_wind_angles:
#         if i < 180:
#             skipped_wind_angles.append(i)
#     # training_wind_angles, skipped_wind_angles = get_new_skipped_values(training_wind_angles, skipped_wind_angles)
#     # def extend_list_to_n_with_repeats(lst, n):
#     #     current_length = len(lst)
#     #     if n <= current_length:
#     #         return lst[:n]  # If n is less than or equal to current length, truncate the list.
#     #     else:
#     #         additional_items_needed = n - current_length
#     #         # Repeat the list items until reaching the desired length.
#     #         extended_list = lst * (n // current_length) + lst[:additional_items_needed % current_length]
#     #         return extended_list
#     # skipped_wind_angles = extend_list_to_n_with_repeats(skipped_wind_angles,len(training_wind_angles))
#     skipped_wind_angles = sorted(skipped_wind_angles)
#     print (f"skipped_wind_angles: {skipped_wind_angles}")
#     skipped_angle_to_label = {angle: idx for idx, angle in enumerate(sorted(skipped_wind_angles))}

#     def get_skipped_train_test_tensor(filenames, datafolder_path, training_wind_angles, angle_to_label, feature_scaler, target_scaler, pca=None):

#         print ("LOADING SKIPPED DATASET")

#         def get_array(filenames, datafolder_path, training_wind_angles, angle_to_label, feature_scaler, target_scaler):
#             all_data, labels = concatenate_data_files(filenames, datafolder_path, training_wind_angles, angle_to_label)
#             all_features = all_data[input_params]
#             all_targets = all_data[output_params]
#             normalized_features, normalized_targets = transform_data_with_scalers(all_features, all_targets, feature_scaler, target_scaler)

#             unique_labels = np.unique(np.array(labels))
#             n = len(unique_labels)  # Number of unique labels
#             m = normalized_features.shape[0] // n  # Number of data points per label
#             p_combined = normalized_features.shape[1] + normalized_targets.shape[1]  # Combined number of features and targets

#             label_column_indices = {}
#             for i, label in enumerate(unique_labels):
#                 start_index = i * p_combined
#                 end_index = start_index + p_combined
#                 label_column_indices[label] = (start_index, end_index)
#             # label_column_indices now maps each label to its column range in X_flattened
#             # specific_label = unique_labels[0]  # Example: Getting the first label
#             # start_index, end_index = label_column_indices[specific_label]
#             # # You can now use start_index and end_index to slice X_flattened for operations specific to this label
#             # specific_label_data = X_flattened[:, start_index:end_index]

#             combined_3d = np.zeros((n, m, p_combined))
#             for i, label in enumerate(unique_labels):
#                 indices = np.where(np.array(labels) == label)[0]
#                 combined = np.hstack((normalized_features[indices, :], normalized_targets[indices, :]))
#                 combined_3d[i, :, :] = combined

#             X_flattened = combined_3d.reshape(combined_3d.shape[1], -1)

#             return combined_3d, X_flattened, label_column_indices

#         def apply_PCA(X_3d, X_flattened, n_components=None, pca=None):
#             if pca is None:
#                 pca = PCA(n_components)
#                 X_pca = (pca.fit_transform(X_flattened.T)).T

#                 explained_variance_ratio = pca.explained_variance_ratio_
#                 print(f"Explained variance ratio of the first {len(explained_variance_ratio)} principal components:", explained_variance_ratio)
#                 print(f"Explained variance sum of the first {len(explained_variance_ratio)} principal components:", np.sum(explained_variance_ratio))

#                 print("Original shape:", X_3d.shape)
#                 print("Flattened shape:", X_flattened.shape)
#                 print("After PCA shape:", X_pca.shape)
#             else:
#                 X_pca = (pca.transform(X_flattened.T)).T
#                 print("Original shape:", X_3d.shape)
#                 print("Flattened shape:", X_flattened.shape)
#                 print("After PCA shape:", X_pca.shape)

#             return pca, X_pca

#         def invert_pca(pca, X_pca):
#             X_inverted = (pca.inverse_transform(X_pca.T)).T
#             original_shape = X_3d.shape 
#             X_3d_reconstructed = X_inverted.reshape(original_shape)

#             mse = mean_squared_error(X_flattened, X_inverted)
#             mse_3d = mean_squared_error(X_3d.flatten(), X_3d_reconstructed.flatten())

#             print("Shape after PCA and inverse transformation:", X_inverted.shape)
#             print("Reconstructed 3D shape:", X_3d_reconstructed.shape)
#             print ("MSE inverse transformation:", mse, "MSE reconstructed 3D:", mse_3d)
            
#             return X_inverted, X_3d_reconstructed
        
#         def get_features_targets(X_pca, n, m):
#             first_n_cols = []
#             last_m_cols = []
#             for i in range(0, X_pca.shape[1], n+m):
#                 first_n = X_pca[:, i:i+n]
#                 last_m = X_pca[:, i+n:i+(n+m)]
#                 first_n_cols.append(first_n)
#                 last_m_cols.append(last_m)
#             array_first_n = np.concatenate(first_n_cols, axis=1)
#             array_last_m = np.concatenate(last_m_cols, axis=1)
#             return array_first_n, array_last_m

#         def invert_features_targets(array_first_n, array_last_m, n_, m):
#             n = array_first_n.shape[1] // n_
#             reconstructed_cols = []
#             for i in range(n):
#                 cols_first_n = array_first_n[:, i*n_:(i+1)*n_]
#                 cols_last_m = array_last_m[:, i*m:(i+1)*m]
#                 for col in cols_first_n.T:
#                     reconstructed_cols.append(col)
#                 for col in cols_last_m.T:
#                     reconstructed_cols.append(col)
#             original_array_reconstructed = np.column_stack(reconstructed_cols)
#             return original_array_reconstructed

#         X_3d, X_flattened, label_column_indices = get_array(filenames, datafolder_path, training_wind_angles, angle_to_label, feature_scaler, target_scaler)
#         pca, X_pca = apply_PCA(X_3d, X_flattened, n_components=None)
#         X_inverted, X_3d_reconstructed = invert_pca(pca, X_pca)
#         features, targets = get_features_targets(X_pca, len(input_params), len(output_params))
#         original_array_reconstructed = invert_features_targets(features, targets, len(input_params), len(output_params))
#         X_test = features
#         y_test = targets
#         X_test_tensor = torch.tensor(np.array(X_test), dtype=torch.float32)
#         y_test_tensor = torch.tensor(np.array(y_test), dtype=torch.float32)
#         X_test_tensor = X_test_tensor.to(device)
#         y_test_tensor = y_test_tensor.to(device)

#         print (f"Features Size: {features.shape}, Targets Size: {targets.shape}")
#         print (f"Reconstructed Array: {original_array_reconstructed.shape}, Original Array: {X_pca.shape}")

#         print ("LOADED SKIPPED DATASET")

#         return X_test_tensor, y_test_tensor, pca

#     X_test_tensor_skipped, y_test_tensor_skipped, pca_skipped = get_skipped_train_test_tensor(filenames, datafolder_path, skipped_wind_angles, skipped_angle_to_label, feature_scaler, target_scaler, pca)

#     data_dict = {
#         "X_train_tensor": X_train_tensor,
#         "y_train_tensor": y_train_tensor,
#         "X_test_tensor_skipped": X_test_tensor_skipped,
#         "y_test_tensor_skipped": y_test_tensor_skipped,
#         "feature_scaler": feature_scaler,
#         "target_scaler": target_scaler,
#         "pca": pca,
#         "pca_skipped": pca_skipped,
#         "original_shape": original_shape,
#     }

#     return data_dict

def load_data_PCA(config, device):
    chosen_machine_key = config["chosen_machine"]
    datafolder_path = os.path.join(config["machine"][chosen_machine_key], config["data"]["data_folder_name"])
    filenames = get_filenames_from_folder(datafolder_path, config["data"]["extension"], config["data"]["startname_data"])
    training_wind_angles = config["training"]["angles_to_train"]
    skipped_wind_angles = config["training"]["angles_to_leave_out"]
    output_params = config["training"]["output_params"]
    input_params = config["training"]["input_params"]
    input_params_points = ['Points:0', 'Points:1', 'Points:2']
    angle_to_label = {angle: idx for idx, angle in enumerate(sorted(training_wind_angles))}

    def get_train_test_tensor(filenames, datafolder_path, training_wind_angles, angle_to_label):

        print ("LOADING TRAINING DATASET")

        def get_array(filenames, datafolder_path, training_wind_angles, angle_to_label):
            all_data, labels = concatenate_data_files(filenames, datafolder_path, training_wind_angles, angle_to_label)
            all_features = all_data[input_params]
            all_targets = all_data[output_params]
            feature_scaler, target_scaler = initialize_and_fit_scalers(all_features, all_targets, config)
            normalized_features, normalized_targets = transform_data_with_scalers(all_features, all_targets, feature_scaler, target_scaler)

            unique_labels = np.unique(np.array(labels))
            n = len(unique_labels)  # Number of unique labels
            m = normalized_features.shape[0] // n  # Number of data points per label
            p_combined = normalized_features.shape[1] + normalized_targets.shape[1]  # Combined number of features and targets

            label_column_indices = {}
            for i, label in enumerate(unique_labels):
                start_index = i * p_combined
                end_index = start_index + p_combined
                label_column_indices[label] = (start_index, end_index)

            # label_column_indices now maps each label to its column range in X_flattened
            # specific_label = unique_labels[0]  # Example: Getting the first label
            # start_index, end_index = label_column_indices[specific_label]
            # # You can now use start_index and end_index to slice X_flattened for operations specific to this label
            # specific_label_data = X_flattened[:, start_index:end_index]

            combined_3d = np.zeros((n, m, p_combined))
            for i, label in enumerate(unique_labels):
                indices = np.where(np.array(labels) == label)[0]
                combined = np.hstack((normalized_features[indices, :], normalized_targets[indices, :]))
                combined_3d[i, :, :] = combined
            X_flattened = combined_3d.reshape(combined_3d.shape[1], -1)
            
            return combined_3d, X_flattened, label_column_indices, feature_scaler, target_scaler

        def apply_PCA(X_3d, X_flattened, n_components=None):
            pca_rows = PCA(n_components)
            X_pca_rows = (pca_rows.fit_transform(X_flattened.T)).T

            explained_variance_ratio = pca_rows.explained_variance_ratio_
            print(f"Explained variance ratio of the first {len(explained_variance_ratio)} principal components:", explained_variance_ratio)
            print(f"Explained variance sum of the first {len(explained_variance_ratio)} principal components:", np.sum(explained_variance_ratio))

            print(f"Original shape: {X_3d.shape}, {X_3d.shape[0]} Angles, {X_3d.shape[1]} Rows, {X_3d.shape[2]} Columns")
            print(f"Flattened shape: {X_flattened.shape}, {X_flattened.shape[0]} Rows and {X_flattened.shape[1]} Columns")
            print(f"After PCA Rows shape: {X_pca_rows.shape}, {X_pca_rows.shape[0]} Rows and {X_pca_rows.shape[1]} Columns")

            pca_columns = PCA(n_components)
            X_pca_final = (pca_columns.fit_transform(X_pca_rows))

            explained_variance_ratio = pca_columns.explained_variance_ratio_
            print(f"Explained variance ratio of the first {len(explained_variance_ratio)} principal components:", explained_variance_ratio)
            print(f"Explained variance sum of the first {len(explained_variance_ratio)} principal components:", np.sum(explained_variance_ratio))

            print(f"After PCA Rows shape: {X_pca_final.shape}, {X_pca_final.shape[0]} Rows and {X_pca_final.shape[1]} Columns")

            return pca_rows, pca_columns, X_pca_final

        def invert_pca(pca_rows, pca_columns, X_pca, X_original):
            X_inverted = (pca_rows.inverse_transform((pca_columns.inverse_transform(X_pca)).T)).T
            mse = mean_squared_error(X_original, X_inverted)

            print("Shape after PCA and inverse transformation:", X_inverted.shape)
            print ("MSE inverse transformation:", mse)

            return X_inverted
    
        def get_features_targets(X_pca, n, m):
            first_n_cols = []
            last_m_cols = []
            for i in range(0, X_pca.shape[1], n+m):
                first_n = X_pca[:, i:i+n]
                last_m = X_pca[:, i+n:i+(n+m)]
                first_n_cols.append(first_n)
                last_m_cols.append(last_m)
            array_first_n = np.concatenate(first_n_cols, axis=1)
            array_last_m = np.concatenate(last_m_cols, axis=1)
            return array_first_n, array_last_m

        def invert_features_targets(array_first_n, array_last_m, n_, m):
            n = array_first_n.shape[1] // n_
            reconstructed_cols = []
            for i in range(n):
                cols_first_n = array_first_n[:, i*n_:(i+1)*n_]
                cols_last_m = array_last_m[:, i*m:(i+1)*m]
                for col in cols_first_n.T:
                    reconstructed_cols.append(col)
                for col in cols_last_m.T:
                    reconstructed_cols.append(col)
            original_array_reconstructed = np.column_stack(reconstructed_cols)
            return original_array_reconstructed

        X_3d, X_flattened, label_column_indices, feature_scaler, target_scaler = get_array(filenames, datafolder_path, training_wind_angles, angle_to_label)
        pca_rows, pca_columns, X_pca = apply_PCA(X_3d, X_flattened, n_components=None)
        X_inverted = invert_pca(pca_rows, pca_columns, X_pca, X_flattened)
        features, targets = get_features_targets(X_pca, len(input_params), len(output_params))
        original_array_reconstructed = invert_features_targets(features, targets, len(input_params), len(output_params))
        X_train = features
        y_train = targets
        X_train_tensor = torch.tensor(np.array(X_train), dtype=torch.float32)
        y_train_tensor = torch.tensor(np.array(y_train), dtype=torch.float32)
        X_train_tensor = X_train_tensor.to(device)
        y_train_tensor = y_train_tensor.to(device)

        print (f"Features Size: {features.shape}, Targets Size: {targets.shape}")
        print (f"Reconstructed Array: {original_array_reconstructed.shape}, Original Array: {X_pca.shape}")

        print ("LOADED TRAINING DATASET")

        return X_train_tensor, y_train_tensor, pca_rows, pca_columns, feature_scaler, target_scaler, X_3d.shape
    
    def get_skipped_test_tensor(filenames, datafolder_path, training_wind_angles, angle_to_label, feature_scaler, target_scaler, pca_rows, pca_columns):

        print ("LOADING SKIPPED DATASET")

        def get_array(filenames, datafolder_path, training_wind_angles, angle_to_label, feature_scaler, target_scaler):
            all_data, labels = concatenate_data_files(filenames, datafolder_path, training_wind_angles, angle_to_label)
            all_features = all_data[input_params]
            all_targets = all_data[output_params]
            normalized_features, normalized_targets = transform_data_with_scalers(all_features, all_targets, feature_scaler, target_scaler)

            unique_labels = np.unique(np.array(labels))
            n = len(unique_labels)  # Number of unique labels
            m = normalized_features.shape[0] // n  # Number of data points per label
            p_combined = normalized_features.shape[1] + normalized_targets.shape[1]  # Combined number of features and targets

            label_column_indices = {}
            for i, label in enumerate(unique_labels):
                start_index = i * p_combined
                end_index = start_index + p_combined
                label_column_indices[label] = (start_index, end_index)

            # label_column_indices now maps each label to its column range in X_flattened
            # specific_label = unique_labels[0]  # Example: Getting the first label
            # start_index, end_index = label_column_indices[specific_label]
            # # You can now use start_index and end_index to slice X_flattened for operations specific to this label
            # specific_label_data = X_flattened[:, start_index:end_index]

            combined_3d = np.zeros((n, m, p_combined))
            for i, label in enumerate(unique_labels):
                indices = np.where(np.array(labels) == label)[0]
                combined = np.hstack((normalized_features[indices, :], normalized_targets[indices, :]))
                combined_3d[i, :, :] = combined
            X_flattened = combined_3d.reshape(combined_3d.shape[1], -1)
            
            return combined_3d, X_flattened, label_column_indices

        def apply_PCA(X_3d, X_flattened, pca_rows, pca_columns):
            X_pca_rows = (pca_rows.transform(X_flattened.T)).T

            print(f"Original shape: {X_3d.shape}, {X_3d.shape[0]} Angles, {X_3d.shape[1]} Rows, {X_3d.shape[2]} Columns")
            print(f"Flattened shape: {X_flattened.shape}, {X_flattened.shape[0]} Rows and {X_flattened.shape[1]} Columns")
            print(f"After PCA Rows shape: {X_pca_rows.shape}, {X_pca_rows.shape[0]} Rows and {X_pca_rows.shape[1]} Columns")

            X_pca_final = (pca_columns.transform(X_pca_rows))

            print(f"After PCA Rows shape: {X_pca_final.shape}, {X_pca_final.shape[0]} Rows and {X_pca_final.shape[1]} Columns")

            return X_pca_final

        def invert_pca(pca_rows, pca_columns, X_pca, X_original):
            X_inverted = (pca_rows.inverse_transform((pca_columns.inverse_transform(X_pca)).T)).T
            mse = mean_squared_error(X_original, X_inverted)

            print("Shape after PCA and inverse transformation:", X_inverted.shape)
            print ("MSE inverse transformation:", mse)

            return X_inverted
    
        def get_features_targets(X_pca, n, m):
            first_n_cols = []
            last_m_cols = []
            for i in range(0, X_pca.shape[1], n+m):
                first_n = X_pca[:, i:i+n]
                last_m = X_pca[:, i+n:i+(n+m)]
                first_n_cols.append(first_n)
                last_m_cols.append(last_m)
            array_first_n = np.concatenate(first_n_cols, axis=1)
            array_last_m = np.concatenate(last_m_cols, axis=1)
            return array_first_n, array_last_m

        def invert_features_targets(array_first_n, array_last_m, n_, m):
            n = array_first_n.shape[1] // n_
            reconstructed_cols = []
            for i in range(n):
                cols_first_n = array_first_n[:, i*n_:(i+1)*n_]
                cols_last_m = array_last_m[:, i*m:(i+1)*m]
                for col in cols_first_n.T:
                    reconstructed_cols.append(col)
                for col in cols_last_m.T:
                    reconstructed_cols.append(col)
            original_array_reconstructed = np.column_stack(reconstructed_cols)
            return original_array_reconstructed

        X_3d, X_flattened, label_column_indices = get_array(filenames, datafolder_path, training_wind_angles, angle_to_label, feature_scaler, target_scaler)
        X_pca = apply_PCA(X_3d, X_flattened, pca_rows, pca_columns)
        X_inverted = invert_pca(pca_rows, pca_columns, X_pca, X_flattened)
        features, targets = get_features_targets(X_pca, len(input_params), len(output_params))
        original_array_reconstructed = invert_features_targets(features, targets, len(input_params), len(output_params))
        X_train = features
        y_train = targets
        X_train_tensor = torch.tensor(np.array(X_train), dtype=torch.float32)
        y_train_tensor = torch.tensor(np.array(y_train), dtype=torch.float32)
        X_train_tensor = X_train_tensor.to(device)
        y_train_tensor = y_train_tensor.to(device)

        print (f"Features Size: {features.shape}, Targets Size: {targets.shape}")
        print (f"Reconstructed Array: {original_array_reconstructed.shape}, Original Array: {X_pca.shape}")

        print ("LOADED SKIPPED DATASET")

        return X_train_tensor, y_train_tensor

    for i in training_wind_angles:
        if i < 180:
            skipped_wind_angles.append(i)

    skipped_wind_angles = sorted(skipped_wind_angles)
    print (f"skipped_wind_angles: {skipped_wind_angles}")
    skipped_angle_to_label = {angle: idx for idx, angle in enumerate(sorted(skipped_wind_angles))}

    X_train_tensor, y_train_tensor, pca_rows, pca_columns, feature_scaler, target_scaler, original_shape = get_train_test_tensor(filenames, datafolder_path, training_wind_angles, angle_to_label)
    X_test_tensor_skipped, y_test_tensor_skipped = get_skipped_test_tensor(filenames, datafolder_path, skipped_wind_angles, skipped_angle_to_label, feature_scaler, target_scaler, pca_rows, pca_columns)

    data_dict = {
        "X_train_tensor": X_train_tensor,
        "y_train_tensor": y_train_tensor,
        "X_test_tensor_skipped": X_test_tensor_skipped,
        "y_test_tensor_skipped": y_test_tensor_skipped,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "pca_rows": pca_rows,
        "pca_columns": pca_columns,
        "original_shape": original_shape,
    }

    return data_dict

# device = 'cpu'

# data_dict = load_data_PCA(config, device)

# print (data_dict)

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
    
    if config["training"]["use_data_for_cont_loss"]:
        div_data = load_derivative_data(config, training_wind_angles)
        input_params_modf = config["training"]["input_params_modf"]
        output_params_modf = config["training"]["output_params_modf"]
        input_stds_means = extract_stds_means(feature_scaler, input_params_modf)
        output_stds_means = extract_stds_means(target_scaler, output_params_modf)
        stds_means_dict = {**input_stds_means, **output_stds_means}

        u_x = div_data["u_x"]/(stds_means_dict['Velocity_X_std']/stds_means_dict['X_std'])
        v_y = div_data["v_y"]/(stds_means_dict['Velocity_Y_std']/stds_means_dict['Y_std'])
        w_z = div_data["w_z"]/(stds_means_dict['Velocity_Z_std']/stds_means_dict['Z_std'])

        normalized_divergences = u_x + v_y + w_z
        X_train_divergences, X_test_divergences, y_train_divergences, y_test_divergences = train_test_split(normalized_features, normalized_divergences, test_size=config["train_test"]["test_size"], random_state=config["train_test"]["random_state"])
        X_train_divergences_tensor, X_test_divergences_tensor, y_train_divergences_tensor, y_test_divergences_tensor = convert_to_tensor(X_train_divergences, X_test_divergences, y_train_divergences, y_test_divergences, device=device)
    else:
        X_train_divergences_tensor, X_test_divergences_tensor, y_train_divergences_tensor, y_test_divergences_tensor = None,None,None,None

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
        "target_scaler": target_scaler,
        "X_train_divergences_tensor": X_train_divergences_tensor,
        "y_train_divergences_tensor": y_train_divergences_tensor,
        "X_test_divergences_tensor": X_test_divergences_tensor,
        "y_test_divergences_tensor": y_test_divergences_tensor
    }

    return data_dict

def load_data_new_angles(device, config, feature_scaler, target_scaler, wind_angles=None):
    chosen_machine_key = config["chosen_machine"]
    datafolder_path = os.path.join(config["machine"][chosen_machine_key], config["data"]["data_folder_name"])
    filenames = get_filenames_from_folder(datafolder_path, config["data"]["extension"], config["data"]["startname_data"])
    dfs = []
    if wind_angles is None:
        wind_angles = config["train_test"]["new_angles"]
        for wind_angle in wind_angles:
            df = pd.read_csv(os.path.join(datafolder_path, filenames[0]))
            df['cos(WindAngle)'] = (np.cos(np.deg2rad(wind_angle)))
            df['sin(WindAngle)'] = (np.sin(np.deg2rad(wind_angle)))
            dfs.append(df)
    else:
        for filename in sorted(filenames):
            df = pd.read_csv(os.path.join(datafolder_path, filename))
            wind_angle = int(filename.split('_')[-1].split('.')[0])
            if wind_angle in wind_angles:
                df['cos(WindAngle)'] = (np.cos(np.deg2rad(wind_angle)))
                df['sin(WindAngle)'] = (np.sin(np.deg2rad(wind_angle)))
                dfs.append(df)
    data = pd.concat(dfs)
    features = data[config["training"]["input_params"]]
    targets = data[config["training"]["output_params"]]
    normalized_features, normalized_targets = transform_data_with_scalers(features, targets, feature_scaler, target_scaler)
    X_train, X_test, y_train, y_test = train_test_split(normalized_features, normalized_targets,test_size=config["train_test"]["new_angles_test_size"], random_state=config["train_test"]["random_state"])
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    X_test_tensor = X_test_tensor.to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    y_test_tensor = y_test_tensor.to(device)
    return X_test_tensor, y_test_tensor

def get_train_loader(X_train_tensor, y_train_tensor, labels_train_tensor, batch_size, wind_angles):
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    dataset = WindAngleDataset(train_dataset, labels_train_tensor)
    sampler = BalancedWindAngleSampler(dataset, wind_angles=np.arange(len(wind_angles)))
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return train_loader

def load_ensight_data(file_name):
    reader = vtkEnSightGoldBinaryReader()
    reader.SetCaseFileName(file_name)
    reader.ReadAllVariablesOn()
    reader.Update()
    return reader.GetOutput()

def find_closest_cell_ids(df, cell_id_to_position):
    cell_positions = list(cell_id_to_position.values())
    cell_ids = list(cell_id_to_position.keys())
    tree = cKDTree(cell_positions)
    df_positions = df[['X', 'Y', 'Z']].values  
    distances, indices = tree.query(df_positions)
    closest_cell_ids = [cell_ids[index] for index in indices]   
    return closest_cell_ids

def add_cell_ids_to_df(df, cell_id_to_position):
    closest_cell_ids = find_closest_cell_ids(df, cell_id_to_position)
    df['cell_id'] = closest_cell_ids
    return df

def slice_lists(input_list):
    velocities = [item for item in input_list if 'Velocity' in item]
    extras = [item for item in input_list if 'Velocity' not in item]
    print (velocities, extras)
    return velocities, extras

def append_nn2CFD(CFD_data, nn_data, name):

    nn_data = nn_data.to_numpy()
    csv_ind = nn_data[:, 0].astype(int)
    # Scalar data
    if nn_data.shape[1] == 2:
        csv_val = nn_data[:, 1]
    # 3D vector data
    elif nn_data.shape[1] == 4:
        csv_val = nn_data[:, 1:4]
    else:
        raise ValueError('User-defined data must either be a scalar or a 3D vector!')

    assert CFD_data.GetNumberOfCells() == len(csv_ind)

    nn_dict = {}
    for i in range(len(csv_ind)):
        cell_id  = csv_ind[i]
        cell_val = csv_val[i]
        nn_dict[cell_id] = cell_val

    nn_dict = {k: nn_dict[k] for k in sorted(nn_dict)}
    nn_array = np.array(list(nn_dict.values()))

    vtk_user_data = numpy_to_vtk(nn_array, deep=True)
    vtk_user_data.SetName(name) 
    CFD_data.GetCellData().AddArray(vtk_user_data)

    return CFD_data

def save_vtu_data(file_name, ugrid):
    vtk_writer = vtkXMLUnstructuredGridWriter()
    vtk_writer.SetFileName(file_name)
    vtk_writer.SetInputData(ugrid)
    vtk_writer.Write()

def output_nn_to_vtk(config, angle, filename, df, column_names, output_folder):
    chosen_machine_key = config["chosen_machine"]
    datafolder_path = os.path.join(config["machine"][chosen_machine_key], config["data"]["data_folder_name"])
    core_data = os.path.join(datafolder_path, "core_data", "core_data", f"deg_{angle}")
    case_file = os.path.join(core_data, "RESULTS_FLUID_DOMAIN.case")

    ensight_data = load_ensight_data(case_file)
    ensight_data = ensight_data.GetBlock(0)
    cell_id_to_position = {}
    cell_centers_filter = vtk.vtkCellCenters()
    cell_centers_filter.SetInputData(ensight_data)
    cell_centers_filter.VertexCellsOn()
    cell_centers_filter.Update()
    centers_polydata = cell_centers_filter.GetOutput()
    points = centers_polydata.GetPoints()
    for i in range(points.GetNumberOfPoints()):
        position = points.GetPoint(i)
        cell_id_to_position[i] = position
    output_data = add_cell_ids_to_df(df, cell_id_to_position)
    indices_to_keep = output_data['cell_id']

    ids = vtk.vtkIdTypeArray()
    ids.SetNumberOfComponents(1)
    for cell_id in indices_to_keep:
        ids.InsertNextValue(cell_id)

    selectionNode = vtk.vtkSelectionNode()
    selectionNode.SetFieldType(vtk.vtkSelectionNode.CELL)
    selectionNode.SetContentType(vtk.vtkSelectionNode.INDICES)
    selectionNode.SetSelectionList(ids)
    selection = vtk.vtkSelection()
    selection.AddNode(selectionNode)
    extractSelection = vtk.vtkExtractSelection()
    extractSelection.SetInputData(0, ensight_data)
    extractSelection.SetInputData(1, selection)
    extractSelection.Update()
    trimmed_data = extractSelection.GetOutput()

    column_names.append('cell_id')
    velocities, extras = slice_lists(column_names)
    velocities = ['cell_id', *velocities]

    output_data_trimmed = output_data[column_names]
    output_data_velocity = output_data_trimmed[velocities]

    # Function to find and drop duplicate columns
    def drop_duplicate_columns(df):
        # Transpose the DataFrame to work with rows (which are now original columns)
        df_T = df.T
        # Identify duplicate rows (which are originally duplicate columns)
        duplicate_columns = df_T.duplicated()
        # Filter out the duplicate column names
        columns_to_drop = duplicate_columns[duplicate_columns].index.tolist()
        # Drop the duplicate columns from the original DataFrame
        df.drop(columns=columns_to_drop, inplace=True)
        return df

    print (velocities)
    print (output_data_trimmed)
    print (output_data_velocity)
    output_data_velocity = drop_duplicate_columns(output_data_velocity)
    print (output_data_velocity)

    trimmed_data = append_nn2CFD(trimmed_data, output_data_velocity, 'Velocity_Predicted')

    if len(extras)!=0:
        for i in extras:
            if i != 'cell_id':
                column_name_ = ['cell_id', i]
                output_data_i = output_data_trimmed[column_name_]
                trimmed_data = append_nn2CFD(trimmed_data, output_data_i, i)
    
    save_vtu_data(os.path.join(output_folder, f'{filename}.vtu'), trimmed_data)

def evaluate_model(config, model, wind_angles, X_test_tensor, feature_scaler, target_scaler, y_test_tensor = None, output_folder = None, physics = None, vtk_output = None):
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
                    if vtk_output is not None and physics is not None and wind_angle in config["training"]["all_angles"]:
                        os.makedirs(vtk_output, exist_ok=True)
                        output_nn_to_vtk(config, wind_angle, f'{physics}_predictions_for_wind_angle_{wind_angle}', combined_df, predictions_column_names, vtk_output)
                    if config["plotting"]["save_csv_predictions"]:
                        data_folder = os.path.join(output_folder, f'data_output_for_wind_angle_{wind_angle}')
                        os.makedirs(data_folder, exist_ok=True)
                        combined_df = pd.concat([filtered_X_test_dataframe, filtered_predictions], axis=1)
                        combined_file_path = os.path.join(data_folder, f'combined_predictions_for_wind_angle_{wind_angle}.csv')
                        combined_df.to_csv(combined_file_path, index=False)
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
                    if vtk_output is not None and physics is not None and wind_angle in config["training"]["all_angles"]:
                        os.makedirs(vtk_output, exist_ok=True)
                        output_nn_to_vtk(config, wind_angle, f'{physics}_predictions_for_wind_angle_{wind_angle}', combined_df, predictions_column_names, vtk_output)
                        # output_nn_to_vtk(config, wind_angle, f'{physics}_predictions_for_wind_angle_{wind_angle}', combined_df, y_test_column_names, vtk_output)
                    if config["plotting"]["save_csv_predictions"]:
                        combined_df.to_csv(combined_file_path, index=False)
                    metrics_df = pd.DataFrame(rows_list)   
                    metrics_file_path = os.path.join(data_folder, f'metrics_for_wind_angle_{wind_angle}.csv')
                    metrics_df.to_csv(metrics_file_path, index=False)
                    dfs.append(combined_df)
        data = pd.concat(dfs)
        return data

def evaluate_model_PCA(config, model, wind_angles, X_test_tensor, feature_scaler, target_scaler, y_test_tensor, pca_rows, pca_columns, original_shape, output_folder, physics, vtk_output):
    model.eval()
    dfs = []
    with torch.no_grad():
        predictions_tensor = model(X_test_tensor)
        predictions_tensor_cpu = predictions_tensor.cpu()
        y_test_tensor_cpu = y_test_tensor.cpu()
        X_test_tensor_cpu = X_test_tensor.cpu()
        X_test_tensor_cpu, y_test_tensor_cpu, predictions_tensor_cpu = invert_data_PCA(config, X_test_tensor_cpu, y_test_tensor_cpu, predictions_tensor_cpu, pca_rows, pca_columns, original_shape)
        X_test_tensor_cpu = inverse_transform_features(X_test_tensor_cpu, feature_scaler)
        X_test_column_names = config["training"]["input_params_modf"]
        X_test_dataframe = pd.DataFrame(X_test_tensor_cpu, columns=X_test_column_names)
        os.makedirs(output_folder, exist_ok=True)
        y_test_tensor_cpu = inverse_transform_targets(y_test_tensor_cpu, target_scaler)
        output_column_names = config["training"]["output_params_modf"]
        y_test_column_names = [item + "_Actual" for item in output_column_names]
        y_test_dataframe = pd.DataFrame(y_test_tensor_cpu, columns=y_test_column_names)
        y_test_dataframe['Velocity_Magnitude_Actual'] = np.sqrt(y_test_dataframe['Velocity_X_Actual']**2 + y_test_dataframe['Velocity_Y_Actual']**2 + y_test_dataframe['Velocity_Z_Actual']**2)
        predictions_tensor_cpu = inverse_transform_targets(predictions_tensor_cpu, target_scaler)
        predictions_column_names = [item + "_Predicted" for item in output_column_names]
        predictions_dataframe = pd.DataFrame(predictions_tensor_cpu, columns=predictions_column_names)
        predictions_dataframe['Velocity_Magnitude_Predicted'] = np.sqrt(predictions_dataframe['Velocity_X_Predicted']**2 + predictions_dataframe['Velocity_Y_Predicted']**2 + predictions_dataframe['Velocity_Z_Predicted']**2)
        for wind_angle in wind_angles:
            print (wind_angle)
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
                # if vtk_output is not None and physics is not None and wind_angle in config["training"]["all_angles"]:
                #     os.makedirs(vtk_output, exist_ok=True)
                #     output_nn_to_vtk(config, wind_angle, f'{physics}_predictions_for_wind_angle_{wind_angle}', combined_df, predictions_column_names, vtk_output)
                if config["plotting"]["save_csv_predictions"]:
                    combined_df.to_csv(combined_file_path, index=False)
                metrics_df = pd.DataFrame(rows_list)   
                metrics_file_path = os.path.join(data_folder, f'metrics_for_wind_angle_{wind_angle}.csv')
                metrics_df.to_csv(metrics_file_path, index=False)
                dfs.append(combined_df)
        data = pd.concat(dfs)
        return data