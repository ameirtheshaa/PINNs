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
from stl import mesh
from pathlib import Path

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
    if wind_angles is None:
        wind_angles = config["train_test"]["new_angles"]
    datafolder_path = os.path.join(config["machine"][chosen_machine_key], config["data"]["data_folder_name"])
    filenames = get_filenames_from_folder(datafolder_path, config["data"]["extension"], config["data"]["startname_data"])
    dfs = []
    for wind_angle in wind_angles:
        df = pd.read_csv(os.path.join(datafolder_path, filenames[0]))
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
    return X_test_tensor

def get_train_loader(X_train_tensor, y_train_tensor, labels_train_tensor, batch_size, wind_angles):
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    dataset = WindAngleDataset(train_dataset, labels_train_tensor)
    sampler = BalancedWindAngleSampler(dataset, wind_angles=np.arange(len(wind_angles)))
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return train_loader

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