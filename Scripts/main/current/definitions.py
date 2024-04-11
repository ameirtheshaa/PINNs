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
import copy
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
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import MiniBatchSparsePCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.metrics import mean_squared_error
import dask.array as da
from dask.array.linalg import svd
from stl import mesh
from pathlib import Path
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkIOEnSight import vtkEnSightGoldBinaryReader
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridWriter
from scipy.spatial import cKDTree
from PINN import *

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

def get_available_device_memory(config, device):
    if config["training"]["force_device"] is not None:
        device = config["training"]["force_device"]
        if device == 'cpu':
            cpu_memory = psutil.virtual_memory()
            total_cpu_memory = cpu_memory.total/(1024**3)
            available_cpu_memory = cpu_memory.available/(1024**3)
            free_memory = available_cpu_memory
            total_memory = total_cpu_memory
        elif device.startswith('cuda'):
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
            free_memory, total_memory = None, None
    else:
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

def print_and_set_available_gpus(config):
    if config["training"]["force_device"] is not None:
        device = config["training"]["force_device"]
    else:
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            print(f"{num_gpus} GPUs available:")
            free_memories = []
            for i in range(num_gpus):
                free_memory, _ = get_available_device_memory(config, i)
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

def capture_model_init_params(config, input_params, output_params):
    model_params = {
        'input_params': input_params,
        'output_params': output_params,
        'hidden_layers': config["training"]["number_of_hidden_layers"],
        'neurons_per_layer': [config["training"]["neuron_number"]] * config["training"]["number_of_hidden_layers"],
        'activation': config["training"]["activation_function"],
        'use_batch_norm': config["training"]["batch_normalization"],
        'dropout_rate': config["training"]["dropout_rate"]
    }
    return model_params

def extract_relevant_training_config(config):
    return {key: config['training'].get(key) for key in ['PCA', 'output_params', 'output_params_modf']}

def initalize_model(config, device, input_params, output_params):
    model = PINN(input_params=input_params, output_params=output_params, hidden_layers=config["training"]["number_of_hidden_layers"], neurons_per_layer=[config["training"]["neuron_number"]] * config["training"]["number_of_hidden_layers"], activation=config["training"]["activation_function"], use_batch_norm=config["training"]["batch_normalization"], dropout_rate=config["training"]["dropout_rate"]).to(device)
    return model

def initialize_data(config):
    device = print_and_set_available_gpus(config)
    start_time = time.time()
    print ('Starting to load data.. ')
    if config["train_test"]["train"] or config["train_test"]["test"] or config["train_test"]["evaluate"] or config["train_test"]["evaluate_new_angles"] or config["train_test"]["boundary_test"] or config["plotting"]["make_div_plots"] or config["plotting"]["make_RANS_plots"]:
        if config["training"]["use_PCA"]:
            data_dict = load_data_PCA(config, device)
            input_params = [f"{value}_{angle}" for angle in config["training"]["angles_to_train"] for value in config["training"]["input_params"]]*config["training"]["features_factor"]
            output_params = [f"{value}_{angle}" for angle in config["training"]["angles_to_train"] for value in config["training"]["output_params"]]*config["training"]["targets_factor"]
        elif config['training']['use_fft']:
            data_dict = load_data_fft(config, device)
            input_params = [f"{value}_{angle}" for angle in config["training"]["angles_to_train"] for value in config["training"]["input_params"]]*config["training"]["features_factor"]
            output_params = [f"{value}_{angle}" for angle in config["training"]["angles_to_train"] for value in config["training"]["output_params"]]*config["training"]["targets_factor"]
        else:
            data_dict = load_data(config, device)
            input_params = config["training"]["input_params"]
            output_params = config["training"]["output_params"]
        model = initalize_model(config, device, input_params, output_params)
    else:
        data_dict, input_params, output_params, model = None, None, None, None
    print ('Data loaded! ', time.time() - start_time)
    return device, data_dict, input_params, output_params, model

def reinitialize_data(config, input_params, output_params, previous_params):
    def compare_and_report_changes(previous_params, current_params):
        changes = {}
        for key in current_params.keys():
            if previous_params is not None and current_params[key] != previous_params.get(key):
                changes[key] = (previous_params[key], current_params[key])
        return changes
    current_params = capture_model_init_params(config, input_params, output_params)
    if previous_params is not None and current_params != previous_params:
        changes = compare_and_report_changes(previous_params, current_params)
        print (f"there are changes, why are there changes - {changes}")
        if 'input_params' in changes or 'output_params' in changes:
            device, data_dict, input_params, output_params, model = initialize_data(config)
            return data_dict, input_params, output_params, model, current_params
        else:
            model = initalize_model(config, device, input_params, output_params)
            return model, current_params
    else:
        return current_params

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
        'Grad Norm': loss_dict.get('grad_norm', 0),
        'Total Loss': loss_dict.get('total_loss', 0),
        'Total Loss Weighted': loss_dict.get('total_loss_weighted', 0)
    }
    for key in data:
        if key != 'Epoch' and key != 'Total Time Elapsed (hours)' and key != 'Free Memory (GB)' and data[key] != 0:
            data[key] = data[key].cpu().detach().numpy()
    file_exists = os.path.isfile(file_path) and os.path.getsize(file_path) > 0
    if file_exists:
        with open(file_path, mode='r', newline='') as file:
            reader = csv.reader(file)
            existing_headers = next(reader, None)
        row_to_write = [data.get(header, '') for header in existing_headers]
        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row_to_write)
    else:
        with open(file_path, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=data.keys())
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
        optimizer = torch.optim.LBFGS(model.parameters(), 
            # lr=optimizer_config["learning_rate"], 
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
    data = pd.concat(dfs, ignore_index=True)
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
    df_ = df.copy()
    closest_cell_ids = find_closest_cell_ids(df_, cell_id_to_position)
    df_['cell_id'] = closest_cell_ids
    return df_

def slice_lists(input_list):
    velocities = [item for item in input_list if 'Velocity' in item]
    extras = [item for item in input_list if 'Velocity' not in item]
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

    if 'cell_id' not in column_names:
        column_names.append('cell_id')
    velocities_sliced, extras = slice_lists(column_names)
    velocities = ['cell_id', *velocities_sliced]

    output_data_trimmed = output_data[column_names].copy()
    output_data_velocity = output_data_trimmed[velocities].copy()

    trimmed_data = append_nn2CFD(trimmed_data, output_data_velocity, 'Velocity_Predicted')

    if len(extras)!=0:
        for i in extras:
            if i != 'cell_id':
                column_name_ = ['cell_id', i]
                output_data_i = output_data_trimmed[column_name_].copy()
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
                        if config["plotting"]["save_vtk"]:
                            output_nn_to_vtk(config, wind_angle, f'{physics}_predictions_for_wind_angle_{wind_angle}', combined_df, predictions_column_names, vtk_output)
                    if config["plotting"]["save_csv_predictions"]:
                        combined_df.to_csv(combined_file_path, index=False)
                    metrics_df = pd.DataFrame(rows_list)   
                    metrics_file_path = os.path.join(data_folder, f'metrics_for_wind_angle_{wind_angle}.csv')
                    metrics_df.to_csv(metrics_file_path, index=False)
                    dfs.append(combined_df)
        data = pd.concat(dfs)
        return data

# ###PCA###
# def load_data_PCA(config, device):
#     chosen_machine_key = config["chosen_machine"]
#     datafolder_path = os.path.join(config["machine"][chosen_machine_key], config["data"]["data_folder_name"])
#     filenames = get_filenames_from_folder(datafolder_path, config["data"]["extension"], config["data"]["startname_data"])
#     training_wind_angles = config["training"]["angles_to_train"]
#     skipped_wind_angles = config["training"]["angles_to_leave_out"]
#     output_params = config["training"]["output_params"]
#     input_params = config["training"]["input_params"]
#     input_params_points = config["training"]["input_params_points"]
#     angle_to_label = {angle: idx for idx, angle in enumerate(sorted(training_wind_angles))}

#     def get_features(df, input_params, feature_scaler):
#         df = df[input_params]
#         features = np.array(df)
#         if feature_scaler is None:
#             feature_scaler = config["training"]["feature_scaler"]
#             feature_scaler.fit(features[:, :3])
#         features[:, :3] = feature_scaler.transform(features[:, :3])
#         return features, feature_scaler

#     def get_targets(df, output_params, target_scaler):
#         targets = df[output_params]
#         if target_scaler is None:
#             target_scaler = config["training"]["target_scaler"]
#             target_scaler.fit(targets)
#         targets = target_scaler.transform(targets)
#         return targets, target_scaler

#     def reshaped_array(arr,n):
#         num_rows, num_cols = arr.shape 
#         new_shape = (num_rows//n, num_cols*n)
#         reshaped_arr = arr.reshape(n, int(num_rows/n), num_cols).reshape(new_shape)
#         return reshaped_arr

#     def inverse_reshaped_array(arr, n):
#         num_rows, num_cols = arr.shape
#         new_shape = (num_rows * n, num_cols // n)
#         reshaped_arr = arr.reshape(n, int(num_rows), int(num_cols/n)).reshape(new_shape)
#         return reshaped_arr

#     def get_tensor(config, filenames, datafolder_path, wind_angles, angle_to_label, device, pca_reduce, feature_scaler=None, target_scaler=None, pca_features=None, pca_targets=None):
#         data, labels = concatenate_data_files(filenames, datafolder_path, wind_angles, angle_to_label)
#         features, feature_scaler = get_features(data, input_params, feature_scaler)
#         targets, target_scaler = get_targets(data, output_params, target_scaler)

#         features = reshaped_array(features, len(wind_angles))
#         targets = reshaped_array(targets, len(wind_angles))

#         reduced_features, pca_features = compute_PCA(features, pca_reduce, pca=pca_features)
#         reduced_targets, pca_targets = compute_PCA(targets, pca_reduce, pca=pca_targets)

#         x = [reduced_features, pca_features, reduced_targets, pca_targets, features, targets, feature_scaler, target_scaler]

#         # print ('I NEED TO CHECK NOW')
#         # targets_reconstructed = compute_inverse_PCA_from_PCA(reduced_targets, pca_targets, 'rows')
#         # mse = ((targets - targets_reconstructed) ** 2).mean(axis=0)
#         # for i, mse_value in enumerate(mse, 1):
#         #     print(f"Column {i} MSE: {mse_value}")
#         # print ('I HAVE CHECK NOW')

#         return x 

#     def compute_PCA(Z, pca_reduce, pca=None, n_components=None):
#         if pca is None:
#                 print ('fitting')
#                 pca = PCA(n_components, svd_solver='full')
#                 if pca_reduce == 'rows':
#                     Z_reduced = (pca.fit_transform(Z.T)).T
#                 elif pca_reduce == 'columns':
#                     Z_reduced = (pca.fit_transform(Z))
#                 eigenvalues_ratio = pca.explained_variance_ratio_
#                 print(f"Explained variance ratio of the first {len(eigenvalues_ratio)} principal components: {eigenvalues_ratio} w sum = {np.sum(eigenvalues_ratio)} w reduced matrix {Z_reduced.shape}")
#         else:
#                 print ('im not fitting')
#                 if pca_reduce == 'rows':
#                     Z_reduced = (pca.transform(Z.T)).T
#                 elif pca_reduce == 'columns':
#                     Z_reduced = (pca.transform(Z))
#         Z_r = compute_inverse_PCA_from_PCA(Z_reduced, pca, pca_reduce)
#         compute_mse(Z, Z_r, 'inverse of computed PCA')
#         return Z_reduced, pca

#     def compute_inverse_PCA_from_PCA(Z_reduced, pca, pca_reduce):
#             if pca_reduce == 'rows':
#                 Z_r = (pca.inverse_transform((Z_reduced).T)).T
#             if pca_reduce == 'columns':
#                 Z_r = (pca.inverse_transform((Z_reduced)))
#             return Z_r

#     def get_skipped_angles(old_skipped, fixed_training):
#         some_temp_shit = []
#         for i in fixed_training:
#             some_temp_shit.append(i)
#         for val in old_skipped:
#             index_of_closest = np.argmin(np.abs(np.array(some_temp_shit) - val))
#             some_temp_shit[index_of_closest] = val

#         skipped_angle_to_label = {angle: idx for idx, angle in enumerate(sorted(some_temp_shit))}
#         return some_temp_shit, skipped_angle_to_label

#     def compute_mse(Z, Z_r, description=None):
#         mse = mean_squared_error(np.abs(Z), np.abs(Z_r))
#         print (f'mse: {mse} - {description}')
#         return mse

#     pca_reduce = 'rows'
#     x_training = get_tensor(config, filenames, datafolder_path, training_wind_angles, angle_to_label, device, pca_reduce)

#     some_temp = training_wind_angles
#     skipped_wind_angles, skipped_angle_to_label = get_skipped_angles(skipped_wind_angles, some_temp)

#     print ('starting skipped', skipped_wind_angles)

#     x_skipped = get_tensor(config, filenames, datafolder_path, skipped_wind_angles, skipped_angle_to_label, device, pca_reduce, x_training[6], x_training[7], x_training[1], x_training[3])

#     X_train_tensor = (torch.tensor(np.array(x_training[0]), dtype=torch.float32)).to(device)
#     y_train_tensor = (torch.tensor(np.array(x_training[2]), dtype=torch.float32)).to(device)

#     X_test_tensor_skipped = (torch.tensor(np.array(x_skipped[0]), dtype=torch.float32)).to(device)
#     y_test_tensor_skipped = (torch.tensor(np.array(x_skipped[2]), dtype=torch.float32)).to(device)

#     x_training.append(pca_reduce)
#     x_skipped.append(pca_reduce)

#     data_dict = {
#         "X_train_tensor": X_train_tensor,
#         "y_train_tensor": y_train_tensor,
#         "X_test_tensor_skipped": X_test_tensor_skipped,
#         "y_test_tensor_skipped": y_test_tensor_skipped,
#         "relevant_data_training": x_training,
#         "relevant_data_skipped": x_skipped
#         }
    
#     return data_dict

# def invert_data_PCA(config, pred, x):

#     def compute_inverse_PCA_from_PCA(Z_reduced, pca, pca_reduce):
#         if pca_reduce == 'rows':
#             Z_r = (pca.inverse_transform((Z_reduced).T)).T
#         if pca_reduce == 'columns':
#             Z_r = (pca.inverse_transform((Z_reduced)))
#         return Z_r

#     def compute_mse(Z, Z_r, description=None):
#         mse = mean_squared_error(np.abs(Z), np.abs(Z_r))
#         print (f'mse: {mse} - {description}')
#         return mse

#     [reduced_features, pca_features, reduced_targets, pca_targets, features, targets, feature_scaler, target_scaler, pca_reduce] = x

#     def inverse_reshaped_array(arr, n):
#         num_rows, num_cols = arr.shape
#         new_shape = (num_rows * n, num_cols // n)
#         reshaped_arr = arr.reshape(n, int(num_rows), int(num_cols/n)).reshape(new_shape)
#         return reshaped_arr


#     pred_reconstructed = compute_inverse_PCA_from_PCA(pred, pca_targets, pca_reduce)
#     pred_reconstructed = inverse_reshaped_array(pred_reconstructed, len(config["training"]["angles_to_train"]))
#     targets = inverse_reshaped_array(targets, len(config["training"]["angles_to_train"]))
#     pred_reconstructed = inverse_transform_targets(pred_reconstructed, target_scaler)
#     targets = inverse_transform_targets(targets, target_scaler)
#     assert targets.shape == pred_reconstructed.shape
#     x_reconstructed = [targets, pred_reconstructed]
#     compute_mse(targets, pred_reconstructed, description='TARGETS vs PRED MSE inv FFT inv PCA')
#     return x_reconstructed

# def evaluate_model_training_PCA(config, device, model, model_file_path, X_test_tensor, y_test_tensor, x):
#     checkpoint = torch.load(model_file_path, map_location=device)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.eval()
#     with torch.no_grad():
#         predictions_tensor = model(X_test_tensor)
#         predictions_tensor_cpu = predictions_tensor.cpu()
#         y_test_tensor_cpu = y_test_tensor.cpu()
#         X_test_tensor_cpu = X_test_tensor.cpu()
#         x_reconstructed = invert_data_PCA(config, predictions_tensor_cpu, x)
#         [targets_final, targets_pred_final] = x_reconstructed
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

# def evaluate_model_PCA(config, model, wind_angles, output_folder, physics, vtk_output, X_test_tensor, y_test_tensor, x):

#     def extract_dataset_by_wind_angle(df, dataset_size, lower_bound, upper_bound):
#         for start_idx in range(0, len(df), dataset_size):
#             if df.iloc[start_idx]['WindAngle'] >= lower_bound and df.iloc[start_idx]['WindAngle'] <= upper_bound:
#                 return df.iloc[start_idx:start_idx+dataset_size]
#         return pd.DataFrame()

#     def compute_inverse_PCA_from_PCA(Z_reduced, pca, pca_reduce):
#         if pca_reduce == 'rows':
#             Z_r = (pca.inverse_transform((Z_reduced).T)).T
#         if pca_reduce == 'columns':
#             Z_r = (pca.inverse_transform((Z_reduced)))
#         return Z_r

#     def reshaped_array(arr,n):
#         num_rows, num_cols = arr.shape 
#         new_shape = (num_rows//n, num_cols*n)
#         reshaped_arr = arr.reshape(n, int(num_rows/n), num_cols).reshape(new_shape)
#         return reshaped_arr

#     def inverse_reshaped_array(arr, n):
#         num_rows, num_cols = arr.shape
#         new_shape = (num_rows * n, num_cols // n)
#         reshaped_arr = arr.reshape(n, int(num_rows), int(num_cols/n)).reshape(new_shape)
#         return reshaped_arr

#     def compute_mse(Z, Z_r, description=None):
#         mse = mean_squared_error(np.abs(Z), np.abs(Z_r))
#         print (f'mse: {mse} - {description}')
#         return mse

#     model.eval()
#     dfs = []
#     with torch.no_grad():
#         predictions_tensor = model(X_test_tensor)
#         predictions_tensor_cpu = predictions_tensor.cpu()
#         [reduced_features, pca_features, reduced_targets, pca_targets, features, targets, feature_scaler, target_scaler, pca_reduce] = x

#         features = inverse_reshaped_array(features, len(config["training"]["angles_to_train"]))
#         features[:, :3] = inverse_transform_features(features[:, :3], feature_scaler)
        
#         targets = inverse_reshaped_array(targets, len(config["training"]["angles_to_train"]))
#         targets = inverse_transform_targets(targets, target_scaler)

#         X_test_tensor_cpu = features
#         y_test_tensor_cpu = targets

#         predictions_tensor_cpu = compute_inverse_PCA_from_PCA(predictions_tensor_cpu, pca_targets, pca_reduce)
#         predictions_tensor_cpu = inverse_reshaped_array(predictions_tensor_cpu, len(config["training"]["angles_to_train"]))
#         predictions_tensor_cpu = inverse_transform_targets(predictions_tensor_cpu, target_scaler)
        
#         whole_tensor = np.hstack([X_test_tensor_cpu, y_test_tensor_cpu, predictions_tensor_cpu])

#         X_test_column_names = config["training"]["input_params_modf"]
#         output_column_names = config["training"]["output_params_modf"]
#         y_test_column_names = [item + "_Actual" for item in output_column_names]
#         predictions_column_names = [item + "_Predicted" for item in output_column_names]
#         whole_column_names = X_test_column_names + y_test_column_names + predictions_column_names

#         length_dataset = int(X_test_tensor_cpu.shape[0]/len(config["training"]["angles_to_train"]))

#         whole = pd.DataFrame(whole_tensor, columns=whole_column_names)
#         whole['Velocity_Magnitude_Actual'] = np.sqrt(whole['Velocity_X_Actual']**2 + whole['Velocity_Y_Actual']**2 + whole['Velocity_Z_Actual']**2)
#         whole['Velocity_Magnitude_Predicted'] = np.sqrt(whole['Velocity_X_Predicted']**2 + whole['Velocity_Y_Predicted']**2 + whole['Velocity_Z_Predicted']**2)
#         whole['WindAngle_rad'] = np.arctan2(whole['sin(WindAngle)'], whole['cos(WindAngle)'])
#         whole['WindAngle'] = np.abs(whole['WindAngle_rad'].apply(lambda x: int(np.ceil(np.degrees(x)))))
#         whole.sort_values(by='WindAngle', ascending=True, inplace=True)

#         for wind_angle in wind_angles:
#             lower_bound = wind_angle - 5
#             upper_bound = wind_angle + 5

#             filtered_whole = extract_dataset_by_wind_angle(whole, length_dataset, lower_bound, upper_bound)

#             predictions_column_names.append('Velocity_Magnitude_Predicted')
#             y_test_column_names.append('Velocity_Magnitude_Actual')
#             filtered_predictions = filtered_whole[predictions_column_names]
#             filtered_y_test = filtered_whole[y_test_column_names]
#             if len(filtered_predictions)!= 0 and len(filtered_y_test)!=0:
#                 rows_list = []
#                 for i, var in enumerate(y_test_column_names):
#                     var_cleaned = var.replace('_Actual', '')
#                     actuals = filtered_y_test.iloc[:, i]
#                     preds = filtered_predictions.iloc[:, i]
#                     mse = sklearn.metrics.mean_squared_error(actuals, preds)
#                     rmse = np.sqrt(mse)
#                     mae = sklearn.metrics.mean_absolute_error(actuals, preds)
#                     r2 = sklearn.metrics.r2_score(actuals, preds)
#                     rows_list.append({
#                         'Variable': var_cleaned, 
#                         'MSE': mse,
#                         'RMSE': rmse,
#                         'MAE': mae,
#                         'R2': r2
#                     })

#                 dfs.append(filtered_whole)

#                 data_folder = os.path.join(output_folder, f'data_output_for_wind_angle_{wind_angle}')
#                 os.makedirs(output_folder, exist_ok=True)
#                 os.makedirs(data_folder, exist_ok=True)

#                 combined_file_path = os.path.join(data_folder, f'combined_actuals_and_predictions_for_wind_angle_{wind_angle}.csv')
#                 metrics_file_path = os.path.join(data_folder, f'metrics_for_wind_angle_{wind_angle}.csv')
#                 metrics_df = pd.DataFrame(rows_list)
#                 metrics_df.to_csv(metrics_file_path, index=False)

#                 if config["plotting"]["save_csv_predictions"]:
#                     filtered_whole.to_csv(combined_file_path, index=False)
                       
#                 if vtk_output is not None and physics is not None and wind_angle in config["training"]["all_angles"]:
#                     if config["plotting"]["save_vtk"]:
#                         output_nn_to_vtk(config, wind_angle, f'{physics}_predictions_for_wind_angle_{wind_angle}', filtered_whole, predictions_column_names, vtk_output)
    
#     data = pd.concat(dfs)
#     return data
# ###PCA###

###PCA-STYLE-2###
def load_data_PCA(config, device):
    chosen_machine_key = config["chosen_machine"]
    datafolder_path = os.path.join(config["machine"][chosen_machine_key], config["data"]["data_folder_name"])
    filenames = get_filenames_from_folder(datafolder_path, config["data"]["extension"], config["data"]["startname_data"])
    training_wind_angles = config["training"]["angles_to_train"]
    skipped_wind_angles = config["training"]["angles_to_leave_out"]
    output_params = config["training"]["output_params"]
    input_params = config["training"]["input_params"]
    input_params_points = config["training"]["input_params_points"]
    angle_to_label = {angle: idx for idx, angle in enumerate(sorted(training_wind_angles))}

    # def get_scaled_df(df, input_params, output_params, feature_scaler, target_scaler):
    #     all_params = input_params + output_params
    #     print (all_params)
    #     df = df[all_params]
    #     print (df)
    #     features_targets = np.array(df)
    #     if feature_scaler is None:
    #         feature_scaler = config["training"]["feature_scaler"]
    #         feature_scaler.fit(features_targets[:, :3])
    #     if target_scaler is None:
    #         target_scaler = config["training"]["target_scaler"]
    #         target_scaler.fit(features_targets[:, -5:])

    #     features_targets[:, :3] = feature_scaler.transform(features_targets[:, :3])
    #     features_targets[:, -5:] = target_scaler.transform(features_targets[:, -5:])

    #     return features_targets, feature_scaler, target_scaler

    def get_scaled_features_targets(df, input_params, output_params, scaler):
        all_params = input_params + output_params
        df = df[all_params]
        features_targets = np.array(df)
        if scaler is None:
            scaler = config["training"]["feature_scaler"]
            scaler.fit(features_targets)
        features_targets = scaler.transform(features_targets)
        return features_targets, scaler

    def reshaped_array(arr,n):
        num_rows, num_cols = arr.shape 
        new_shape = (num_rows//n, num_cols*n)
        reshaped_arr = arr.reshape(n, int(num_rows/n), num_cols).reshape(new_shape)
        return reshaped_arr

    def inverse_reshaped_array(arr, n):
        num_rows, num_cols = arr.shape
        new_shape = (num_rows * n, num_cols // n)
        reshaped_arr = arr.reshape(n, int(num_rows), int(num_cols/n)).reshape(new_shape)
        return reshaped_arr

    def split_features_targets(arr, m1, m2):
        # Ensure the array width is a multiple of (m1 + m2)
        assert arr.shape[1] % (m1 + m2) == 0, "Array width must be a multiple of (m1 + m2)."
        
        K = arr.shape[1] // (m1 + m2)
        first_parts = []
        last_parts = []
        
        for k in range(K):
            start_index = k * (m1 + m2)
            first_part = arr[:, start_index:start_index + m1]
            last_part = arr[:, start_index + m1:start_index + m1 + m2]
            first_parts.append(first_part)
            last_parts.append(last_part)
        
        # Concatenate the first m1 and last m2 columns separately, then together
        features = np.hstack(first_parts)
        targets = np.hstack(last_parts)
        return features, targets

    def inverse_split_features_targets(features, targets, m1, m2, K):
        N = features.shape[0]
        # Assuming features and targets are the first m1 and last m2 columns, respectively
        original_arr = np.empty((N, K * (m1 + m2)), dtype=features.dtype)
        
        for k in range(K):
            # Extract the relevant sections from features and targets
            feature_part = features[:, k * m1:(k + 1) * m1]
            target_part = targets[:, k * m2:(k + 1) * m2]
            
            # Place the extracted parts back into their original positions
            original_arr[:, k * (m1 + m2):k * (m1 + m2) + m1] = feature_part
            original_arr[:, k * (m1 + m2) + m1:k * (m1 + m2) + m1 + m2] = target_part
        
        return original_arr

    def get_tensor(config, filenames, datafolder_path, wind_angles, angle_to_label, device, pca_reduce, scaler=None, pca=None):

        data, labels = concatenate_data_files(filenames, datafolder_path, wind_angles, angle_to_label)
        features_targets, scaler = get_scaled_features_targets(data, input_params, output_params, scaler)

        features_targets = reshaped_array(features_targets, len(wind_angles))
        reduced_features_targets, pca = compute_PCA(features_targets, pca_reduce, pca=pca)

        print ('I NEED TO CHECK NOW')
        features_targets_reconstructed = compute_inverse_PCA_from_PCA(reduced_features_targets, pca, 'rows')
        mse = ((features_targets - features_targets_reconstructed) ** 2).mean(axis=0)
        for i, mse_value in enumerate(mse, 1):
            print(f"Column {i} MSE: {mse_value}")
        print ('I HAVE CHECK NOW')

        reduced_features, reduced_targets = split_features_targets(reduced_features_targets, len(input_params), len(output_params))
        features, targets = split_features_targets(features_targets, len(input_params), len(output_params))

        x = [reduced_features, reduced_targets, features, targets, pca, scaler]

        return x 

    def compute_PCA(Z, pca_reduce, pca=None, n_components=None):
        if pca is None:
            print ('fitting')
            pca = PCA(n_components, svd_solver='full')
            if pca_reduce == 'rows':
                Z_reduced = (pca.fit_transform(Z.T)).T
            elif pca_reduce == 'columns':
                Z_reduced = (pca.fit_transform(Z))
            eigenvalues_ratio = pca.explained_variance_ratio_
            print(f"Explained variance ratio of the first {len(eigenvalues_ratio)} principal components: {eigenvalues_ratio} w sum = {np.sum(eigenvalues_ratio)} w reduced matrix {Z_reduced.shape}")
        else:
            print ('im not fitting')
            if pca_reduce == 'rows':
                Z_reduced = (pca.transform(Z.T)).T
            elif pca_reduce == 'columns':
                Z_reduced = (pca.transform(Z))
        Z_r = compute_inverse_PCA_from_PCA(Z_reduced, pca, pca_reduce)
        compute_mse(Z, Z_r, 'inverse of computed PCA')
        return Z_reduced, pca

    def compute_inverse_PCA_from_PCA(Z_reduced, pca, pca_reduce):
            if pca_reduce == 'rows':
                Z_r = (pca.inverse_transform((Z_reduced).T)).T
            if pca_reduce == 'columns':
                Z_r = (pca.inverse_transform((Z_reduced)))
            return Z_r

    def get_skipped_angles(old_skipped, fixed_training):
        some_temp_shit = []
        for i in fixed_training:
            some_temp_shit.append(i)
        for val in old_skipped:
            index_of_closest = np.argmin(np.abs(np.array(some_temp_shit) - val))
            some_temp_shit[index_of_closest] = val

        skipped_angle_to_label = {angle: idx for idx, angle in enumerate(sorted(some_temp_shit))}
        return some_temp_shit, skipped_angle_to_label

    def compute_mse(Z, Z_r, description=None):
        mse = mean_squared_error(np.abs(Z), np.abs(Z_r))
        print (f'mse: {mse} - {description}')
        return mse

    pca_reduce = 'rows'
    x_training = get_tensor(config, filenames, datafolder_path, training_wind_angles, angle_to_label, device, pca_reduce)

    some_temp = training_wind_angles
    skipped_wind_angles, skipped_angle_to_label = get_skipped_angles(skipped_wind_angles, some_temp)

    print ('starting skipped', skipped_wind_angles)

    x_skipped = get_tensor(config, filenames, datafolder_path, skipped_wind_angles, skipped_angle_to_label, device, pca_reduce, x_training[5], x_training[4])

    X_train_tensor = (torch.tensor(np.array(x_training[0]), dtype=torch.float32)).to(device)
    y_train_tensor = (torch.tensor(np.array(x_training[1]), dtype=torch.float32)).to(device)

    X_test_tensor_skipped = (torch.tensor(np.array(x_skipped[0]), dtype=torch.float32)).to(device)
    y_test_tensor_skipped = (torch.tensor(np.array(x_skipped[1]), dtype=torch.float32)).to(device)

    x_training.append(pca_reduce)
    x_skipped.append(pca_reduce)

    print (X_train_tensor.shape, y_train_tensor.shape, X_test_tensor_skipped.shape, y_test_tensor_skipped.shape)

    data_dict = {
        "X_train_tensor": X_train_tensor,
        "y_train_tensor": y_train_tensor,
        "X_test_tensor_skipped": X_test_tensor_skipped,
        "y_test_tensor_skipped": y_test_tensor_skipped,
        "relevant_data_training": x_training,
        "relevant_data_skipped": x_skipped
        }
    
    return data_dict

def invert_data_PCA(config, pred, x):

    def compute_inverse_PCA_from_PCA(Z_reduced, pca, pca_reduce):
        if pca_reduce == 'rows':
            Z_r = (pca.inverse_transform((Z_reduced).T)).T
        if pca_reduce == 'columns':
            Z_r = (pca.inverse_transform((Z_reduced)))
        return Z_r

    def compute_mse(Z, Z_r, description=None):
        mse = mean_squared_error(np.abs(Z), np.abs(Z_r))
        print (f'mse: {mse} - {description}')
        return mse

    [reduced_features, reduced_targets, features, targets, pca, scaler, pca_reduce] = x

    def inverse_reshaped_array(arr, n):
        num_rows, num_cols = arr.shape
        new_shape = (num_rows * n, num_cols // n)
        reshaped_arr = arr.reshape(n, int(num_rows), int(num_cols/n)).reshape(new_shape)
        return reshaped_arr

    def inverse_split_features_targets(features, targets, m1, m2, K):
        N = features.shape[0]
        # Assuming features and targets are the first m1 and last m2 columns, respectively
        original_arr = np.empty((N, K * (m1 + m2)), dtype=features.dtype)
        
        for k in range(K):
            # Extract the relevant sections from features and targets
            feature_part = features[:, k * m1:(k + 1) * m1]
            target_part = targets[:, k * m2:(k + 1) * m2]
            
            # Place the extracted parts back into their original positions
            original_arr[:, k * (m1 + m2):k * (m1 + m2) + m1] = feature_part
            original_arr[:, k * (m1 + m2) + m1:k * (m1 + m2) + m1 + m2] = target_part
        
        return original_arr

    def split_features_targets(arr, m1, m2):
        # Ensure the array width is a multiple of (m1 + m2)
        assert arr.shape[1] % (m1 + m2) == 0, "Array width must be a multiple of (m1 + m2)."
        
        K = arr.shape[1] // (m1 + m2)
        first_parts = []
        last_parts = []
        
        for k in range(K):
            start_index = k * (m1 + m2)
            first_part = arr[:, start_index:start_index + m1]
            last_part = arr[:, start_index + m1:start_index + m1 + m2]
            first_parts.append(first_part)
            last_parts.append(last_part)
        
        # Concatenate the first m1 and last m2 columns separately, then together
        features = np.hstack(first_parts)
        targets = np.hstack(last_parts)
        return features, targets

    features_targets = inverse_split_features_targets(features, targets, len(config["training"]["input_params"]), len(config["training"]["output_params"]), len(config["training"]["angles_to_train"]))
    features_preds = inverse_split_features_targets(reduced_features, pred, len(config["training"]["input_params"]), len(config["training"]["output_params"]), len(config["training"]["angles_to_train"]))

    print (features_targets.shape, features_preds.shape)

    features_preds_reconstructed = compute_inverse_PCA_from_PCA(features_preds, pca, pca_reduce)

    print (features_targets.shape, features_preds_reconstructed.shape)

    features_preds_reconstructed = inverse_reshaped_array(features_preds_reconstructed, len(config["training"]["angles_to_train"]))
    features_targets = inverse_reshaped_array(features_targets, len(config["training"]["angles_to_train"]))

    print (features_targets.shape, features_preds_reconstructed.shape)

    features_preds_reconstructed = inverse_transform_targets(features_preds_reconstructed, scaler)
    features_targets = inverse_transform_targets(features_targets, scaler)

    print (features_targets.shape, features_preds_reconstructed.shape)

    features, targets = split_features_targets(features_targets, len(config["training"]["input_params"]), len(config["training"]["output_params"]))
    features, preds_reconstructed = split_features_targets(features_preds_reconstructed, len(config["training"]["input_params"]), len(config["training"]["output_params"]))

    assert targets.shape == preds_reconstructed.shape
    x_reconstructed = [targets, preds_reconstructed]
    compute_mse(targets, preds_reconstructed, description='TARGETS vs PRED MSE inv FFT inv PCA')
    return x_reconstructed

def evaluate_model_training_PCA(config, device, model, model_file_path, X_test_tensor, y_test_tensor, x):
    checkpoint = torch.load(model_file_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    with torch.no_grad():
        predictions_tensor = model(X_test_tensor)
        predictions_tensor_cpu = predictions_tensor.cpu()
        y_test_tensor_cpu = y_test_tensor.cpu()
        X_test_tensor_cpu = X_test_tensor.cpu()
        x_reconstructed = invert_data_PCA(config, predictions_tensor_cpu, x)
        [targets_final, targets_pred_final] = x_reconstructed
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

def evaluate_model_PCA(config, model, wind_angles, output_folder, physics, vtk_output, X_test_tensor, y_test_tensor, x):

    def extract_dataset_by_wind_angle(df, dataset_size, lower_bound, upper_bound):
        for start_idx in range(0, len(df), dataset_size):
            if df.iloc[start_idx]['WindAngle'] >= lower_bound and df.iloc[start_idx]['WindAngle'] <= upper_bound:
                return df.iloc[start_idx:start_idx+dataset_size]
        return pd.DataFrame()

    def compute_inverse_PCA_from_PCA(Z_reduced, pca, pca_reduce):
        if pca_reduce == 'rows':
            Z_r = (pca.inverse_transform((Z_reduced).T)).T
        if pca_reduce == 'columns':
            Z_r = (pca.inverse_transform((Z_reduced)))
        return Z_r

    def inverse_reshaped_array(arr, n):
        num_rows, num_cols = arr.shape
        new_shape = (num_rows * n, num_cols // n)
        reshaped_arr = arr.reshape(n, int(num_rows), int(num_cols/n)).reshape(new_shape)
        return reshaped_arr

    def inverse_split_features_targets(features, targets, m1, m2, K):
        N = features.shape[0]
        # Assuming features and targets are the first m1 and last m2 columns, respectively
        original_arr = np.empty((N, K * (m1 + m2)), dtype=features.dtype)
        
        for k in range(K):
            # Extract the relevant sections from features and targets
            feature_part = features[:, k * m1:(k + 1) * m1]
            target_part = targets[:, k * m2:(k + 1) * m2]
            
            # Place the extracted parts back into their original positions
            original_arr[:, k * (m1 + m2):k * (m1 + m2) + m1] = feature_part
            original_arr[:, k * (m1 + m2) + m1:k * (m1 + m2) + m1 + m2] = target_part
        
        return original_arr

    def compute_mse(Z, Z_r, description=None):
        mse = mean_squared_error(np.abs(Z), np.abs(Z_r))
        print (f'mse: {mse} - {description}')
        return mse

    def split_features_targets(arr, m1, m2):
        # Ensure the array width is a multiple of (m1 + m2)
        assert arr.shape[1] % (m1 + m2) == 0, "Array width must be a multiple of (m1 + m2)."
        
        K = arr.shape[1] // (m1 + m2)
        first_parts = []
        last_parts = []
        
        for k in range(K):
            start_index = k * (m1 + m2)
            first_part = arr[:, start_index:start_index + m1]
            last_part = arr[:, start_index + m1:start_index + m1 + m2]
            first_parts.append(first_part)
            last_parts.append(last_part)
        
        # Concatenate the first m1 and last m2 columns separately, then together
        features = np.hstack(first_parts)
        targets = np.hstack(last_parts)
        return features, targets

    model.eval()
    dfs = []
    with torch.no_grad():
        predictions_tensor = model(X_test_tensor)
        predictions_tensor_cpu = predictions_tensor.cpu()
        [reduced_features, reduced_targets, features, targets, pca, scaler, pca_reduce] = x


        features_targets = inverse_split_features_targets(features, targets, len(config["training"]["input_params"]), len(config["training"]["output_params"]), len(config["training"]["angles_to_train"]))
        features_preds = inverse_split_features_targets(reduced_features, predictions_tensor_cpu, len(config["training"]["input_params"]), len(config["training"]["output_params"]), len(config["training"]["angles_to_train"]))

        print (features_targets.shape, features_preds.shape)

        features_preds_reconstructed = compute_inverse_PCA_from_PCA(features_preds, pca, pca_reduce)

        print (features_targets.shape, features_preds_reconstructed.shape)

        features_preds_reconstructed = inverse_reshaped_array(features_preds_reconstructed, len(config["training"]["angles_to_train"]))
        features_targets = inverse_reshaped_array(features_targets, len(config["training"]["angles_to_train"]))

        print (features_targets.shape, features_preds_reconstructed.shape)

        features_preds_reconstructed = inverse_transform_targets(features_preds_reconstructed, scaler)
        features_targets = inverse_transform_targets(features_targets, scaler)

        print (features_targets.shape, features_preds_reconstructed.shape)

        X_test_tensor_cpu, y_test_tensor_cpu = split_features_targets(features_targets, len(config["training"]["input_params"]), len(config["training"]["output_params"]))
        _, predictions_tensor_cpu = split_features_targets(features_preds_reconstructed, len(config["training"]["input_params"]), len(config["training"]["output_params"]))
        
        whole_tensor = np.hstack([X_test_tensor_cpu, y_test_tensor_cpu, predictions_tensor_cpu])

        X_test_column_names = config["training"]["input_params_modf"]
        output_column_names = config["training"]["output_params_modf"]
        y_test_column_names = [item + "_Actual" for item in output_column_names]
        predictions_column_names = [item + "_Predicted" for item in output_column_names]
        whole_column_names = X_test_column_names + y_test_column_names + predictions_column_names

        length_dataset = int(X_test_tensor_cpu.shape[0]/len(config["training"]["angles_to_train"]))

        whole = pd.DataFrame(whole_tensor, columns=whole_column_names)
        whole['Velocity_Magnitude_Actual'] = np.sqrt(whole['Velocity_X_Actual']**2 + whole['Velocity_Y_Actual']**2 + whole['Velocity_Z_Actual']**2)
        whole['Velocity_Magnitude_Predicted'] = np.sqrt(whole['Velocity_X_Predicted']**2 + whole['Velocity_Y_Predicted']**2 + whole['Velocity_Z_Predicted']**2)
        whole['WindAngle_rad'] = np.arctan2(whole['sin(WindAngle)'], whole['cos(WindAngle)'])
        whole['WindAngle'] = np.abs(whole['WindAngle_rad'].apply(lambda x: int(np.ceil(np.degrees(x)))))
        whole.sort_values(by='WindAngle', ascending=True, inplace=True)

        for wind_angle in wind_angles:
            lower_bound = wind_angle - 5
            upper_bound = wind_angle + 5

            filtered_whole = extract_dataset_by_wind_angle(whole, length_dataset, lower_bound, upper_bound)

            predictions_column_names.append('Velocity_Magnitude_Predicted')
            y_test_column_names.append('Velocity_Magnitude_Actual')
            filtered_predictions = filtered_whole[predictions_column_names]
            filtered_y_test = filtered_whole[y_test_column_names]
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

                dfs.append(filtered_whole)

                data_folder = os.path.join(output_folder, f'data_output_for_wind_angle_{wind_angle}')
                os.makedirs(output_folder, exist_ok=True)
                os.makedirs(data_folder, exist_ok=True)

                combined_file_path = os.path.join(data_folder, f'combined_actuals_and_predictions_for_wind_angle_{wind_angle}.csv')
                metrics_file_path = os.path.join(data_folder, f'metrics_for_wind_angle_{wind_angle}.csv')
                metrics_df = pd.DataFrame(rows_list)
                metrics_df.to_csv(metrics_file_path, index=False)

                if config["plotting"]["save_csv_predictions"]:
                    filtered_whole.to_csv(combined_file_path, index=False)
                       
                if vtk_output is not None and physics is not None and wind_angle in config["training"]["all_angles"]:
                    if config["plotting"]["save_vtk"]:
                        output_nn_to_vtk(config, wind_angle, f'{physics}_predictions_for_wind_angle_{wind_angle}', filtered_whole, predictions_column_names, vtk_output)
    
    data = pd.concat(dfs)
    return data
###PCA-STYLE-2###

# ###FFT###
# def load_data_fft(config, device):
#     chosen_machine_key = config["chosen_machine"]
#     datafolder_path = os.path.join(config["machine"][chosen_machine_key], config["data"]["data_folder_name"])
#     filenames = get_filenames_from_folder(datafolder_path, config["data"]["extension"], config["data"]["startname_data"])
#     training_wind_angles = config["training"]["angles_to_train"]
#     skipped_wind_angles = config["training"]["angles_to_leave_out"]
#     output_params = config["training"]["output_params"]
#     input_params = config["training"]["input_params"]
#     input_params_points = config["training"]["input_params_points"]
#     angle_to_label = {angle: idx for idx, angle in enumerate(sorted(training_wind_angles))}

#     def get_features(df, input_params, feature_scaler):
#         df = df[input_params]
#         features = np.array(df)
#         if feature_scaler is None:
#             feature_scaler = config["training"]["feature_scaler"]
#             feature_scaler.fit(features[:, :3])
#         features[:, :3] = feature_scaler.transform(features[:, :3])
#         return features, feature_scaler

#     def get_targets(df, output_params, target_scaler):
#         targets = df[output_params]
#         if target_scaler is None:
#             target_scaler = config["training"]["target_scaler"]
#             target_scaler.fit(targets)
#         targets = target_scaler.transform(targets)
#         return targets, target_scaler

#     def reshaped_array(arr,n):
#         num_rows, num_cols = arr.shape 
#         new_shape = (num_rows//n, num_cols*n)
#         reshaped_arr = arr.reshape(n, int(num_rows/n), num_cols).reshape(new_shape)
#         return reshaped_arr

#     def inverse_reshaped_array(arr, n):
#         num_rows, num_cols = arr.shape
#         new_shape = (num_rows * n, num_cols // n)
#         reshaped_arr = arr.reshape(n, int(num_rows), int(num_cols/n)).reshape(new_shape)
#         return reshaped_arr

#     def compute_mse(Z, Z_r, description=None):
#         mse = mean_squared_error(np.abs(Z), np.abs(Z_r))
#         print (f'mse: {mse} - {description}')
#         return mse

#     def compute_PCA(A):
#         data_dask = da.from_array(A, chunks=A.shape)
#         u, s, v = svd(data_dask)
#         U, s, Vh = da.compute(u, s, v)

#         V = Vh.conjugate().T
#         variance_explained = s**2 / np.sum(s**2)
#         projected_data = np.dot(A, V)

#         print(f"SVD: U - {U.shape}, s - {s.shape}, V_dagger - {Vh.shape}")
#         print("Variance explained by each principal component:", variance_explained)
#         print("Projected data:\n", projected_data.shape)

#         return projected_data, Vh

#     def apply_PCA(A, Vh):
#         V = Vh.conjugate().T
#         projected_data = np.dot(A, V)
#         return projected_data, Vh

#     def compute_inverse_PCA(A, Vh):
#         reconstructed_data = np.dot(A, Vh)
#         return reconstructed_data

#     def compute_fft_pca(config, filenames, datafolder_path, wind_angles, angle_to_label, feature_scaler=None, target_scaler=None, features_fft_eigenvectors=None, targets_fft_eigenvectors=None):
#         data, labels = concatenate_data_files(filenames, datafolder_path, wind_angles, angle_to_label)
#         features, feature_scaler = get_features(data, input_params, feature_scaler)
#         targets, target_scaler = get_targets(data, output_params, target_scaler)

#         features = reshaped_array(features, len(wind_angles))
#         targets = reshaped_array(targets, len(wind_angles))

#         features, targets = features.T, targets.T

#         features_fft = np.fft.rfftn(features) #["X", "Y", "Z", "cos(WindAngle)", "sin(WindAngle)"]
#         targets_fft = np.fft.rfftn(targets) #['Pressure', 'Velocity_X', 'Velocity_Y', 'Velocity_Z', 'TurbVisc']

#         print (f'FEATURES before FFT - {features.shape} ; after FFT - {features_fft.shape}')
#         print (f'TARGETS before FFT - {targets.shape} ; after FFT - {targets_fft.shape}')

#         if features_fft_eigenvectors is None:
#             features_fft_pca, features_fft_eigenvectors = compute_PCA(features_fft)
#             targets_fft_pca, targets_fft_eigenvectors = compute_PCA(targets_fft)
#         else:
#             features_fft_pca, features_fft_eigenvectors = apply_PCA(features_fft, features_fft_eigenvectors)
#             targets_fft_pca, targets_fft_eigenvectors = apply_PCA(targets_fft, targets_fft_eigenvectors)

#         print (f'FEATURES before FFT - {features.shape} ; after FFT, PCA - {features_fft_pca.shape}')
#         print (f'TARGETS before FFT - {targets.shape} ; after FFT, PCA - {targets_fft_pca.shape}')

#         x = [features_fft, features_fft_pca, features_fft_eigenvectors, targets_fft, targets_fft_pca, targets_fft_eigenvectors, features, targets, feature_scaler, target_scaler]

#         return x

#     def compute_inv_fft_inv_pca(x):

#         [features_fft, features_fft_pca, features_fft_eigenvectors, targets_fft, targets_fft_pca, targets_fft_eigenvectors, features, targets, feature_scaler, target_scaler] = x

#         features_fft_reconstructed = compute_inverse_PCA(features_fft_pca, features_fft_eigenvectors)
#         targets_fft_reconstructed = compute_inverse_PCA(targets_fft_pca, targets_fft_eigenvectors)

#         print (f'FEATURES before FFT - {features.shape} ; after FFT, inverse PCA - {features_fft_reconstructed.shape}')
#         print (f'TARGETS before FFT - {targets.shape} ; after FFT, inverse PCA - {targets_fft_reconstructed.shape}')

#         compute_mse(features_fft, features_fft_reconstructed, description='FEATURES MSE FFT inv PCA')
#         compute_mse(targets_fft, targets_fft_reconstructed, description='TARGETS MSE FFT inv PCA')

#         features_reconstructed = np.fft.irfftn(features_fft, s=features.shape)
#         targets_reconstructed = np.fft.irfftn(targets_fft, s=targets.shape)

#         print (f'FEATURES before FFT - {features.shape} ; after inv FFT reconstructed - {features_reconstructed.shape}')
#         print (f'TARGETS before FFT - {targets.shape} ; after inv FFT reconstructed - {targets_reconstructed.shape}')

#         compute_mse(features, features_reconstructed, description='FEATURES MSE inv FFT')
#         compute_mse(targets, targets_reconstructed, description='TARGETS MSE inv FFT')

#         features_reconstructed = np.fft.irfftn(features_fft_reconstructed, s=features.shape)
#         targets_reconstructed = np.fft.irfftn(targets_fft_reconstructed, s=targets.shape)

#         print (f'FEATURES before FFT - {features.shape} ; after inv FFT reconstructed - {features_reconstructed.shape}')
#         print (f'TARGETS before FFT - {targets.shape} ; after inv FFT reconstructed - {targets_reconstructed.shape}')

#         x_reconstructed = [features_reconstructed, targets_reconstructed]

#         return x_reconstructed

#     def get_skipped_angles(old_skipped, fixed_training):
#         some_temp_shit = []
#         for i in fixed_training:
#             some_temp_shit.append(i)
#         for val in old_skipped:
#             index_of_closest = np.argmin(np.abs(np.array(some_temp_shit) - val))
#             some_temp_shit[index_of_closest] = val

#         skipped_angle_to_label = {angle: idx for idx, angle in enumerate(sorted(some_temp_shit))}
#         return some_temp_shit, skipped_angle_to_label

#     x_training = compute_fft_pca(config, filenames, datafolder_path, training_wind_angles, angle_to_label)
#     x_training_reconstructed = compute_inv_fft_inv_pca(x_training)
#     compute_mse(x_training[6], x_training_reconstructed[0], description='FEATURES TRAINING MSE inv FFT inv PCA')
#     compute_mse(x_training[7], x_training_reconstructed[1], description='TARGETS TRAINING MSE inv FFT inv PCA')

#     skipped_wind_angles, skipped_angle_to_label = get_skipped_angles(skipped_wind_angles, training_wind_angles)
#     print ('skipped', skipped_wind_angles)

#     x_skipped = compute_fft_pca(config, filenames, datafolder_path, skipped_wind_angles, skipped_angle_to_label, x_training[8], x_training[9], x_training[2], x_training[5])
#     x_skipped_reconstructed = compute_inv_fft_inv_pca(x_skipped)
#     compute_mse(x_skipped[6], x_skipped_reconstructed[0], description='FEATURES SKIPPED MSE inv FFT inv PCA')
#     compute_mse(x_skipped[7], x_skipped_reconstructed[1], description='TARGETS SKIPPED MSE inv FFT inv PCA')

#     def to_real_imag(tensor_complex):
#         real_part = tensor_complex.real
#         imag_part = tensor_complex.imag
#         return np.stack((real_part, imag_part), axis=-1)

#     X_train_real_imag = to_real_imag(np.array(x_training[1]))
#     y_train_real_imag = to_real_imag(np.array(x_training[4]))

#     X_train_tensor = torch.tensor(X_train_real_imag, dtype=torch.float32).to(device)
#     y_train_tensor = torch.tensor(y_train_real_imag, dtype=torch.float32).to(device)

#     X_test_real_imag_skipped = to_real_imag(np.array(x_skipped[1]))
#     y_test_real_imag_skipped = to_real_imag(np.array(x_skipped[4]))

#     X_test_tensor_skipped = torch.tensor(X_test_real_imag_skipped, dtype=torch.float32).to(device)
#     y_test_tensor_skipped = torch.tensor(y_test_real_imag_skipped, dtype=torch.float32).to(device)

#     original_shape = X_train_tensor.shape

#     X_train_tensor = X_train_tensor.view(X_train_tensor.size(0), -1)
#     y_train_tensor = y_train_tensor.view(y_train_tensor.size(0), -1)
#     X_test_tensor_skipped = X_test_tensor_skipped.view(X_test_tensor_skipped.size(0), -1)
#     y_test_tensor_skipped = y_test_tensor_skipped.view(y_test_tensor_skipped.size(0), -1)

#     x_training.append(original_shape)
#     x_skipped.append(original_shape)

#     data_dict = {
#         "X_train_tensor": X_train_tensor,
#         "y_train_tensor": y_train_tensor,
#         "X_test_tensor_skipped": X_test_tensor_skipped,
#         "y_test_tensor_skipped": y_test_tensor_skipped,
#         "relevant_data_training": x_training,
#         "relevant_data_skipped": x_skipped
#         }

#     return data_dict

# def evaluate_model_training_fft(config, device, model, model_file_path, X_test_tensor, y_test_tensor, x):
#     checkpoint = torch.load(model_file_path, map_location=device)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.eval()
#     with torch.no_grad():
#         predictions_tensor = model(X_test_tensor)
#         predictions_tensor_cpu = predictions_tensor.cpu()
#         y_test_tensor_cpu = y_test_tensor.cpu()
#         X_test_tensor_cpu = X_test_tensor.cpu()
#         x_reconstructed = compute_inv_fft_inv_pca(config, y_test_tensor_cpu, predictions_tensor_cpu, x)
#         [targets_final, targets_pred_final] = x_reconstructed
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

# def compute_inv_fft_inv_pca(config, y, pred, x):

#     def compute_inverse_PCA(A, Vh):
#         reconstructed_data = np.dot(A, Vh)
#         return reconstructed_data

#     def compute_mse(Z, Z_r, description=None):
#         mse = mean_squared_error(np.abs(Z), np.abs(Z_r))
#         print (f'mse: {mse} - {description}')
#         return mse

#     def from_real_imag(real_imag_array):
#         real_part = real_imag_array[..., 0]
#         imag_part = real_imag_array[..., 1]
    
#         complex_array = real_part + 1j * imag_part
#         return complex_array

#     def inverse_reshaped_array(arr, n):
#         num_rows, num_cols = arr.shape
#         new_shape = (num_rows * n, num_cols // n)
#         reshaped_arr = arr.reshape(n, int(num_rows), int(num_cols/n)).reshape(new_shape)
#         return reshaped_arr

#     [features_fft, features_fft_pca, features_fft_eigenvectors, targets_fft, targets_fft_pca, targets_fft_eigenvectors, features, targets, feature_scaler, target_scaler, original_shape] = x

#     pred = pred.view(original_shape)
#     pred = from_real_imag(pred)

#     pred_fft_reconstructed = compute_inverse_PCA(pred, targets_fft_eigenvectors)

#     pred_reconstructed = np.fft.irfftn(pred_fft_reconstructed, s=targets.shape)

#     pred_reconstructed = pred_reconstructed.T
#     targets = targets.T

#     pred_reconstructed = inverse_reshaped_array(pred_reconstructed, len(config["training"]["angles_to_train"]))
#     targets = inverse_reshaped_array(targets, len(config["training"]["angles_to_train"]))

#     pred_reconstructed = inverse_transform_targets(pred_reconstructed, target_scaler)
#     targets = inverse_transform_targets(targets, target_scaler)

#     assert targets.shape == pred_reconstructed.shape
#     x_reconstructed = [targets, pred_reconstructed]
#     compute_mse(targets, pred_reconstructed, description='TARGETS vs PRED MSE inv FFT inv PCA')
#     return x_reconstructed

# def evaluate_model_fft(config, model, wind_angles, output_folder, physics, vtk_output, X_test_tensor, y_test_tensor, x):

#     def extract_dataset_by_wind_angle(df, dataset_size, lower_bound, upper_bound):
#         for start_idx in range(0, len(df), dataset_size):
#             if df.iloc[start_idx]['WindAngle'] >= lower_bound and df.iloc[start_idx]['WindAngle'] <= upper_bound:
#                 return df.iloc[start_idx:start_idx+dataset_size]
#         return pd.DataFrame()

#     def compute_inverse_PCA_from_PCA(Z_reduced, pca, pca_reduce):
#         if pca_reduce == 'rows':
#             Z_r = (pca.inverse_transform((Z_reduced).T)).T
#         if pca_reduce == 'columns':
#             Z_r = (pca.inverse_transform((Z_reduced)))
#         return Z_r

#     def reshaped_array(arr,n):
#         num_rows, num_cols = arr.shape 
#         new_shape = (num_rows//n, num_cols*n)
#         reshaped_arr = arr.reshape(n, int(num_rows/n), num_cols).reshape(new_shape)
#         return reshaped_arr

#     def inverse_reshaped_array(arr, n):
#         num_rows, num_cols = arr.shape
#         new_shape = (num_rows * n, num_cols // n)
#         reshaped_arr = arr.reshape(n, int(num_rows), int(num_cols/n)).reshape(new_shape)
#         return reshaped_arr

#     def compute_mse(Z, Z_r, description=None):
#         mse = mean_squared_error(np.abs(Z), np.abs(Z_r))
#         print (f'mse: {mse} - {description}')
#         return mse

#     model.eval()
#     dfs = []
#     with torch.no_grad():
#         predictions_tensor = model(X_test_tensor)
#         predictions_tensor_cpu = predictions_tensor.cpu()
#         [features_fft, features_fft_pca, features_fft_eigenvectors, targets_fft, targets_fft_pca, targets_fft_eigenvectors, features, targets, feature_scaler, target_scaler, original_shape] = x

#         features = features.T
#         features = inverse_reshaped_array(features, len(config["training"]["angles_to_train"]))
#         features[:, :3] = inverse_transform_features(features[:, :3], feature_scaler)
        
#         # targets = inverse_reshaped_array(targets, len(config["training"]["angles_to_train"]))
#         # targets = inverse_transform_targets(targets, target_scaler)

        

#         # predictions_tensor_cpu = compute_inverse_PCA_from_PCA(predictions_tensor_cpu, pca_targets, pca_reduce)
#         # predictions_tensor_cpu = inverse_reshaped_array(predictions_tensor_cpu, len(config["training"]["angles_to_train"]))
#         # predictions_tensor_cpu = inverse_transform_targets(predictions_tensor_cpu, target_scaler)
        
#         [targets, predictions_tensor_cpu] = compute_inv_fft_inv_pca(config, targets, predictions_tensor_cpu, x)
        
#         X_test_tensor_cpu = features
#         y_test_tensor_cpu = targets

#         whole_tensor = np.hstack([X_test_tensor_cpu, y_test_tensor_cpu, predictions_tensor_cpu])

#         X_test_column_names = config["training"]["input_params_modf"]
#         output_column_names = config["training"]["output_params_modf"]
#         y_test_column_names = [item + "_Actual" for item in output_column_names]
#         predictions_column_names = [item + "_Predicted" for item in output_column_names]
#         whole_column_names = X_test_column_names + y_test_column_names + predictions_column_names

#         length_dataset = int(X_test_tensor_cpu.shape[0]/len(config["training"]["angles_to_train"]))

#         whole = pd.DataFrame(whole_tensor, columns=whole_column_names)
#         whole['Velocity_Magnitude_Actual'] = np.sqrt(whole['Velocity_X_Actual']**2 + whole['Velocity_Y_Actual']**2 + whole['Velocity_Z_Actual']**2)
#         whole['Velocity_Magnitude_Predicted'] = np.sqrt(whole['Velocity_X_Predicted']**2 + whole['Velocity_Y_Predicted']**2 + whole['Velocity_Z_Predicted']**2)
#         whole['WindAngle_rad'] = np.arctan2(whole['sin(WindAngle)'], whole['cos(WindAngle)'])
#         whole['WindAngle'] = np.abs(whole['WindAngle_rad'].apply(lambda x: int(np.ceil(np.degrees(x)))))
#         whole.sort_values(by='WindAngle', ascending=True, inplace=True)

#         for wind_angle in wind_angles:
#             lower_bound = wind_angle - 5
#             upper_bound = wind_angle + 5

#             filtered_whole = extract_dataset_by_wind_angle(whole, length_dataset, lower_bound, upper_bound)

#             predictions_column_names.append('Velocity_Magnitude_Predicted')
#             y_test_column_names.append('Velocity_Magnitude_Actual')
#             filtered_predictions = filtered_whole[predictions_column_names]
#             filtered_y_test = filtered_whole[y_test_column_names]
#             if len(filtered_predictions)!= 0 and len(filtered_y_test)!=0:
#                 rows_list = []
#                 for i, var in enumerate(y_test_column_names):
#                     var_cleaned = var.replace('_Actual', '')
#                     actuals = filtered_y_test.iloc[:, i]
#                     preds = filtered_predictions.iloc[:, i]
#                     mse = sklearn.metrics.mean_squared_error(actuals, preds)
#                     rmse = np.sqrt(mse)
#                     mae = sklearn.metrics.mean_absolute_error(actuals, preds)
#                     r2 = sklearn.metrics.r2_score(actuals, preds)
#                     rows_list.append({
#                         'Variable': var_cleaned, 
#                         'MSE': mse,
#                         'RMSE': rmse,
#                         'MAE': mae,
#                         'R2': r2
#                     })

#                 dfs.append(filtered_whole)

#                 data_folder = os.path.join(output_folder, f'data_output_for_wind_angle_{wind_angle}')
#                 os.makedirs(output_folder, exist_ok=True)
#                 os.makedirs(data_folder, exist_ok=True)

#                 combined_file_path = os.path.join(data_folder, f'combined_actuals_and_predictions_for_wind_angle_{wind_angle}.csv')
#                 metrics_file_path = os.path.join(data_folder, f'metrics_for_wind_angle_{wind_angle}.csv')
#                 metrics_df = pd.DataFrame(rows_list)
#                 metrics_df.to_csv(metrics_file_path, index=False)

#                 if config["plotting"]["save_csv_predictions"]:
#                     filtered_whole.to_csv(combined_file_path, index=False)
                       
#                 if vtk_output is not None and physics is not None and wind_angle in config["training"]["all_angles"]:
#                     if config["plotting"]["save_vtk"]:
#                         output_nn_to_vtk(config, wind_angle, f'{physics}_predictions_for_wind_angle_{wind_angle}', filtered_whole, predictions_column_names, vtk_output)
    
#     data = pd.concat(dfs)
#     return data
# ###FFT###

###FFT-STYLE-2###
def get_scaled_features_targets(config, df, input_params, output_params, scaler):
    all_params = input_params + output_params
    df = df[all_params]
    features_targets = np.array(df)
    if scaler is None:
        scaler = config["training"]["feature_scaler"]
        scaler.fit(features_targets)
    features_targets = scaler.transform(features_targets)
    return features_targets, scaler

def reshaped_array(arr,n):
    num_rows, num_cols = arr.shape 
    new_shape = (num_rows//n, num_cols*n)
    reshaped_arr = arr.reshape(n, int(num_rows/n), num_cols).reshape(new_shape)
    return reshaped_arr

def inverse_reshaped_array(arr, n):
    num_rows, num_cols = arr.shape
    new_shape = (num_rows * n, num_cols // n)
    reshaped_arr = arr.reshape(n, int(num_rows), int(num_cols/n)).reshape(new_shape)
    return reshaped_arr

def split_features_targets(arr, m1, m2):
    # Ensure the array width is a multiple of (m1 + m2)
    assert arr.shape[1] % (m1 + m2) == 0, "Array width must be a multiple of (m1 + m2)."
    
    K = arr.shape[1] // (m1 + m2)
    first_parts = []
    last_parts = []
    
    for k in range(K):
        start_index = k * (m1 + m2)
        first_part = arr[:, start_index:start_index + m1]
        last_part = arr[:, start_index + m1:start_index + m1 + m2]
        first_parts.append(first_part)
        last_parts.append(last_part)
    
    # Concatenate the first m1 and last m2 columns separately, then together
    features = np.hstack(first_parts)
    targets = np.hstack(last_parts)
    return features, targets

def inverse_split_features_targets(features, targets, m1, m2, K):
    N = features.shape[0]
    # Assuming features and targets are the first m1 and last m2 columns, respectively
    original_arr = np.empty((N, K * (m1 + m2)), dtype=features.dtype)
    
    for k in range(K):
        # Extract the relevant sections from features and targets
        feature_part = features[:, k * m1:(k + 1) * m1]
        target_part = targets[:, k * m2:(k + 1) * m2]
        
        # Place the extracted parts back into their original positions
        original_arr[:, k * (m1 + m2):k * (m1 + m2) + m1] = feature_part
        original_arr[:, k * (m1 + m2) + m1:k * (m1 + m2) + m1 + m2] = target_part
    
    return original_arr

def compute_mse(Z, Z_r, description=None):
    mse = mean_squared_error(np.abs(Z), np.abs(Z_r))
    print (f'mse: {mse} - {description}')
    return mse

def compute_PCA(A):
    data_dask = da.from_array(A, chunks=A.shape)
    u, s, v = svd(data_dask)
    U, s, Vh = da.compute(u, s, v)

    V = Vh.conjugate().T
    variance_explained = s**2 / np.sum(s**2)
    projected_data = np.dot(A, V)

    print(f"SVD: U - {U.shape}, s - {s.shape}, V_dagger - {Vh.shape}")
    print("Variance explained by each principal component:", variance_explained)
    print("Projected data:\n", projected_data.shape)

    return projected_data, Vh

def apply_PCA(A, Vh):
    V = Vh.conjugate().T
    projected_data = np.dot(A, V)
    return projected_data, Vh

def compute_inverse_PCA(A, Vh):
    reconstructed_data = np.dot(A, Vh)
    return reconstructed_data

def compute_fft_pca(config, filenames, datafolder_path, wind_angles, angle_to_label, scaler=None, features_targets_fft_eigenvectors=None):
    data, labels = concatenate_data_files(filenames, datafolder_path, wind_angles, angle_to_label)
    features_targets, scaler = get_scaled_features_targets(config, data, config["training"]["input_params"], config["training"]["output_params"], scaler)
    features_targets = reshaped_array(features_targets, len(wind_angles))
    features_targets = features_targets.T
    features_targets_fft = np.fft.rfftn(features_targets)

    print (f'FEATURES & TARGETS before FFT - {features_targets.shape} ; after FFT - {features_targets_fft.shape}')

    if features_targets_fft_eigenvectors is None:
        features_targets_fft_pca, features_targets_fft_eigenvectors = compute_PCA(features_targets_fft)
    else:
        features_targets_fft_pca, features_targets_fft_eigenvectors = apply_PCA(features_targets_fft, features_targets_fft_eigenvectors)

    features_targets_fft_pca = features_targets_fft_pca.T
    features_targets = features_targets.T

    print (f'FEATURES & TARGETS before FFT - {features_targets.shape} ; after FFT, PCA - {features_targets_fft_pca.shape}')

    reduced_features, reduced_targets = split_features_targets(features_targets_fft_pca, len(config["training"]["input_params"]), len(config["training"]["output_params"]))
    features, targets = split_features_targets(features_targets, len(config["training"]["input_params"]), len(config["training"]["output_params"]))

    x = [reduced_features, reduced_targets, features, targets, features_targets_fft_eigenvectors, scaler]

    return x

def compute_inv_fft_inv_pca(config, x, pred=None):

    if pred is not None:
        [reduced_features, reduced_targets, features, targets, features_targets_fft_eigenvectors, scaler, original_shape] = x
        pred = from_real_imag(pred, original_shape)
        reduced_targets = pred
    else:
        [reduced_features, reduced_targets, features, targets, features_targets_fft_eigenvectors, scaler] = x

    features_targets = inverse_split_features_targets(features, targets, len(config["training"]["input_params"]), len(config["training"]["output_params"]), len(config["training"]["angles_to_train"]))
    features_preds = inverse_split_features_targets(reduced_features, reduced_targets, len(config["training"]["input_params"]), len(config["training"]["output_params"]), len(config["training"]["angles_to_train"]))

    features_targets = features_targets.T
    features_preds = features_preds.T

    features_targets_fft_reconstructed = compute_inverse_PCA(features_preds, features_targets_fft_eigenvectors)

    print (f'FEATURES and TARGETS before FFT - {features_targets.shape} ; after FFT, inverse PCA - {features_targets_fft_reconstructed.shape}')

    features_targets_reconstructed = np.fft.irfftn(features_targets_fft_reconstructed, s=features_targets.shape)
    
    features_targets_reconstructed = features_targets_reconstructed.T
    features_targets = features_targets.T

    print (f'FEATURES and TARGETS before FFT - {features_targets.shape} ; after inv FFT reconstructed - {features_targets_reconstructed.shape}')

    features_targets_reconstructed = inverse_reshaped_array(features_targets_reconstructed, len(config["training"]["angles_to_train"]))
    features_targets = inverse_reshaped_array(features_targets, len(config["training"]["angles_to_train"]))

    print (f'FEATURES and TARGETS before FFT - {features_targets.shape} ; after inverse FFT, inverse PCA - {features_targets_reconstructed.shape}')

    features_targets_reconstructed = inverse_transform_targets(features_targets_reconstructed, scaler)
    features_targets = inverse_transform_targets(features_targets, scaler)

    compute_mse(features_targets, features_targets_reconstructed, description='FEATURES AND TARGETS MSE inv FFT inv PCA')

    features, targets = split_features_targets(features_targets, len(config["training"]["input_params"]), len(config["training"]["output_params"]))
    _, targets_reconstructed = split_features_targets(features_targets_reconstructed, len(config["training"]["input_params"]), len(config["training"]["output_params"]))

    assert targets.shape == targets_reconstructed.shape

    print (f'TARGETS before FFT - {targets.shape} ; after inv FFT inv PCA reconstructed - {targets_reconstructed.shape}')

    compute_mse(targets, targets_reconstructed, description='TARGETS MSE inv FFT inv PCA')

    if pred is None:
        x_reconstructed = [targets, targets_reconstructed]
    else:
        x_reconstructed = [features, targets, targets_reconstructed]

    return x_reconstructed

def get_skipped_angles(old_skipped, fixed_training):
    some_temp_shit = []
    for i in fixed_training:
        some_temp_shit.append(i)
    for val in old_skipped:
        index_of_closest = np.argmin(np.abs(np.array(some_temp_shit) - val))
        some_temp_shit[index_of_closest] = val

    skipped_angle_to_label = {angle: idx for idx, angle in enumerate(sorted(some_temp_shit))}
    return some_temp_shit, skipped_angle_to_label

def to_real_imag(tensor_complex):
    def reshaped_array_real_imag(arr):
        num_rows, num_cols, num_dims = arr.shape 
        new_shape = (num_rows, num_cols*num_dims)
        reshaped_arr = arr.reshape(new_shape)
        return reshaped_arr, arr.shape

    real_part = tensor_complex.real
    imag_part = tensor_complex.imag
    real_imag = np.stack((real_part, imag_part), axis=-1)

    original_shape = None

    # real_imag, original_shape = reshaped_array_real_imag(real_imag)

    return real_imag, original_shape

def from_real_imag(real_imag_array, original_shape):
    def inv_reshaped_array_real_imag(arr, original_shape):
        num_rows, num_cols = arr.shape 
        reshaped_arr = arr.reshape(original_shape)
        return reshaped_arr
    real_imag_array = inv_reshaped_array_real_imag(real_imag_array, original_shape)
    real_part = real_imag_array[..., 0]
    imag_part = real_imag_array[..., 1]
    complex_array = real_part + 1j * imag_part
    return complex_array


def get_real_tensor(X, y, device):
    X_train_real_imag, _ = to_real_imag(np.array(X))
    y_train_real_imag, original_shape = to_real_imag(np.array(y))

    X_train_tensor = torch.tensor(X_train_real_imag, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train_real_imag, dtype=torch.float32).to(device)

    original_shape = (X_train_tensor).shape

    print (f'original_shape = {original_shape}')

    N, M, D = X_train_tensor.size()
    new_shape = (N, M*D)

    X_train_tensor = X_train_tensor.reshape(new_shape)
    y_train_tensor = y_train_tensor.reshape(new_shape)

    # print (X.shape, X_train_real_imag.shape)

    # X_train_tensor = X_train_tensor.view(X_train_tensor.size(0), -1)
    # y_train_tensor = y_train_tensor.view(y_train_tensor.size(0), -1)

    # X_train_tensor = X_train_tensor.view(X_train_tensor.size(0), -1)
    # y_train_tensor = y_train_tensor.view(y_train_tensor.size(0), -1)

    return X_train_tensor, y_train_tensor, original_shape

def load_data_fft(config, device):
    chosen_machine_key = config["chosen_machine"]
    datafolder_path = os.path.join(config["machine"][chosen_machine_key], config["data"]["data_folder_name"])
    filenames = get_filenames_from_folder(datafolder_path, config["data"]["extension"], config["data"]["startname_data"])
    training_wind_angles = config["training"]["angles_to_train"]
    skipped_wind_angles = config["training"]["angles_to_leave_out"]
    output_params = config["training"]["output_params"]
    input_params = config["training"]["input_params"]
    input_params_points = config["training"]["input_params_points"]
    angle_to_label = {angle: idx for idx, angle in enumerate(sorted(training_wind_angles))}

    x_training = compute_fft_pca(config, filenames, datafolder_path, training_wind_angles, angle_to_label)
    x_training_reconstructed = compute_inv_fft_inv_pca(config, x_training)

    skipped_wind_angles, skipped_angle_to_label = get_skipped_angles(skipped_wind_angles, training_wind_angles)
    print ('skipped', skipped_wind_angles)

    x_skipped = compute_fft_pca(config, filenames, datafolder_path, skipped_wind_angles, skipped_angle_to_label, x_training[5], x_training[4])
    x_skipped_reconstructed = compute_inv_fft_inv_pca(config, x_skipped) 

    X_train_tensor, y_train_tensor, original_shape_train = get_real_tensor(x_training[0], x_training[0], device)

    X_test_tensor_skipped, y_test_tensor_skipped, original_shape_skipped = get_real_tensor(x_skipped[0], x_skipped[0], device)

    # X_test_real_imag_skipped = to_real_imag(np.array(x_skipped[0].T))
    # y_test_real_imag_skipped = to_real_imag(np.array(x_skipped[1].T))

    # X_test_tensor_skipped = torch.tensor(X_test_real_imag_skipped, dtype=torch.float32).to(device)
    # y_test_tensor_skipped = torch.tensor(y_test_real_imag_skipped, dtype=torch.float32).to(device)

    print (x_training[0].shape, x_training[1].shape, x_skipped[0].shape, x_skipped[1].shape)
    print (X_train_tensor.shape, y_train_tensor.shape, X_test_tensor_skipped.shape, y_test_tensor_skipped.shape)

    # original_shape = (X_train_tensor.T).shape

    # X_train_tensor = X_train_tensor.view(X_train_tensor.size(0), -1)
    # y_train_tensor = y_train_tensor.view(y_train_tensor.size(0), -1)
    # X_test_tensor_skipped = X_test_tensor_skipped.view(X_test_tensor_skipped.size(0), -1)
    # y_test_tensor_skipped = y_test_tensor_skipped.view(y_test_tensor_skipped.size(0), -1)

    # print (X_train_tensor.shape, y_train_tensor.shape, X_test_tensor_skipped.shape, y_test_tensor_skipped.shape)

    # X_train_tensor, y_train_tensor, X_test_tensor_skipped, y_test_tensor_skipped = X_train_tensor.T, y_train_tensor.T, X_test_tensor_skipped.T, y_test_tensor_skipped.T

    # print (X_train_tensor.shape, y_train_tensor.shape, X_test_tensor_skipped.shape, y_test_tensor_skipped.shape)

    x_training.append(original_shape_train)
    x_skipped.append(original_shape_skipped)

    data_dict = {
        "X_train_tensor": X_train_tensor,
        "y_train_tensor": y_train_tensor,
        "X_test_tensor_skipped": X_test_tensor_skipped,
        "y_test_tensor_skipped": y_test_tensor_skipped,
        "relevant_data_training": x_training,
        "relevant_data_skipped": x_skipped
        }

    return data_dict

def evaluate_model_training_fft(config, device, model, model_file_path, X_test_tensor, y_test_tensor, x):
    checkpoint = torch.load(model_file_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    with torch.no_grad():
        predictions_tensor = model(X_test_tensor)
        predictions_tensor_cpu = predictions_tensor.cpu()
        y_test_tensor_cpu = y_test_tensor.cpu()
        X_test_tensor_cpu = X_test_tensor.cpu()
        x_reconstructed = compute_inv_fft_inv_pca(config, x, predictions_tensor_cpu)
        [features, targets_final, targets_pred_final] = x_reconstructed
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

def evaluate_model_fft(config, model, wind_angles, output_folder, physics, vtk_output, X_test_tensor, y_test_tensor, x):

    def extract_dataset_by_wind_angle(df, dataset_size, lower_bound, upper_bound):
        for start_idx in range(0, len(df), dataset_size):
            if df.iloc[start_idx]['WindAngle'] >= lower_bound and df.iloc[start_idx]['WindAngle'] <= upper_bound:
                return df.iloc[start_idx:start_idx+dataset_size]
        return pd.DataFrame()

    model.eval()
    dfs = []
    with torch.no_grad():
        predictions_tensor = model(X_test_tensor)
        predictions_tensor_cpu = predictions_tensor.cpu()
        [reduced_features, reduced_targets, features, targets, features_targets_fft_eigenvectors, scaler, original_shape] = x

        # features = features.T
        # features = inverse_reshaped_array(features, len(config["training"]["angles_to_train"]))
        # features[:, :3] = inverse_transform_features(features[:, :3], feature_scaler)
        
        # targets = inverse_reshaped_array(targets, len(config["training"]["angles_to_train"]))
        # targets = inverse_transform_targets(targets, target_scaler)

        

        # predictions_tensor_cpu = compute_inverse_PCA_from_PCA(predictions_tensor_cpu, pca_targets, pca_reduce)
        # predictions_tensor_cpu = inverse_reshaped_array(predictions_tensor_cpu, len(config["training"]["angles_to_train"]))
        # predictions_tensor_cpu = inverse_transform_targets(predictions_tensor_cpu, target_scaler)
        
        [features, targets, predictions_tensor_cpu] = compute_inv_fft_inv_pca(config, x, predictions_tensor_cpu)
        
        X_test_tensor_cpu = features
        y_test_tensor_cpu = targets

        whole_tensor = np.hstack([X_test_tensor_cpu, y_test_tensor_cpu, predictions_tensor_cpu])

        X_test_column_names = config["training"]["input_params_modf"]
        output_column_names = config["training"]["output_params_modf"]
        y_test_column_names = [item + "_Actual" for item in output_column_names]
        predictions_column_names = [item + "_Predicted" for item in output_column_names]
        whole_column_names = X_test_column_names + y_test_column_names + predictions_column_names

        length_dataset = int(X_test_tensor_cpu.shape[0]/len(config["training"]["angles_to_train"]))

        whole = pd.DataFrame(whole_tensor, columns=whole_column_names)
        whole['Velocity_Magnitude_Actual'] = np.sqrt(whole['Velocity_X_Actual']**2 + whole['Velocity_Y_Actual']**2 + whole['Velocity_Z_Actual']**2)
        whole['Velocity_Magnitude_Predicted'] = np.sqrt(whole['Velocity_X_Predicted']**2 + whole['Velocity_Y_Predicted']**2 + whole['Velocity_Z_Predicted']**2)
        whole['WindAngle_rad'] = np.arctan2(whole['sin(WindAngle)'], whole['cos(WindAngle)'])
        whole['WindAngle'] = np.abs(whole['WindAngle_rad'].apply(lambda x: int(np.ceil(np.degrees(x)))))
        whole.sort_values(by='WindAngle', ascending=True, inplace=True)

        for wind_angle in wind_angles:
            lower_bound = wind_angle - 5
            upper_bound = wind_angle + 5

            filtered_whole = extract_dataset_by_wind_angle(whole, length_dataset, lower_bound, upper_bound)

            predictions_column_names.append('Velocity_Magnitude_Predicted')
            y_test_column_names.append('Velocity_Magnitude_Actual')
            filtered_predictions = filtered_whole[predictions_column_names]
            filtered_y_test = filtered_whole[y_test_column_names]
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

                dfs.append(filtered_whole)

                data_folder = os.path.join(output_folder, f'data_output_for_wind_angle_{wind_angle}')
                os.makedirs(output_folder, exist_ok=True)
                os.makedirs(data_folder, exist_ok=True)

                combined_file_path = os.path.join(data_folder, f'combined_actuals_and_predictions_for_wind_angle_{wind_angle}.csv')
                metrics_file_path = os.path.join(data_folder, f'metrics_for_wind_angle_{wind_angle}.csv')
                metrics_df = pd.DataFrame(rows_list)
                metrics_df.to_csv(metrics_file_path, index=False)

                if config["plotting"]["save_csv_predictions"]:
                    filtered_whole.to_csv(combined_file_path, index=False)
                       
                if vtk_output is not None and physics is not None and wind_angle in config["training"]["all_angles"]:
                    if config["plotting"]["save_vtk"]:
                        output_nn_to_vtk(config, wind_angle, f'{physics}_predictions_for_wind_angle_{wind_angle}', filtered_whole, predictions_column_names, vtk_output)
    
    data = pd.concat(dfs)
    return data
###FFT-STYLE-2###