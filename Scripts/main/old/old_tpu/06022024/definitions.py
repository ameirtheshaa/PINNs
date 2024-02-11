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
import numpy as np
import pandas as pd
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
import sklearn.metrics
from pathlib import Path
import json
import tensorflow as tf
from stl import mesh

def init_tpu():
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
        print("Running on TPU:", resolver.master())
        return strategy
    except ValueError:
        print("Could not connect to TPU")
        return None

class Logger(object):
    def __init__(self, filename='Default.log'):
        self.terminal = sys.stdout  # Save the original stdout
        self.log = open(filename, 'a')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass

def create_dataset(X_train, y_train, labels_train, batch_size):
    # Create a TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train, labels_train))

    # Shuffle and batch the dataset
    dataset = dataset.shuffle(buffer_size=batch_size).batch(batch_size)

    return dataset

def save_to_csv(epoch, epochs, use_epoch, current_loss, current_elapsed_time_hours, file_path):
    data = {
        'Epoch': f'{epoch}',
        'Loss': current_loss,
        'Total Time Elapsed (hours)': f'{current_elapsed_time_hours:.2f}'
    }
    file_exists = os.path.isfile(file_path) and os.path.getsize(file_path) > 0
    with open(file_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data.keys())
        # Write the header only if the file is new or empty
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

def get_filenames_from_folder(path, extension, startname):
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith(extension) and f.startswith(startname)]

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
                dfs.append(df)
                if angle_labels is not None:
                    label = angle_labels[wind_angle]
                    labels.extend([label] * len(df))
                    df['WindAngle'] = (wind_angle) #temp
    data = pd.concat(dfs)
    if angle_labels is not None:
        labels = np.array(labels)
        return data, labels
    else:
        return data

def initialize_and_fit_scalers(features, targets, config):
    feature_scaler = config["training"]["feature_scaler"]
    target_scaler = config["training"]["target_scaler"]
    feature_scaler.fit(features)
    target_scaler.fit(targets)
    return feature_scaler, target_scaler

def transform_data_with_scalers(features, targets, feature_scaler, target_scaler):
    normalized_features = feature_scaler.transform(features)
    normalized_targets = target_scaler.transform(targets)
    return normalized_features, normalized_targets

def inverse_transform_features(features_normalized, feature_scaler):
    return feature_scaler.inverse_transform(features_normalized)

def inverse_transform_targets(targets_normalized, target_scaler):
    return target_scaler.inverse_transform(targets_normalized)

def convert_to_tensor(X_train, X_test, y_train, y_test):
    X_train_tensor = tf.convert_to_tensor(np.array(X_train), dtype=tf.float16)
    y_train_tensor = tf.convert_to_tensor(np.array(y_train), dtype=tf.float16)
    X_test_tensor = tf.convert_to_tensor(np.array(X_test), dtype=tf.float16)
    y_test_tensor = tf.convert_to_tensor(np.array(y_test), dtype=tf.float16)
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor

def print_and_set_available_gpus():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print(f"{len(gpus)} GPUs available:")
        for i, gpu in enumerate(gpus):
            tf.config.experimental.set_memory_growth(gpu, True)
            gpu_details = tf.config.experimental.get_device_details(gpu)
            free_memory = gpu_details['memory_size']
            print(f"GPU {i}: {gpu}, Free Memory: {free_memory / (1024**3):.2f} GB")
        selected_gpu = 0  # Defaulting to the first GPU if multiple are available
        device = tf.device(f'/device:GPU:{selected_gpu}')
        print(f"Using GPU: {device}")
    else:
        print("No GPUs available.")
        device = tf.device('cpu')
        print(f"Using CPU: {device}")
    return device

def get_optimizer(model, chosen_optimizer_key, optimizer_config):
    if chosen_optimizer_key == "lbfgs_optimizer":
        # TensorFlow doesn't have a direct equivalent of LBFGS.
        # Using SGD as a placeholder.
        optimizer = tf.keras.optimizers.SGD(learning_rate=optimizer_config["learning_rate"], 
                                            momentum=0.9, 
                                            nesterov=True)
    elif chosen_optimizer_key == "adam_optimizer":
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=optimizer_config["learning_rate"])
    else:
        raise ValueError(f"Unsupported optimizer type: {chosen_optimizer_key}")
    return optimizer

def load_data(config):
    rho = config["data"]["density"]
    nu = config["data"]["kinematic_viscosity"]
    chosen_machine_key = config["chosen_machine"]
    datafolder_path = os.path.join(config["machine"][chosen_machine_key], config["data"]["data_folder_name"])
    filenames = get_filenames_from_folder(datafolder_path, config["data"]["extension"], config["data"]["startname_data"])
    training_wind_angles = config["training"]["angles_to_train"]
    angle_to_label = {angle: idx for idx, angle in enumerate(sorted(training_wind_angles))}
    skipped_wind_angles = config["training"]["angles_to_leave_out"]
    data, labels = concatenate_data_files(filenames, datafolder_path, training_wind_angles, angle_to_label)
    features = data[config["training"]["input_params"]]
    targets = data[config["training"]["output_params"]]
    feature_scaler, target_scaler = initialize_and_fit_scalers(features, targets, config)
    normalized_features, normalized_targets = transform_data_with_scalers(features, targets, feature_scaler, target_scaler)
    X_train, X_test, y_train, y_test, labels_train, labels_test = train_test_split(normalized_features, normalized_targets, labels, test_size=config["train_test"]["test_size"], random_state=config["train_test"]["random_state"])
    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = convert_to_tensor(X_train, X_test, y_train, y_test)
    data_skipped = concatenate_data_files(filenames, datafolder_path, skipped_wind_angles)
    features_skipped = data_skipped[config["training"]["input_params"]]
    targets_skipped = data_skipped[config["training"]["output_params"]]
    normalized_features_skipped, normalized_targets_skipped = transform_data_with_scalers(features_skipped, targets_skipped, feature_scaler, target_scaler)
    X_train_skipped, X_test_skipped, y_train_skipped, y_test_skipped = train_test_split(normalized_features_skipped, normalized_targets_skipped,test_size=len(data_skipped)-1, random_state=config["train_test"]["random_state"])
    X_train_tensor_skipped, y_train_tensor_skipped, X_test_tensor_skipped, y_test_tensor_skipped = convert_to_tensor(X_train_skipped, X_test_skipped, y_train_skipped, y_test_skipped)
    labels_train_tensor = labels_train
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, labels_train_tensor, X_train_tensor_skipped, y_train_tensor_skipped, X_test_tensor_skipped, y_test_tensor_skipped, feature_scaler, target_scaler


def evaluate_model(config, model, wind_angles, X_test_tensor, feature_scaler, target_scaler, y_test_tensor = None, output_folder = None):
    dfs = []
    predictions_tensor = model.predict(X_test_tensor)
    X_test_tensor = inverse_transform_features(X_test_tensor, feature_scaler)
    X_test_column_names = config["training"]["input_params_modf"]
    X_test_dataframe = pd.DataFrame(X_test_tensor, columns=X_test_column_names)
    os.makedirs(output_folder, exist_ok=True)
    y_test_tensor = inverse_transform_targets(y_test_tensor, target_scaler)
    output_column_names = config["training"]["output_params_modf"]
    y_test_column_names = [item + "_Actual" for item in output_column_names]
    y_test_dataframe = pd.DataFrame(y_test_tensor, columns=y_test_column_names)
    y_test_dataframe['Velocity_Magnitude_Actual'] = np.sqrt(y_test_dataframe['Velocity_X_Actual']**2 + y_test_dataframe['Velocity_Y_Actual']**2 + y_test_dataframe['Velocity_Z_Actual']**2)
    predictions_tensor = inverse_transform_targets(predictions_tensor, target_scaler)
    predictions_column_names = [item + "_Predicted" for item in output_column_names]
    predictions_dataframe = pd.DataFrame(predictions_tensor, columns=predictions_column_names)
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