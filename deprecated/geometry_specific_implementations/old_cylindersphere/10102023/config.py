import os
import numpy as np
import subprocess
import sys
import sys
import os
import torch
import time
import datetime
import psutil
import argparse
import sklearn
import itertools
import GPUtil
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import numpy as np
from scipy.interpolate import griddata
import importlib.util
import os
import shutil

config = {
    "lbfgs_optimizer": {
        "type": "LBFGS",
        "learning_rate": 0.001,
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
        "max_iter": 200000,
        "max_eval": 50000,
        "history_size": 50,
        "tolerance_grad": 1e-05,
        "tolerance_change": 0.5 * np.finfo(float).eps,
        "line_search_fn": "strong_wolfe"
    },
    "loss_components": {
        "data_loss": True,
        "cont_loss": True,
        "momentum_loss": False
    },
    "train_test": {
        "train": True,
        "test": True,
        "evaluate": True,
        "test_size": 0.2,
        "random_state": 42
    },
    "batches": {
        "use_batches": True,
        "force": True,
        "batch_size_ratio": 0.01
    },
    "lxplus": {
        "data_folder": os.path.join('/afs/cern.ch/user/a/abinakbe/PINNs','cylinder_cell', 'data'),
    },
    "mac": {
        "data_folder": os.path.join(os.path.expanduser("~"), "Dropbox", "School", "Graduate", "CERN", "Temp_Files", "nonlineardynamics", "ameir_PINNs", "cylinder_cell", "data"),
    },
    "workstation": {
        "data_folder": os.path.join('E:\\','ameir', "Dropbox", "School", "Graduate", "CERN", "Temp_Files", "nonlineardynamics", "ameir_PINNs", "cylinder_cell", "data"),
    },
    "use_epoch": False,
    "make_individual_plots": True,
    "one_file_test": False,
    "chosen_machine": "mac",
    "chosen_optimizer": "adam_optimizer",
    "angle_to_leave_out": 90,
    "epochs": 1000,
    "loss_diff_threshold": 1e-5,
    "consecutive_count_threshold": 10,
    "density": 1,
    "kinematic_viscosity": 1e-5,    
    "extension": '.csv',
    "startname": 'CFD'
}

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

def print_and_set_available_gpus():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"{num_gpus} GPUs available:")

        if num_gpus > 1:
            gpus = GPUtil.getGPUs()
            free_memory = [gpu.memoryFree for gpu in gpus]
            selected_gpu = free_memory.index(max(free_memory))
        else:
            selected_gpu = 0
            free_memory = [torch.cuda.get_device_properties(0).total_memory / (1024 ** 2) - torch.cuda.memory_allocated(0) / (1024 ** 2)] # Free memory in MB

        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

        device = torch.device(f'cuda:{selected_gpu}')
        print(f"Using device: {device}, with {free_memory[selected_gpu]/1000} GB free memory")
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
    
    return total_memory  # in bytes

def select_device_and_batch_size(model, input_tensor):
    single_sample_memory = estimate_memory(model, input_tensor, batch_size=1)
    
    # Check for GPU availability
    if torch.cuda.is_available():
        # Get the ID of the first available GPU
        gpu_id = torch.cuda.current_device()
        
        # Clear the unused memory and get the available memory on the GPU
        torch.cuda.empty_cache()
        total_gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory
        available_gpu_memory = total_gpu_memory - torch.cuda.memory_allocated(gpu_id)
        
        print(f"GPU Total Memory: {total_gpu_memory / (1024 ** 3):.2f} GB")
        print(f"GPU Available Memory: {available_gpu_memory / (1024 ** 3):.2f} GB")

        # Determine the maximum batch size that can fit in the GPU memory
        max_batch_size_gpu = available_gpu_memory // single_sample_memory
        print(f"GPU with optimal batch size: {max_batch_size_gpu}")
        
    # Print CPU memory
    cpu_memory = psutil.virtual_memory()
    print(f"CPU Total Memory: {cpu_memory.total / (1024 ** 3):.2f} GB")
    print(f"CPU Available Memory: {cpu_memory.available / (1024 ** 3):.2f} GB")

    # Determine the maximum batch size that can fit in the CPU memory
    max_batch_size_cpu = cpu_memory.available // single_sample_memory
    print(f"CPU with optimal batch size: {max_batch_size_cpu}")
    

    # Select the device with the maximum possible batch size
    if torch.cuda.is_available():
        optimal_batch_size = max_batch_size_gpu
        device = torch.device("cuda", gpu_id)
    else:
        optimal_batch_size = max_batch_size_cpu
        device = torch.device("cpu")

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

def load_data(filenames, base_directory, datafolder_path, device, config):

    rho = config["density"]
    nu = config["kinematic_viscosity"]
    angle_to_leave_out = config["angle_to_leave_out"]

    dfs = []

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
            df['WindAngle'] = wind_angle  
            df['cos(WindAngle)'] = np.abs(np.cos(np.deg2rad(wind_angle)))
            df['sin(WindAngle)'] = np.abs(np.sin(np.deg2rad(wind_angle)))
            df['Density'] = rho  
            df['Kinematic_Viscosity'] = nu  
            
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
                df['WindAngle'] = wind_angle  
                df['cos(WindAngle)'] = np.abs(np.cos(np.deg2rad(wind_angle)))
                df['sin(WindAngle)'] = np.abs(np.sin(np.deg2rad(wind_angle)))
                df['Density'] = rho  
                df['Kinematic_Viscosity'] = nu  
                
                # Append the modified DataFrame to the list
                dfs.append(df)

    # Concatenate the list of DataFrames
    data = pd.concat(dfs)

    # Extract features from the dataframe
    features = data[['Points:0', 'Points:1', 'Points:2', 'TurbVisc', 'WindAngle', 'cos(WindAngle)', 'sin(WindAngle)', 'Density', 'Kinematic_Viscosity']]

    targets = data[['Pressure', 'Velocity:0', 'Velocity:1', 'Velocity:2']]

    # Initialize Standard Scalers for features and targets
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    # Fit the scalers and transform the features and targets
    normalized_features = feature_scaler.fit_transform(features)
    normalized_targets = target_scaler.fit_transform(targets)

    # Perform the train-test split and get the indices
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(normalized_features, normalized_targets, range(len(normalized_features)),test_size=config["train_test"]["test_size"], random_state=config["train_test"]["random_state"])

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

def load_skipped_angle_data(filenames, base_directory, datafolder_path, device, config):

    rho = config["density"]
    nu = config["kinematic_viscosity"]
    angle_to_leave_out = config["angle_to_leave_out"]

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
                df['WindAngle'] = wind_angle  
                df['cos(WindAngle)'] = np.abs(np.cos(np.deg2rad(wind_angle)))
                df['sin(WindAngle)'] = np.abs(np.sin(np.deg2rad(wind_angle)))
                df['Density'] = rho  
                df['Kinematic_Viscosity'] = nu  
                dfs.append(df)
            else:
                print (f'Skipping Angle = {wind_angle} degrees that was trained earlier')

    # Concatenate the list of DataFrames
    data = pd.concat(dfs)

    # Extract features from the dataframe
    features = data[['Points:0', 'Points:1', 'Points:2', 'TurbVisc', 'WindAngle', 'cos(WindAngle)', 'sin(WindAngle)', 'Density', 'Kinematic_Viscosity']]

    targets = data[['Pressure', 'Velocity:0', 'Velocity:1', 'Velocity:2']]

    # Initialize Standard Scalers for features and targets
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    # Fit the scalers and transform the features and targets
    normalized_features = feature_scaler.fit_transform(features)
    normalized_targets = target_scaler.fit_transform(targets)

    # Perform the train-test split and get the indices
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

def train_model(model, X_train_tensor, y_train_tensor, config, batch_size, model_file_path, epochs):
    # Capture start time
    start_time = time.time()
    model_file_path = Path(model_file_path)  # Convert the string to a Path object
    if config["batches"]["use_batches"]:
        if not config["batches"]["force"]:
            train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
            print (f"using batches with batch size: {batch_size}")
        else:
            train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
            batch_size = int(len(X_train_tensor)*float(config["batches"]["batch_size_ratio"]))
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
            print (f"using batches with forced batch size: {batch_size}")
    else:
        train_loader = [(X_train_tensor, y_train_tensor)]  # Use the entire dataset as one "batch"
        print (f"using full tensor with size: {len(X_train_tensor)}")
    
    model.train()

    chosen_optimizer_key = config["chosen_optimizer"]
    chosen_optimizer_config = config[chosen_optimizer_key]
    optimizer = get_optimizer(model, chosen_optimizer_config)

    def calculate_total_loss(X, y, config):
        total_loss = 0
        if config["loss_components"]["data_loss"]:
            data_loss = model.compute_data_loss(X, y)
            total_loss += data_loss
        
        if config["loss_components"]["momentum_loss"]:
            momentum_loss = model.compute_physics_momentum_loss(X)
            total_loss += momentum_loss
        
        if config["loss_components"]["cont_loss"]:
            cont_loss = model.compute_physics_cont_loss(X)
            total_loss += cont_loss
        
        return total_loss

    def closure():
        optimizer.zero_grad()
        predictions, stress_tensor = model(X_batch)
        total_loss = calculate_total_loss(X_batch, y_batch, config)
        total_loss.backward()
        return total_loss

    previous_loss = None
    use_epoch = config["use_epoch"]
    loss_diff_threshold = config["loss_diff_threshold"]
    consecutive_count_threshold = config["consecutive_count_threshold"]
    consecutive_count = 0
    early_stop = False  # Flag to break out of both loops

    if os.path.exists(model_file_path):
        checkpoint = torch.load(model_file_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        training_completed = checkpoint.get('training_completed', False)  # Default to False if the flag is not in the saved state
        if training_completed:
            print("Training has already been completed.")
            return model
        else:
            start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 1
    for epoch in itertools.count(start=start_epoch):  # infinite loop, will only break if the stopping condition is met.
        if early_stop or (use_epoch and epoch > epochs):
            training_completed = True
            # Save the final state and training completed status
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': current_loss,
                'training_completed': training_completed
            }, model_file_path)
            break
        else:
            training_completed = False
        for X_batch, y_batch in train_loader:
            current_loss = optimizer.step(closure)
            if previous_loss is not None:
                loss_diff = abs(current_loss - previous_loss)
                if loss_diff < loss_diff_threshold:
                    consecutive_count += 1
                    if consecutive_count >= consecutive_count_threshold:
                        print(f"Consecutive loss difference less than {loss_diff_threshold} for {consecutive_count_threshold} epochs at epoch {epoch}. Stopping training...")
                        early_stop = True  # Set the early_stop flag to True to break out of the outer loop
                        break  # Break the inner loop
                else:
                    consecutive_count = 0
            if epoch % 100 == 0:
                current_elapsed_time = time.time() - start_time
                current_elapsed_time_hours = current_elapsed_time / 3600
                print(f'Epoch [{epoch}/{epochs if use_epoch else "infinity"}], Loss: {current_loss:.4f}, Total Time elapsed: {current_elapsed_time_hours:.2f} hours')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': current_loss,
                    'training_completed': training_completed
                }, model_file_path)
            previous_loss = current_loss

    # Capture end time
    end_time = time.time()
    total_elapsed_time = end_time - start_time
    total_elapsed_time_hours = total_elapsed_time / 3600
    print(f'Training completed in {total_elapsed_time_hours:.2f} hours')

    # Save the final state and training completed status
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': current_loss,
        'training_completed': training_completed
    }, model_file_path)

    return model

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

def evaluate_model(model, variables, features, idx_test, X_test_tensor, y_test, output_folder):
    model.eval()
    test_predictions_wind_angle = []
    with torch.no_grad():
        predictions, stress_tensor = model(X_test_tensor)
        unique_wind_angles = list(features['WindAngle'].unique())
        for wind_angle in unique_wind_angles:
            # Create a boolean mask for the current wind_angle using idx_test
            mask = features.iloc[idx_test]['WindAngle'] == wind_angle

            # Filter the features DataFrame using the mask, then select the desired columns
            filtered_features = features.iloc[idx_test][mask]
            x = filtered_features['Points:0']
            y = filtered_features['Points:1']
            z = filtered_features['Points:2']

            positions = {'x': x, 'y': y, 'z': z}
            
            # Convert the boolean mask to a torch tensor
            mask_tensor = torch.BoolTensor(mask.to_numpy())
            
            # Use the mask tensor to index predictions and y_test
            filtered_predictions = predictions[mask_tensor]
            filtered_y_test = y_test[mask_tensor]

            test_predictions_wind_angle.append([wind_angle, positions, filtered_y_test, filtered_predictions])

            # Initialize a list to store the rows
            rows_list = []
            for i, var in enumerate(variables):
                actuals = filtered_y_test[:, i].cpu().numpy()
                preds = filtered_predictions.cpu().numpy()[:, i]

                mse = sklearn.metrics.mean_squared_error(actuals, preds)
                rmse = np.sqrt(mse)
                mae = sklearn.metrics.mean_absolute_error(actuals, preds)
                r2 = sklearn.metrics.r2_score(actuals, preds)
                
                # Append the new row as a dictionary to the list
                rows_list.append({
                    'Variable': var, 
                    'MSE': mse,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R2': r2
                })

            # Create the DataFrame from the list of rows at the end
            metrics_df = pd.DataFrame(rows_list)   
            # Create a DataFrame directly from the NumPy arrays
            positions_df = pd.DataFrame({
                f"Position_X": x,
                f"Position_Y": y,
                f"Position_Z": z,
            })

            data_folder = os.path.join(output_folder, f'data_output_for_wind_angle_{wind_angle}')
            os.makedirs(data_folder, exist_ok=True)
            actuals_df = pd.DataFrame(filtered_y_test.cpu().numpy(), columns=[f"Actual_{var}" for var in variables])
            predictions_df = pd.DataFrame(filtered_predictions.cpu().numpy(), columns=[f"Predicted_{var}" for var in variables])
            combined_df = pd.concat([positions_df, actuals_df, predictions_df], axis=1)
            combined_file_path = os.path.join(data_folder, f'combined_actuals_and_predictions_for_wind_angle_{wind_angle}.csv')
            combined_df.to_csv(combined_file_path, index=False)
            metrics_file_path = os.path.join(data_folder, f'metrics_for_wind_angle_{wind_angle}.csv')
            metrics_df.to_csv(metrics_file_path, index=False)
    with torch.no_grad():
        predictions, stress_tensor = model(X_test_tensor)
        
        # Initialize a list to store the rows
        rows_list = []
        for i, var in enumerate(variables):
            actuals = y_test[:, i].cpu().numpy()
            preds = predictions.cpu().numpy()[:, i]

            mse = sklearn.metrics.mean_squared_error(actuals, preds)
            rmse = np.sqrt(mse)
            mae = sklearn.metrics.mean_absolute_error(actuals, preds)
            r2 = sklearn.metrics.r2_score(actuals, preds)
            
            # Append the new row as a dictionary to the list
            rows_list.append({
                'Variable': var, 
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            })

        # Create the DataFrame from the list of rows at the end
        metrics_df = pd.DataFrame(rows_list)

        list_of_directions = ['Points:0','Points:1','Points:2']
        x = features[list_of_directions[0]].to_numpy()
        y = features[list_of_directions[1]].to_numpy()
        z = features[list_of_directions[2]].to_numpy()
        x = x[idx_test]
        y = y[idx_test]
        z = z[idx_test]

        # Create a DataFrame directly from the NumPy arrays
        positions_df = pd.DataFrame({
            f"Position_{list_of_directions[0]}": x,
            f"Position_{list_of_directions[1]}": y,
            f"Position_{list_of_directions[2]}": z,
        })
        actuals_df = pd.DataFrame(y_test.cpu().numpy(), columns=[f"Actual_{var}" for var in variables])
        predictions_df = pd.DataFrame(predictions.cpu().numpy(), columns=[f"Predicted_{var}" for var in variables])
        combined_df = pd.concat([positions_df, actuals_df, predictions_df], axis=1)
        combined_file_path = os.path.join(output_folder, 'combined_actuals_and_predictions.csv')
        combined_df.to_csv(combined_file_path, index=False)
        metrics_file_path = os.path.join(output_folder, 'metrics.csv')
        metrics_df.to_csv(metrics_file_path, index=False)

    return predictions, test_predictions_wind_angle

# Define the PINN model
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.fc1 = nn.Linear(9, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.stress_tensor_layer = nn.Linear(128, 6)
        self.output_layer = nn.Linear(128, 4)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        stress_tensor = self.stress_tensor_layer(x)
        output = self.output_layer(x)
        return output, stress_tensor

    def compute_data_loss(self, X, y):
        criterion = nn.MSELoss()
        predictions, stress_tensor = self(X)
        loss = criterion(predictions, y)
        return loss
    
    def extract_parameters(self, X):
        wind_angle = X[:, 4:5]  # WindAngle
        x = X[:, 0:1]  # Points:0
        y = X[:, 1:2]  # Points:1
        z = X[:, 2:3]  # Points:2
        nu_t = X[:, 3:4]  # TurbVisc
        rho = X[:, 7:8] # Density
        nu = X[:, 8:9] # Kinematic Viscosity

        return wind_angle, x, y, z, nu_t, rho, nu

    def RANS(self, wind_angle, x, y, z, nu_t, rho, nu):
        # Ensure x, y, and z have requires_grad set to True
        x.requires_grad_(True)
        y.requires_grad_(True)
        z.requires_grad_(True)

        # Compute absolute values of sin and cos of wind_angle
        abs_cos_theta = torch.abs(torch.cos(torch.deg2rad(wind_angle)))
        abs_sin_theta = torch.abs(torch.sin(torch.deg2rad(wind_angle)))

        # Stack the input data
        input_data = torch.hstack((x, y, z, nu_t, wind_angle, abs_cos_theta, abs_sin_theta, rho, nu))

        # Get the output and stress_tensor from the model
        output, stress_tensor = self(input_data)

        # Extract u, v, w, and p from the output
        u, v, w, p = output[:, 0], output[:, 1], output[:, 2], output[:, 3]

        def compute_gradients_and_second_order_gradients(tensor, coord):
            # Compute the gradients and second order gradients of tensor with respect to coord
            grad_first = torch.autograd.grad(tensor, coord, grad_outputs=torch.ones_like(tensor), create_graph=True)[0]
            grad_second = torch.autograd.grad(grad_first, coord, grad_outputs=torch.ones_like(grad_first), create_graph=True)[0]
            return grad_first, grad_second

        # Compute gradients and second order gradients
        u_x, u_xx = compute_gradients_and_second_order_gradients(u, x)
        u_y, u_yy = compute_gradients_and_second_order_gradients(u, y)
        u_z, u_zz = compute_gradients_and_second_order_gradients(u, z)

        v_x, v_xx = compute_gradients_and_second_order_gradients(v, x)
        v_y, v_yy = compute_gradients_and_second_order_gradients(v, y)
        v_z, v_zz = compute_gradients_and_second_order_gradients(v, z)

        w_x, w_xx = compute_gradients_and_second_order_gradients(w, x)
        w_y, w_yy = compute_gradients_and_second_order_gradients(w, y)
        w_z, w_zz = compute_gradients_and_second_order_gradients(w, z)

        # Compute the gradients of p
        p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        p_z = torch.autograd.grad(p, z, grad_outputs=torch.ones_like(p), create_graph=True)[0]

        # Compute f, g, h, and cont using the obtained gradients and second order gradients
        f = rho * (u * u_x + v * u_y + w * u_z) + p_x - nu * (u_xx + u_yy + u_zz) - stress_tensor[:, 0] - stress_tensor[:, 3] - stress_tensor[:, 4]
        g = rho * (u * v_x + v * v_y + w * v_z) + p_y - nu * (v_xx + v_yy + v_zz) - stress_tensor[:, 3] - stress_tensor[:, 1] - stress_tensor[:, 5]
        h = rho * (u * w_x + v * w_y + w * w_z) + p_z - nu * (w_xx + w_yy + w_zz) - stress_tensor[:, 4] - stress_tensor[:, 5] - stress_tensor[:, 2]

        cont = u_x + v_y + w_z

        return u, v, w, p, f, g, h, cont

    def compute_physics_momentum_loss(self, X):
        wind_angle, x, y, z, nu_t, rho, nu = self.extract_parameters(X)  # Extract the required input parameters for the RANS function
        u_pred, v_pred, w_pred, p_pred, f, g, h, cont = self.RANS(wind_angle, x, y, z, nu_t, rho, nu)
        
        loss_f = nn.MSELoss()(f, torch.zeros_like(f))
        loss_g = nn.MSELoss()(g, torch.zeros_like(g))
        loss_h = nn.MSELoss()(h, torch.zeros_like(h))
        
        loss_physics_momentum = loss_f + loss_g + loss_h 
        return loss_physics_momentum

    def compute_physics_cont_loss(self, X):
        wind_angle, x, y, z, nu_t, rho, nu  = self.extract_parameters(X)  # Extract the required input parameters for the RANS function
        u_pred, v_pred, w_pred, p_pred, f, g, h, cont = self.RANS(wind_angle, x, y, z, nu_t, rho, nu)
        
        loss_cont = nn.MSELoss()(cont, torch.zeros_like(cont))

        return loss_cont

def plot_predictions(y_test, predictions, output_folder, variables):
    for i, var in enumerate(variables):
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(y_test)), y_test.cpu().numpy()[:, i], label='Test', s=5)
        plt.scatter(range(len(predictions.cpu())), predictions.cpu().numpy()[:, i], label='Prediction', s=5)
        plt.title(f'Test vs Prediction for {var}')
        plt.legend()
        
        safe_var_name = var.replace(':', '_')
        figure_file_path = os.path.join(output_folder, f'{safe_var_name}_test_vs_prediction.png')
        
        plt.savefig(figure_file_path)
        plt.close()

def plot_3d_scatter_comparison(features, actual, predicted, output_folder, variables):
    x = features['Points:0'].to_numpy()
    y = features['Points:1'].to_numpy()
    z = features['Points:2'].to_numpy()

    actual = actual.cpu().numpy()
    predicted = predicted.cpu().numpy()
    
    for i, var in enumerate(variables):
        # Create a subplot
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=(f'Actual {var}', f'Predicted {var}'),
                            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]])
        
        # Actual scatter
        scatter_actual = go.Scatter3d(x=x, y=y, z=z, mode='markers',
                                      marker=dict(size=2, color=actual[:, i], colorscale='Viridis', opacity=0.8, colorbar=dict(title=var, x=-0.07)),
                                      name='Actual')
        
        # Predicted scatter
        scatter_pred = go.Scatter3d(x=x, y=y, z=z, mode='markers',
                                    marker=dict(size=2, color=predicted[:, i], colorscale='Viridis', opacity=0.8, colorbar=dict(title=var, x=1.07)),
                                    name='Predicted')
        
        fig.add_trace(scatter_actual, row=1, col=1)
        fig.add_trace(scatter_pred, row=1, col=2)
        
        fig.update_layout(title=f"Comparison of Actual vs. Predicted {var} values")
        
        # To save the figure as an interactive HTML
        safe_var_name = var.replace(':', '_')
        fig.write_html(os.path.join(output_folder, f"{safe_var_name}_figure.html"))

def plot_2d_contour_comparison(features, actual, predicted, idx_test, output_folder, variables, variables_to_plot):  
    x_feature, y_feature = variables_to_plot[0]
    variable_to_plot = variables_to_plot[1]
    
    if variable_to_plot not in variables:
        raise ValueError(f"{variable_to_plot} not in provided variables")
    
    x = features['Points:0'].to_numpy()
    y = features['Points:1'].to_numpy()
    z = features['Points:2'].to_numpy()

    x = x[idx_test]
    y = y[idx_test]
    z = z[idx_test]
    
    # Apply mask based on the domain and the idx_test
    mask = (x >= 400) & (x <= 600) & (y >= 400) & (y <= 600) & (z >= 0) & (z <= 100)
    
    # Get the indices where the mask is True
    indices = np.where(mask)[0]

    # Intersect indices with idx_test
    indices = np.intersect1d(indices, idx_test)

    # Use the indices to filter x, y, z, z_actual, and z_predicted
    x = x[indices]
    y = y[indices]
    z = z[indices]
    z_actual = actual[indices, variables.index(variable_to_plot)].cpu().numpy()
    z_predicted = predicted[indices, variables.index(variable_to_plot)].cpu().numpy()

    # Creating a 2D grid and interpolating z values for each grid point
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
    xi, yi = np.linspace(x_plot.min(), x_plot.max(), 256), np.linspace(y_plot.min(), y_plot.max(), 256)
    xi, yi = np.meshgrid(xi, yi)
    
    zi_actual = griddata((x_plot, y_plot), z_actual, (xi, yi), method='cubic')
    zi_predicted = griddata((x_plot, y_plot), z_predicted, (xi, yi), method='cubic')
    
    # Creating the contour plots side by side
    fig, axs = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    
    # Actual contour plot
    c_actual = axs[0].contourf(xi, yi, zi_actual, extend='both', cmap=plt.cm.RdBu_r, levels=np.linspace(0,20,1000))
    axs[0].set_title(f'Actual {variable_to_plot}')
    fig.colorbar(c_actual, ax=axs[0], orientation="vertical", ticks=np.linspace(0,20,10, endpoint=True))
    
    # Predicted contour plot
    c_predicted = axs[1].contourf(xi, yi, zi_predicted, extend='both', cmap=plt.cm.RdBu_r, levels=np.linspace(0,20,1000))
    axs[1].set_title(f'Predicted {variable_to_plot}')
    fig.colorbar(c_predicted, ax=axs[1], orientation="vertical", ticks=np.linspace(0,20,10, endpoint=True))
    
    # Setting labels and title
    axs[0].set_xlabel(x_feature)
    axs[0].set_ylabel(y_feature)
    axs[1].set_xlabel(x_feature)
    axs[1].set_ylabel(y_feature)
    fig.suptitle(f'Comparison of Actual vs. Predicted {variable_to_plot} values')
    
    # Saving the figure
    safe_var_name = variable_to_plot.replace(':', '_')
    safe_x_feature = x_feature.replace(':', '')
    safe_y_feature = y_feature.replace(':', '')
    plt.savefig(os.path.join(output_folder, f"{safe_x_feature}_{safe_y_feature}_{safe_var_name}_contour_comparison.png"))
    
    # plt.show()
    
    plt.close()

def plot_total_velocity(features, actual, predicted, idx_test, output_folder, variables, variables_to_plot):    
    x_feature, y_feature = variables_to_plot[0]

    u_index = variables.index('Velocity:0')
    v_index = variables.index('Velocity:1')
    w_index = variables.index('Velocity:2')
    
    x = features['Points:0'].to_numpy()
    y = features['Points:1'].to_numpy()
    z = features['Points:2'].to_numpy()

    x = x[idx_test]
    y = y[idx_test]
    z = z[idx_test]
    
    # Apply mask based on the domain and the idx_test
    mask = (x >= 400) & (x <= 600) & (y >= 400) & (y <= 600) & (z >= 0) & (z <= 100)
    
    # Get the indices where the mask is True
    indices = np.where(mask)[0]

    # Intersect indices with idx_test
    indices = np.intersect1d(indices, idx_test)

    # Use the indices to filter x, y, z, and velocity components
    x = x[indices]
    y = y[indices]
    z = z[indices]
    u_actual = actual[indices, u_index].cpu().numpy()
    v_actual = actual[indices, v_index].cpu().numpy()
    w_actual = actual[indices, w_index].cpu().numpy()
    u_predicted = predicted[indices, u_index].cpu().numpy()
    v_predicted = predicted[indices, v_index].cpu().numpy()
    w_predicted = predicted[indices, w_index].cpu().numpy()
    
    # Compute the magnitude of the velocity vector
    velocity_actual = np.sqrt(u_actual**2 + v_actual**2 + w_actual**2)
    velocity_predicted = np.sqrt(u_predicted**2 + v_predicted**2 + w_predicted**2)

    z_actual = velocity_actual
    z_predicted = velocity_predicted

    variable_to_plot = 'Total Velocity'

    # Creating a 2D grid and interpolating z values for each grid point
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
    xi, yi = np.linspace(x_plot.min(), x_plot.max(), 256), np.linspace(y_plot.min(), y_plot.max(), 256)
    xi, yi = np.meshgrid(xi, yi)
    
    zi_actual = griddata((x_plot, y_plot), z_actual, (xi, yi), method='cubic')
    zi_predicted = griddata((x_plot, y_plot), z_predicted, (xi, yi), method='cubic')
    
    # Creating the contour plots side by side
    fig, axs = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    
    # Actual contour plot
    c_actual = axs[0].contourf(xi, yi, zi_actual, extend='both', cmap=plt.cm.RdBu_r, levels=np.linspace(0,20,1000))
    axs[0].set_title(f'Actual {variable_to_plot}')
    fig.colorbar(c_actual, ax=axs[0], orientation="vertical", ticks=np.linspace(0,20,10, endpoint=True))
    
    # Predicted contour plot
    c_predicted = axs[1].contourf(xi, yi, zi_predicted, extend='both', cmap=plt.cm.RdBu_r, levels=np.linspace(0,20,1000))
    axs[1].set_title(f'Predicted {variable_to_plot}')
    fig.colorbar(c_predicted, ax=axs[1], orientation="vertical", ticks=np.linspace(0,20,10, endpoint=True))
    
    # Setting labels and title
    axs[0].set_xlabel(x_feature)
    axs[0].set_ylabel(y_feature)
    axs[1].set_xlabel(x_feature)
    axs[1].set_ylabel(y_feature)
    fig.suptitle(f'Comparison of Actual vs. Predicted {variable_to_plot} values')
    
    # Saving the figure
    safe_var_name = variable_to_plot.replace(':', '_')
    safe_x_feature = x_feature.replace(':', '')
    safe_y_feature = y_feature.replace(':', '')
    plt.savefig(os.path.join(output_folder, f"{safe_x_feature}_{safe_y_feature}_{safe_var_name}_contour_comparison.png"))
    
    # plt.show()
    
    plt.close()

def individual_plot_predictions(wind_angle, y_test, predictions, output_folder, variables):
    for i, var in enumerate(variables):
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(y_test)), y_test.cpu().numpy()[:, i], label='Test', s=5)
        plt.scatter(range(len(predictions.cpu())), predictions.cpu().numpy()[:, i], label='Prediction', s=5)
        plt.title(f'Test vs Prediction for {var} with Wind Angle = {wind_angle}')
        plt.legend()

        plot_folder = os.path.join(output_folder, f'plots_{wind_angle}')
        os.makedirs(plot_folder, exist_ok=True)
        
        safe_var_name = var.replace(':', '_')
        figure_file_path = os.path.join(plot_folder, f'{safe_var_name}_test_vs_prediction_for_wind_angle_{wind_angle}.png')
        
        plt.savefig(figure_file_path)
        plt.close()

def individual_plot_3d_scatter_comparison(wind_angle, positions, actual, predicted, output_folder, variables):

    x = positions['x'].to_numpy()
    y = positions['y'].to_numpy()
    z = positions['z'].to_numpy()

    actual = actual.cpu().numpy()
    predicted = predicted.cpu().numpy()
    
    for i, var in enumerate(variables):
        # Create a subplot
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=(f'Actual {var}', f'Predicted {var}'),
                            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]])
        
        # Actual scatter
        scatter_actual = go.Scatter3d(x=x, y=y, z=z, mode='markers',
                                      marker=dict(size=2, color=actual[:, i], colorscale='Viridis', opacity=0.8, colorbar=dict(title=var, x=-0.07)),
                                      name='Actual')
        
        # Predicted scatter
        scatter_pred = go.Scatter3d(x=x, y=y, z=z, mode='markers',
                                    marker=dict(size=2, color=predicted[:, i], colorscale='Viridis', opacity=0.8, colorbar=dict(title=var, x=1.07)),
                                    name='Predicted')
        
        fig.add_trace(scatter_actual, row=1, col=1)
        fig.add_trace(scatter_pred, row=1, col=2)
        
        fig.update_layout(title=f"Comparison of Actual vs. Predicted {var} values with Wind Angle = {wind_angle}")

        plot_folder = os.path.join(output_folder, f'plots_{wind_angle}')
        os.makedirs(plot_folder, exist_ok=True)
        
        # To save the figure as an interactive HTML
        safe_var_name = var.replace(':', '_')
        fig.write_html(os.path.join(plot_folder, f"{safe_var_name}_figure_for_wind_angle_{wind_angle}.html"))

def individual_plot_2d_contour_comparison(wind_angle, positions, actual, predicted, output_folder, variables, variables_to_plot):  
    x_feature, y_feature = variables_to_plot[0]
    variable_to_plot = variables_to_plot[1]
    
    if variable_to_plot not in variables:
        raise ValueError(f"{variable_to_plot} not in provided variables")
    
    x = positions['x'].to_numpy()
    y = positions['y'].to_numpy()
    z = positions['z'].to_numpy()
    
    # Apply mask based on the domain and the idx_test
    mask = (x >= 400) & (x <= 600) & (y >= 400) & (y <= 600) & (z >= 0) & (z <= 100)
    
    # Get the indices where the mask is True
    indices = np.where(mask)[0]

    # Use the indices to filter x, y, z, z_actual, and z_predicted
    x = x[indices]
    y = y[indices]
    z = z[indices]
    z_actual = actual[indices, variables.index(variable_to_plot)].cpu().numpy()
    z_predicted = predicted[indices, variables.index(variable_to_plot)].cpu().numpy()

    # Creating a 2D grid and interpolating z values for each grid point
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
    xi, yi = np.linspace(x_plot.min(), x_plot.max(), 256), np.linspace(y_plot.min(), y_plot.max(), 256)
    xi, yi = np.meshgrid(xi, yi)
    
    zi_actual = griddata((x_plot, y_plot), z_actual, (xi, yi), method='cubic')
    zi_predicted = griddata((x_plot, y_plot), z_predicted, (xi, yi), method='cubic')
    
    # Creating the contour plots side by side
    fig, axs = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    
    # Actual contour plot
    c_actual = axs[0].contourf(xi, yi, zi_actual, extend='both', cmap=plt.cm.RdBu_r, levels=np.linspace(0,20,1000))
    axs[0].set_title(f'Actual {variable_to_plot}')
    fig.colorbar(c_actual, ax=axs[0], orientation="vertical", ticks=np.linspace(0,20,10, endpoint=True))
    
    # Predicted contour plot
    c_predicted = axs[1].contourf(xi, yi, zi_predicted, extend='both', cmap=plt.cm.RdBu_r, levels=np.linspace(0,20,1000))
    axs[1].set_title(f'Predicted {variable_to_plot}')
    fig.colorbar(c_predicted, ax=axs[1], orientation="vertical", ticks=np.linspace(0,20,10, endpoint=True))
    
    # Setting labels and title
    axs[0].set_xlabel(x_feature)
    axs[0].set_ylabel(y_feature)
    axs[1].set_xlabel(x_feature)
    axs[1].set_ylabel(y_feature)
    fig.suptitle(f'Comparison of Actual vs. Predicted {variable_to_plot} values with Wind Angle = {wind_angle}')

    plot_folder = os.path.join(output_folder, f'plots_{wind_angle}')
    os.makedirs(plot_folder, exist_ok=True)
    
    # Saving the figure
    safe_var_name = variable_to_plot.replace(':', '_')
    safe_x_feature = x_feature.replace(':', '')
    safe_y_feature = y_feature.replace(':', '')
    plt.savefig(os.path.join(plot_folder, f"{safe_x_feature}_{safe_y_feature}_{safe_var_name}_contour_comparison_for_wind_angle_{wind_angle}.png"))    
    plt.close()

def individual_plot_total_velocity(wind_angle, positions, actual, predicted, output_folder, variables, variables_to_plot):    
    x_feature, y_feature = variables_to_plot[0]

    u_index = variables.index('Velocity:0')
    v_index = variables.index('Velocity:1')
    w_index = variables.index('Velocity:2')
    
    x = positions['x'].to_numpy()
    y = positions['y'].to_numpy()
    z = positions['z'].to_numpy()
    
    # Apply mask based on the domain and the idx_test
    mask = (x >= 400) & (x <= 600) & (y >= 400) & (y <= 600) & (z >= 0) & (z <= 100)
    
    # Get the indices where the mask is True
    indices = np.where(mask)[0]

    # Use the indices to filter x, y, z, and velocity components
    x = x[indices]
    y = y[indices]
    z = z[indices]
    u_actual = actual[indices, u_index].cpu().numpy()
    v_actual = actual[indices, v_index].cpu().numpy()
    w_actual = actual[indices, w_index].cpu().numpy()
    u_predicted = predicted[indices, u_index].cpu().numpy()
    v_predicted = predicted[indices, v_index].cpu().numpy()
    w_predicted = predicted[indices, w_index].cpu().numpy()
    
    # Compute the magnitude of the velocity vector
    velocity_actual = np.sqrt(u_actual**2 + v_actual**2 + w_actual**2)
    velocity_predicted = np.sqrt(u_predicted**2 + v_predicted**2 + w_predicted**2)

    z_actual = velocity_actual
    z_predicted = velocity_predicted

    variable_to_plot = 'Total Velocity'

    # Creating a 2D grid and interpolating z values for each grid point
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
    xi, yi = np.linspace(x_plot.min(), x_plot.max(), 256), np.linspace(y_plot.min(), y_plot.max(), 256)
    xi, yi = np.meshgrid(xi, yi)
    
    zi_actual = griddata((x_plot, y_plot), z_actual, (xi, yi), method='cubic')
    zi_predicted = griddata((x_plot, y_plot), z_predicted, (xi, yi), method='cubic')
    
    # Creating the contour plots side by side
    fig, axs = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    
    # Actual contour plot
    c_actual = axs[0].contourf(xi, yi, zi_actual, extend='both', cmap=plt.cm.RdBu_r, levels=np.linspace(0,20,1000))
    axs[0].set_title(f'Actual {variable_to_plot}')
    fig.colorbar(c_actual, ax=axs[0], orientation="vertical", ticks=np.linspace(0,20,10, endpoint=True))
    
    # Predicted contour plot
    c_predicted = axs[1].contourf(xi, yi, zi_predicted, extend='both', cmap=plt.cm.RdBu_r, levels=np.linspace(0,20,1000))
    axs[1].set_title(f'Predicted {variable_to_plot}')
    fig.colorbar(c_predicted, ax=axs[1], orientation="vertical", ticks=np.linspace(0,20,10, endpoint=True))
    
    # Setting labels and title
    axs[0].set_xlabel(x_feature)
    axs[0].set_ylabel(y_feature)
    axs[1].set_xlabel(x_feature)
    axs[1].set_ylabel(y_feature)
    fig.suptitle(f'Comparison of Actual vs. Predicted {variable_to_plot} values with Wind Angle = {wind_angle}')
    
    plot_folder = os.path.join(output_folder, f'plots_{wind_angle}')
    os.makedirs(plot_folder, exist_ok=True)

    # Saving the figure
    safe_var_name = variable_to_plot.replace(':', '_')
    safe_x_feature = x_feature.replace(':', '')
    safe_y_feature = y_feature.replace(':', '')
    plt.savefig(os.path.join(plot_folder, f"{safe_x_feature}_{safe_y_feature}_{safe_var_name}_contour_comparison_for_wind_angle_{wind_angle}.png"))
    plt.close()

def main(base_directory, config, output_zip_file):
    overall_start_time = time.time()
    
    output_folder = os.path.join(base_directory, 'analyses_output')
    os.makedirs(output_folder, exist_ok=True)

    log_folder = os.path.join(output_folder, 'log_output')
    os.makedirs(log_folder, exist_ok=True)
    
    log_filename = os.path.join(log_folder,f"output_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    sys.stdout = Logger(log_filename)  # Set the logger as the new stdout
    
    device = print_and_set_available_gpus()

    chosen_machine_key = config["chosen_machine"]
    chosen_machine_config = config[chosen_machine_key]
    
    datafolder_path = chosen_machine_config["data_folder"]
    filenames = get_filenames_from_folder(datafolder_path, config["extension"], config["startname"])
    
    features, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, idx_train, idx_test = load_data(filenames, base_directory, datafolder_path, device, config)

    model = PINN().to(device)
    model_folder = os.path.join(output_folder, 'model_output')
    os.makedirs(model_folder, exist_ok=True)
    model_file_path = os.path.join(model_folder, 'trained_PINN_model.pth')
    
    ###TRAINING###
    if config["train_test"]["train"]:
        device, batch_size = select_device_and_batch_size(model, X_train_tensor)
        required_memory = estimate_memory(model, X_train_tensor, batch_size)
        print(f"Estimated Memory Requirement for the Batch Size: {required_memory / (1024 ** 3):.2f} GB")
        required_memory = estimate_memory(model, X_train_tensor, batch_size=len(X_train_tensor))
        print(f"Estimated Memory Requirement for the Full Size: {required_memory / (1024 ** 3):.2f} GB")
               
        # train and save the trained model
        model = train_model(model, X_train_tensor, y_train_tensor, config, batch_size, model_file_path, epochs=config["epochs"])
    ###TRAINING###

    ###TESTING###
    if config["train_test"]["test"]:
        # Evaluate the model
        checkpoint = torch.load(model_file_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        variables = ['Pressure', 'Velocity:0', 'Velocity:1', 'Velocity:2']
        data_folder = os.path.join(output_folder, 'data_output')
        os.makedirs(data_folder, exist_ok=True)

        predictions, test_predictions_wind_angle = evaluate_model(model, variables, features, idx_test, X_test_tensor, y_test_tensor, data_folder)

        predictions = predictions.cpu()

        plot_folder = os.path.join(output_folder, 'plots_output')
        os.makedirs(plot_folder, exist_ok=True)
        # Plot the test and all predictions for each variable and save the figures
        plot_predictions(y_test_tensor, predictions, plot_folder, variables)
        plot_3d_scatter_comparison(features, y_test_tensor, predictions, plot_folder, variables)
        list_of_directions = ['Points:0','Points:1','Points:2']
        variables_to_plot = get_variables_to_plot(list_of_directions, variables)
        for variable_to_plot in variables_to_plot:
            plot_2d_contour_comparison(features, y_test_tensor, predictions, idx_test, plot_folder, variables, variable_to_plot)
            plot_total_velocity(features, y_test_tensor, predictions, idx_test, plot_folder, variables, variable_to_plot)

        if config["make_individual_plots"]:
            for wind_angle, positions, y_test_tensor, predictions in test_predictions_wind_angle:
                individual_plot_predictions(wind_angle, y_test_tensor, predictions, plot_folder, variables)
                individual_plot_3d_scatter_comparison(wind_angle, positions, y_test_tensor, predictions, plot_folder, variables)
                list_of_directions = ['Points:0','Points:1','Points:2']
                variables_to_plot = get_variables_to_plot(list_of_directions, variables)
                for variable_to_plot in variables_to_plot:
                    individual_plot_2d_contour_comparison(wind_angle, positions, y_test_tensor, predictions, plot_folder, variables, variable_to_plot)
                    individual_plot_total_velocity(wind_angle, positions, y_test_tensor, predictions, plot_folder, variables, variable_to_plot)
    ###TESTING###

    ###EVALUATING###
    if config["train_test"]["evaluate"]:
        features, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, idx_train, idx_test = load_skipped_angle_data(filenames, base_directory, datafolder_path, device, config)
        # Evaluate the model
        checkpoint = torch.load(model_file_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        variables = ['Pressure', 'Velocity:0', 'Velocity:1', 'Velocity:2']
        data_folder = os.path.join(output_folder, 'data_output_for_skipped_angle')
        os.makedirs(data_folder, exist_ok=True)
        plot_folder = os.path.join(output_folder, 'plots_output_for_skipped_angle')
        os.makedirs(plot_folder, exist_ok=True)

        predictions, _ = evaluate_model(model, variables, features, idx_test, X_test_tensor, y_test_tensor, data_folder)
        predictions = predictions.cpu()
        # Plot the test and all predictions for each variable and save the figures
        plot_predictions(y_test_tensor, predictions, plot_folder, variables)
        plot_3d_scatter_comparison(features, y_test_tensor, predictions, plot_folder, variables)
        list_of_directions = ['Points:0','Points:1','Points:2']
        variables_to_plot = get_variables_to_plot(list_of_directions, variables)
        for variable_to_plot in variables_to_plot:
            plot_2d_contour_comparison(features, y_test_tensor, predictions, idx_test, plot_folder, variables, variable_to_plot)
            plot_total_velocity(features, y_test_tensor, predictions, idx_test, plot_folder, variables, variable_to_plot)
    ###EVALUATING###

    shutil.make_archive(output_zip_file[:-4], 'zip', output_folder)

    overall_end_time = time.time()
    overall_elapsed_time = overall_end_time - overall_start_time
    print(f'Overall process completed in {overall_elapsed_time:.2f} seconds')

if __name__ == "__main__":
    if config["chosen_machine"] == "lxplus":
        if len(sys.argv) > 2:
            base_directory = sys.argv[1]
            output_zip_file = sys.argv[2]
            main(base_directory, config, output_zip_file)
        else:
            print("Please provide the base directory and output zip file as arguments.")
    else:
        base_directory = os.getcwd()
        output_zip_file = os.path.join(base_directory,'output.zip')
        main(base_directory, config, output_zip_file)    