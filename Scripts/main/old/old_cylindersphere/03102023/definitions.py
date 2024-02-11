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