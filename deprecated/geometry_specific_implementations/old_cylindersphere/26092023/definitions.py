import sys
import os
import torch
import time
import datetime
import psutil
import argparse
import sklearn
import numpy as np
import pandas as pd
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
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No GPUs available.")

    # Set device to GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

def load_data(filename, base_directory, datafolder_path, device, config):

    # Extract the base name of the file without the extension
    base_filename = os.path.splitext(filename)[0]
    
    # Create a new directory with the base filename
    output_folder = os.path.join(base_directory, base_filename)
    os.makedirs(output_folder, exist_ok=True)

    index_str = filename.split('_')[-1].split('.')[0]  # Extract the index part of the filename
    index = int(index_str)  # Convert the index to integer

    meteo_data = pd.read_csv(os.path.join(datafolder_path,'meteo.csv'))

    # Look up the corresponding row in the meteo.csv file
    meteo_row = meteo_data[meteo_data['index'] == index]

    # Extract the wind angle from the found row
    wind_angle = meteo_row['cs degree'].values[0]  

    rho = config["density"]
    nu = config["kinematic_viscosity"]

    # Load the data
    data = pd.read_csv(os.path.join(datafolder_path,filename))

    # Extract features from the dataframe
    features = data[['Points:0', 'Points:1', 'Points:2', 'TurbVisc']]
    features = features.assign(WindAngle=wind_angle, 
                           abs_cos_theta=np.abs(np.cos(np.deg2rad(wind_angle))),
                           abs_sin_theta=np.abs(np.sin(np.deg2rad(wind_angle))),
                           Density=rho,
                           Kinematic_Viscosity=nu)

    targets = data[['Pressure', 'Velocity:0', 'Velocity:1', 'Velocity:2']]

    # Initialize Standard Scalers for features and targets
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    # Fit the scalers and transform the features and targets
    normalized_features = feature_scaler.fit_transform(features)
    normalized_targets = target_scaler.fit_transform(targets)

    # Perform the train-test split and get the indices
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(normalized_features, normalized_targets, range(len(normalized_features)),test_size=0.2, random_state=42)

    # Convert to PyTorch Tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    X_train_tensor = X_train_tensor.to(device)
    y_train_tensor = y_train_tensor.to(device)
    X_test_tensor = X_test_tensor.to(device)
    y_test_tensor = y_test_tensor.to(device)

    return features, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, idx_train, idx_test, output_folder

def train_model(model, X_train_tensor, y_train_tensor, config, batch_size, epochs):
    # Capture start time
    start_time = time.time()
    
    if config["use_batches"]:
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
        print (f"using batches with batch size: {batch_size}")
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
    loss_diff_threshold = config["loss_diff_threshold"]
    consecutive_count_threshold = config["consecutive_count_threshold"]
    consecutive_count = 0
    early_stop = False  # Flag to break out of both loops

    for epoch in range(epochs):
        if early_stop:
            break  # Break the outer loop if the early_stop flag is True
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
                print(f'Epoch [{epoch}/{epochs}], Loss: {current_loss:.4f}')
            
            previous_loss = current_loss
    
    # Capture end time
    end_time = time.time()
    elapsed_time = end_time - start_time  # Calculate elapsed time
    print(f'Training completed in {elapsed_time:.2f} seconds')
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

    return metrics_df, predictions