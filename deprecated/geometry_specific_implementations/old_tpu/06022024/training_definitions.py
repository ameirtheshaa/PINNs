from definitions import *
from PINN import *
from losses import *

def calculate_total_loss(model, config, X, y):
    total_loss = 0

    if config["loss_components"]["data_loss"]:
        data_loss = compute_data_loss(model, X, y)
    else:
        data_loss = 0

    total_loss += data_loss
    
    losses = [total_loss, data_loss]

    return total_loss, losses

def save_model(strategy, model, model_file_path, epoch, current_loss, training_completed, optimizer, start_time):
    success = False
    error_count = 0
    if strategy:
        model_file_path_new = 'gs://ameir/trained_PINN_model/'
    else:
        model_file_path_new = model_file_path    
    os.makedirs(model_file_path_new, exist_ok=True)
    while not success and error_count < 10:
        try:
            model.save(model_file_path_new)
            # checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
            # checkpoint.save(checkpoint_folder)
            additional_state = {
                'epoch': epoch,
                # 'current_loss': current_loss,
                'training_completed': training_completed
            }
            with open(f'{model_file_path}.json', 'w') as f:
                json.dump(additional_state, f)
            success = True  # If it reaches this point, the block executed without errors, so we set success to True
        except Exception as e:
            error_count += 1
            print(f"An error occurred (Attempt #{error_count}): {e}. Trying again..., time: {(time.time() - start_time):.2f} seconds")

def load_model(strategy, model, model_file_path, optimizer):
    if strategy:
        model_file_path_new = 'gs://ameir/trained_PINN_model/'
    else:
        model_file_path_new = model_file_path    
    model = tf.keras.models.load_model(model_file_path_new)
    with open(f'{model_file_path}.json', 'r') as f:
        additional_state = json.load(f)
    epoch = additional_state['epoch']
    # current_loss = additional_state['current_loss']
    training_completed = additional_state['training_completed']
    return model, epoch, training_completed

def get_available_device_memory():
    gpus = tf.config.list_physical_devices('GPU')
    gpu_memory_info = []
    if gpus:
        try:
            pynvml.nvmlInit()
            for gpu in gpus:
                index = gpus.index(gpu)
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                free_memory = info.free / (1024 ** 3)  # Convert bytes to gigabytes
                total_memory = info.total / (1024 ** 3)
                gpu_memory_info.append((free_memory, total_memory))
            pynvml.nvmlShutdown()
        except pynvml.NVMLError as error:
            print(f"Failed to get NVML GPU info: {error}")
    else:
        cpu_memory = psutil.virtual_memory()
        total_memory = cpu_memory.total / (1024 ** 3)  # Convert bytes to gigabytes
        free_memory = cpu_memory.available / (1024 ** 3)
        gpu_memory_info.append((free_memory, total_memory))
    return gpu_memory_info

def evaluate_model_training(strategy, model, model_file_path, X_test, y_test):
    if strategy:
        model_file_path_new = 'gs://ameir/trained_PINN_model/'
    else:
        model_file_path_new = model_file_path   
    model = tf.keras.models.load_model(model_file_path_new)
    model.evaluate(X_test, y_test, verbose=2)
    predictions = model.predict(X_test)
    mses = []
    r2s = []
    for i in range(predictions.shape[1]):
        predictions_flat = predictions[:, i].flatten()
        # y_test_flat = y_test[:, i].flatten()
        y_test_flat = y_test[:, i]
        # y_test_flat = y_test
        mse = sklearn.metrics.mean_squared_error(y_test_flat, predictions_flat)
        r2 = sklearn.metrics.r2_score(y_test_flat, predictions_flat)
        mses.append(mse)
        r2s.append(r2)
    return mses, r2s

def save_evaluation_results(strategy, config, model, model_file_path, log_folder, X_test_tensor, y_test_tensor, epoch, epochs, use_epoch, save_name):
    mses, r2s = evaluate_model_training(strategy, model, model_file_path, X_test_tensor, y_test_tensor)
    output_params = config["training"]["output_params_modf"]
    mse_strings = [f'MSE {param}: {mse}' for param, mse in zip(output_params, mses)]
    mse_output = ', '.join(mse_strings)
    r2_strings = [f'r2 {param}: {r2}' for param, r2 in zip(output_params, r2s)]
    r2_output = ', '.join(r2_strings)
    print(f'Epoch [{epoch}/{epochs if use_epoch else "infinity"}], {mse_output}, {r2_output} for {save_name} Data - {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    save_evaluation_to_csv(epoch, epochs, use_epoch, output_params, mses, r2s, file_path=os.path.join(log_folder,f'{save_name}'))