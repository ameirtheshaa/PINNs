from definitions import *
from boundary import *
from PINN import *
from weighting import *
from training_definitions import *

def train_model(model, device, config, model_file_path, log_folder):
    model = model.to(device)
    model.train()

    chosen_machine_key = config["chosen_machine"]
    datafolder_path = os.path.join(config["machine"][chosen_machine_key], config["data"]["data_folder_name"])
    meteo_filenames = get_filenames_from_folder(datafolder_path, config["data"]["extension"], config["data"]["startname_meteo"])

    start_time = time.time()
    print(f'starting to train, time: {(time.time() - start_time):.2f} seconds')

    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, labels_train_tensor, X_train_tensor_skipped, y_train_tensor_skipped, X_test_tensor_skipped, y_test_tensor_skipped, X_train_tensor_data, y_train_tensor_data, X_test_tensor_data, y_test_tensor_data, feature_scaler, target_scaler = load_data(config, device)

    model_file_path = Path(model_file_path)
    chosen_optimizer_key = config["chosen_optimizer"]
    use_epoch = config["training"]["use_epochs"]
    epochs = config["training"]["num_epochs"]
    wind_angles = config["training"]["angles_to_train"]
    sma_window_size = 50
    sma_threshold = config["training"]["loss_diff_threshold"]
    consecutive_sma_threshold = config["training"]["consecutive_count_threshold"]
    recent_losses = collections.deque(maxlen=sma_window_size)
    consecutive_sma_count = 0
    early_stop = False  # Flag to break out of both loops
    current_losses = []

    chosen_optimizer_key = config["chosen_optimizer"]
    optimizer_config = config[chosen_optimizer_key]
    optimizer = get_optimizer(model, chosen_optimizer_key, optimizer_config)
    print(f'using {chosen_optimizer_key}, time: {(time.time() - start_time):.2f} seconds')

    if config["training"]["use_batches"]:
        optimal_batch_size = config["training"]["batch_size"]
        train_loader = get_train_loader(X_train_tensor, y_train_tensor, labels_train_tensor, optimal_batch_size, wind_angles)
        print(f'starting batch training with batch size = {optimal_batch_size}, time: {(time.time() - start_time):.2f} seconds')
    else:
        optimal_batch_size = len(X_train_tensor)
        train_loader = get_train_loader(X_train_tensor, y_train_tensor, labels_train_tensor, optimal_batch_size, wind_angles)
        print(f'starting training with full size = {optimal_batch_size}, time: {(time.time() - start_time):.2f} seconds')

    if os.path.exists(model_file_path):
        print(f"continuing from last checkpoint at {model_file_path}..., time: {(time.time() - start_time):.2f} seconds")
        try:
            checkpoint = open_model_file(model_file_path, device)
        except RuntimeError as e:
            print(f"Error loading checkpoint: {e}")
        state_dict = checkpoint['model_state_dict']
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        training_completed = checkpoint.get('training_completed', False)  # Default to False if the flag is not in the saved state
        if training_completed:
            print(f"Training has already been completed., time: {(time.time() - start_time):.2f} seconds")
            return model
        else:
            start_epoch = checkpoint['epoch']
            print(f"continuing from last checkpoint... starting from epoch = {start_epoch}, time: {(time.time() - start_time):.2f} seconds")
    else:
        start_epoch = 1

    for epoch in itertools.count(start=start_epoch):  # infinite loop, will only break if the stopping condition is met.
        if early_stop or (use_epoch and epoch > epochs):
            training_completed = True
            save_model(model, model_file_path, epoch, current_loss, training_completed, optimizer)
            break
        else:
            training_completed = False

        print_lines = 0
        for Xy_batch, labels in train_loader:
            X_batch = Xy_batch[0]
            y_batch = Xy_batch[1]

            X_batch_data = X_train_tensor_data[labels]
            y_batch_data = y_train_tensor_data[labels]
            
            def closure():
                nonlocal current_losses
                optimizer.zero_grad()
                predictions = model(X_batch)
                total_loss, losses = calculate_total_loss(X_batch, y_batch, X_batch_data, y_batch_data, epoch, config, device, model, optimizer, datafolder_path, meteo_filenames, feature_scaler, target_scaler)
                current_losses[:] = losses
                total_loss.backward()
                return total_loss

            current_loss = optimizer.step(closure)
            recent_losses.append(current_loss)
            if len(recent_losses) == sma_window_size:
                sma = sum(recent_losses) / sma_window_size
                if sma < sma_threshold:
                    consecutive_sma_count += 1
                    if consecutive_sma_count >= consecutive_sma_threshold:
                        print(f"SMA of loss below {sma_threshold} for {consecutive_sma_threshold} consecutive epochs at epoch {epoch}. Stopping training..., time: {(time.time() - start_time):.2f} seconds")
                        early_stop = True
                        break
                    else:
                        consecutive_sma_count = 0
                else:
                    consecutive_sma_count = 0

            if epoch % 1 == 0:
                if print_lines < 1:
                    current_elapsed_time = time.time() - start_time
                    current_elapsed_time_hours = current_elapsed_time / 3600
                    free_memory, _ = get_available_device_memory(device.index)
                    total_loss, total_loss_weighted, data_loss, cont_loss, momentum_loss, no_slip_loss, inlet_loss = current_losses
                    print(f'Epoch [{epoch}/{epochs if use_epoch else "infinity"}], Loss: {current_loss}, Total Time Elapsed: {current_elapsed_time_hours:.2f} hours, with free memory: {free_memory:.2f} GB; data_loss = {data_loss}, cont_loss = {cont_loss}, momentum_loss = {momentum_loss}, no_slip_loss = {no_slip_loss}, inlet_loss = {inlet_loss},  total_loss = {total_loss}, total_loss_weighted = {total_loss_weighted} - {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
                    save_to_csv(epoch, epochs, use_epoch, current_loss, current_elapsed_time_hours, free_memory, data_loss, cont_loss, momentum_loss, no_slip_loss, inlet_loss, total_loss, total_loss_weighted, file_path=os.path.join(log_folder, 'info.csv'))
                    save_model(model, model_file_path, epoch, current_loss, training_completed, optimizer)
                    save_evaluation_results(config, device, model, model_file_path, log_folder, X_test_tensor, y_test_tensor, epoch, epochs, use_epoch, save_name='info_test.csv')
                    save_evaluation_results(config, device, model, model_file_path, log_folder, X_test_tensor_skipped, y_test_tensor_skipped, epoch, epochs, use_epoch, save_name='info_skipped.csv')
                    print_lines += 1

    end_time = time.time()
    total_elapsed_time = end_time - start_time
    total_elapsed_time_hours = total_elapsed_time / 3600
    save_model(model, model_file_path, epoch, current_loss, training_completed, optimizer)
    print(f'Training completed in {total_elapsed_time_hours:.2f} hours')

    return model