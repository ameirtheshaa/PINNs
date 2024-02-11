from definitions import *
from boundary import *
from PINN import *
from weighting import *
from training_definitions import *

def train_model(model, device, config, data_dict, model_file_path, log_folder):
    model.train()

    start_time = time.time()
    logging_info_dir = os.path.join(log_folder, 'info.csv')
    logging_info_dir = Path(logging_info_dir)
    if os.path.exists(logging_info_dir):
        df = pd.read_csv(logging_info_dir)
        time_passed = df['Total Time Elapsed (hours)'].iloc[-1]
    else:
        time_passed = None
    print(f'starting to train, time: {(get_time_elapsed(start_time, time_passed)):.2f} seconds')

    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, labels_train_tensor, X_train_tensor_skipped, y_train_tensor_skipped, X_test_tensor_skipped, y_test_tensor_skipped, feature_scaler, target_scaler, X_train_divergences_tensor, X_test_divergences_tensor, y_train_divergences_tensor, y_test_divergences_tensor = data_dict.values()
    
    chosen_machine_key = config["chosen_machine"]
    datafolder_path = os.path.join(config["machine"][chosen_machine_key], config["data"]["data_folder_name"])
    meteo_filenames = get_filenames_from_folder(datafolder_path, config["data"]["extension"], config["data"]["startname_meteo"])

    model_file_path = Path(model_file_path)
    use_epoch = config["training"]["use_epochs"]
    epochs = config["training"]["num_epochs"]
    wind_angles = config["training"]["angles_to_train"]
    
    sma = 0
    sma_window_size = 1000
    sma_threshold = config["training"]["loss_diff_threshold"]
    consecutive_sma_threshold = config["training"]["consecutive_count_threshold"]
    recent_losses = collections.deque(maxlen=sma_window_size)
    consecutive_sma_count = 0
    early_stop = False  # Flag to break out of both loops

    chosen_optimizer_key = config["chosen_optimizer"]
    optimizer_config = config[chosen_optimizer_key]
    optimizer = get_optimizer(model, chosen_optimizer_key, optimizer_config)
    print(f'using {chosen_optimizer_key}, time: {(get_time_elapsed(start_time, time_passed)):.2f} seconds')

    if config["loss_components"]["no_slip_loss"]:
        no_slip_features_normals = no_slip_boundary_conditions(config, device, feature_scaler)
    else:
        no_slip_features_normals = None

    if config["loss_components"]["inlet_loss"]:
        all_inlet_features_targets = sample_domain_boundary(config, device, feature_scaler, target_scaler)
    else:
        all_inlet_features_targets = None

    if config["training"]["use_batches"]:
        optimal_batch_size = config["training"]["batch_size"]
        train_loader = get_train_loader(X_train_tensor, y_train_tensor, labels_train_tensor, optimal_batch_size, wind_angles)
        print(f'starting batch training with batch size = {optimal_batch_size}, time: {(get_time_elapsed(start_time, time_passed)):.2f} seconds')
    else:
        optimal_batch_size = len(X_train_tensor)
        train_loader = get_train_loader(X_train_tensor, y_train_tensor, labels_train_tensor, optimal_batch_size, wind_angles)
        print(f'starting training with full size = {optimal_batch_size}, time: {(get_time_elapsed(start_time, time_passed)):.2f} seconds')

    if os.path.exists(model_file_path):
        print(f"continuing from last checkpoint at {model_file_path}..., time: {(get_time_elapsed(start_time, time_passed)):.2f} seconds")
        try:
            checkpoint = open_model_file(model_file_path, device)
        except RuntimeError as e:
            print(f"Error loading checkpoint: {e}")
        state_dict = checkpoint['model_state_dict']
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        training_completed = checkpoint.get('training_completed', False)  # Default to False if the flag is not in the saved state
        recent_losses, sma, consecutive_sma_count, time_passed_ = load_losses_from_json(os.path.join(log_folder,'sma.json'), sma_window_size)
        if training_completed:
            print(f"Training has already been completed., time: {(get_time_elapsed(start_time, time_passed)):.2f} seconds")
            return model
        else:
            start_epoch = checkpoint['epoch']
            print(f"continuing from last checkpoint... starting from epoch = {start_epoch}, time: {(get_time_elapsed(start_time, time_passed)):.2f} seconds")
    else:
        start_epoch = 1

    for epoch in itertools.count(start=start_epoch):  # infinite loop, will only break if the stopping condition is met.
        if early_stop or (use_epoch and epoch > epochs):
            training_completed = True
            save_model(model, model_file_path, epoch, current_loss, training_completed, optimizer, start_time)
            break
        else:
            training_completed = False

        print_lines = 0
        for Xy_batch, labels in train_loader:
            X_batch = Xy_batch[0]
            y_batch = Xy_batch[1]
            if y_train_divergences_tensor is not None:
                y_batch_divergences = y_train_divergences_tensor[labels]
            else:
                y_batch_divergences = None

            def closure():
                optimizer.zero_grad()
                predictions = model(X_batch)
                total_loss = calculate_total_loss(X_batch, y_batch, epoch, config, model, optimizer, True, no_slip_features_normals, all_inlet_features_targets, y_batch_divergences)
                total_loss.backward()
                return total_loss

            current_loss = optimizer.step(closure)
            recent_losses.append(current_loss.cpu().detach().numpy())
            if len(recent_losses) == sma_window_size:
                sma = sum(recent_losses) / sma_window_size
                if sma < sma_threshold:
                    consecutive_sma_count += 1
                    if consecutive_sma_count >= consecutive_sma_threshold:
                        print(f"SMA of loss below {sma_threshold} for {consecutive_sma_threshold} consecutive epochs at epoch {epoch}. Stopping training..., time: {(get_time_elapsed(start_time, time_passed)):.2f} seconds")
                        early_stop = True
                        break
                    else:
                        consecutive_sma_count = 0
                else:
                    consecutive_sma_count = 0

            if epoch % config["training"]["print_epochs"] == 0:
                if print_lines < 1:
                    current_elapsed_time = get_time_elapsed(start_time, time_passed)
                    current_elapsed_time_hours = current_elapsed_time / 3600
                    free_memory, _ = get_available_device_memory(device.index)
                    loss_dict = calculate_total_loss(X_batch, y_batch, epoch, config, model, optimizer, False, no_slip_features_normals, all_inlet_features_targets)
                    print_statement(epoch, epochs, use_epoch, current_loss, loss_dict, free_memory, current_elapsed_time_hours)
                    save_to_csv(epoch, epochs, use_epoch, current_elapsed_time_hours, free_memory, loss_dict, file_path=os.path.join(log_folder, 'info.csv'))
                    save_losses_to_json(recent_losses, sma, consecutive_sma_count, current_elapsed_time, os.path.join(log_folder,'sma.json'))
                    save_model(model, model_file_path, epoch, current_loss, training_completed, optimizer, start_time)
                    save_evaluation_results(config, device, model, model_file_path, log_folder, X_test_tensor, y_test_tensor, epoch, epochs, use_epoch, save_name='info_test.csv')
                    save_evaluation_results(config, device, model, model_file_path, log_folder, X_test_tensor_skipped, y_test_tensor_skipped, epoch, epochs, use_epoch, save_name='info_skipped.csv')
                    print_lines += 1

    total_elapsed_time = get_time_elapsed(start_time, time_passed)
    total_elapsed_time_hours = total_elapsed_time / 3600
    save_model(model, model_file_path, epoch, current_loss, training_completed, optimizer, start_time)
    print(f'Training completed in {total_elapsed_time_hours:.2f} hours')

    return model