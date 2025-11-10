from definitions import *
from PINN import *


def train_model(rank, world_size, model, device, X_train_tensor, y_train_tensor, config, batch_size, model_file_path, log_folder, epochs):
    if config["train_test"]["distributed_training"]:
        # Initialize distributed training environment
        ip_address_master = config["distributed_training"]["ip_address_master"]
        port_number = config["distributed_training"]["port_number"]
        backend = config["distributed_training"]["backend"]
        init_method = f'tcp://{ip_address_master}:{port_number}'
        dist.init_process_group(backend=backend, init_method=init_method, rank=rank, world_size=world_size)
        model = model.to(device)
        if torch.cuda.is_available():
            model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)
        else:
            model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    else:
        model = model.to(device)

    # Capture start time
    start_time = time.time()
    model_file_path = Path(model_file_path)  # Convert the string to a Path object
    chosen_optimizer_key = config["chosen_optimizer"]
    chosen_optimizer_config = config[chosen_optimizer_key]
    optimizer = get_optimizer(model, chosen_optimizer_config)
    model.train()

    def calculate_total_loss(X, y, config):
        total_loss = 0
        if config["train_test"]["distributed_training"]:
            if config["loss_components"]["data_loss"]:
                data_loss = model.module.compute_data_loss(X, y)
                total_loss += data_loss
            if config["loss_components"]["momentum_loss"]:
                momentum_loss = model.module.compute_physics_momentum_loss(X)
                total_loss += momentum_loss
            if config["loss_components"]["cont_loss"]:
                cont_loss = model.module.compute_physics_cont_loss(X)
                total_loss += cont_loss
        else:
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
        if config["train_test"]["distributed_training"]:
            predictions = model.module(X_batch)
        else:
            predictions = model(X_batch)
        total_loss = calculate_total_loss(X_batch, y_batch, config)
        total_loss.backward()
        return total_loss
    
    if config["training"]["use_batches"]:
        if not config["batches"]["force"]:
            maximum_batch_size = batch_size
            train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
            while True:
                train_loader_temp = torch.utils.data.DataLoader(train_dataset, batch_size=maximum_batch_size, shuffle=True)
                X_batch, y_batch = next(iter(train_loader_temp))  # take a single batch
                try:
                    loss = closure()
                    optimizer.step()
                    maximum_batch_size += int(0.001 * len(X_train_tensor))  # Increase and try again
                    available_gpu_memory, _ = get_available_device_memory(device.index)
                    print(f"Determining maximum batch size: {maximum_batch_size} with free memory = {available_gpu_memory:.2f} GB")
                except StopIteration:
                    break
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        maximum_batch_size -= int(0.001 * len(X_train_tensor))  # The last successful batch size
                        break
                    raise e 
            print(f"Maximum batch size found: {maximum_batch_size}")
            available_gpu_memory, total_gpu_memory = get_available_device_memory(device.index)
            alpha = calculate_alpha(available_gpu_memory, total_gpu_memory)
            optimal_batch_size = int(alpha * maximum_batch_size)
            print(f"Optimal batch size: {optimal_batch_size} with alpha = {alpha}")
            if config["train_test"]["distributed_training"]:
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=optimal_batch_size, sampler=train_sampler)
            else:
                train_loader = torch.utils.data.DataLoader(train_dataset, optimal_batch_size, shuffle=True)
            required_memory_optimal = estimate_memory(model, X_train_tensor, optimal_batch_size)
            print(f"Using batches with batch size: {optimal_batch_size} with Estimated Memory: {required_memory_optimal} GB")
        else:
            train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
            optimal_batch_size = config["batches"]["batch_size"]
            if config["train_test"]["distributed_training"]:
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=optimal_batch_size, sampler=train_sampler)
            else:
                train_loader = torch.utils.data.DataLoader(train_dataset, optimal_batch_size, shuffle=True)
            required_memory_forced = estimate_memory(model, X_train_tensor, optimal_batch_size)
            print(f"Using batches with forced batch size: {optimal_batch_size} with Estimated Memory: {required_memory_forced} GB")
    else:
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        optimal_batch_size = len(X_train_tensor)
        if config["train_test"]["distributed_training"]:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=optimal_batch_size, sampler=train_sampler)
        else:
            train_loader = torch.utils.data.DataLoader(train_dataset, optimal_batch_size, shuffle=True)
        print(f"Using full tensor with size: {optimal_batch_size}")

    previous_loss = None
    use_epoch = config["training"]["use_epochs"]
    loss_diff_threshold = config["training"]["loss_diff_threshold"]
    consecutive_count_threshold = config["training"]["consecutive_count_threshold"]
    consecutive_count = 0
    early_stop = False  # Flag to break out of both loops

    if os.path.exists(model_file_path):
        print("continuing from last checkpoint...")
        checkpoint = torch.load(model_file_path, map_location=device)
        state_dict_keys = list(checkpoint['model_state_dict'].keys())
        if config["train_test"]["distributed_training"]:
            if any(key.startswith("module.module.") for key in state_dict_keys):
                modified_state_dict = {k.replace("module.module.", "module."): v for k, v in checkpoint['model_state_dict'].items()}
            elif not any(key.startswith("module.") for key in state_dict_keys):
                modified_state_dict = {"module." + k: v for k, v in checkpoint['model_state_dict'].items()}
            else:
                modified_state_dict = checkpoint['model_state_dict']
            model.load_state_dict(modified_state_dict)
        else:
            if any(key.startswith("module.") for key in state_dict_keys):
                modified_state_dict = {k.replace("module.", ""): v for k, v in checkpoint['model_state_dict'].items()}
            else:
                modified_state_dict = checkpoint['model_state_dict']
            model.load_state_dict(modified_state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        training_completed = checkpoint.get('training_completed', False)  # Default to False if the flag is not in the saved state
        if training_completed:
            print("Training has already been completed.")
            return model
        else:
            start_epoch = checkpoint['epoch'] + 1
            print(f"continuing from last checkpoint... starting from epoch = {start_epoch}")
    else:
        start_epoch = 1
    for epoch in itertools.count(start=start_epoch):  # infinite loop, will only break if the stopping condition is met.
        if early_stop or (use_epoch and epoch > epochs):
            training_completed = True
            success = False
            error_count = 0
            while not success:
                try:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': current_loss,
                        'training_completed': training_completed
                    }, model_file_path)
                    success = True  # If it reaches this point, the block executed without errors, so we set success to True
                except Exception as e:
                    error_count += 1
                    print(f"An error occurred (Attempt #{error_count}): {e}. Trying again...")
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
            if epoch % 5 == 0:
                current_elapsed_time = time.time() - start_time
                current_elapsed_time_hours = current_elapsed_time / 3600
                free_memory, _ = get_available_device_memory(device.index)
                print(f'Epoch [{epoch}/{epochs if use_epoch else "infinity"}], Loss: {current_loss}, Total Time elapsed: {current_elapsed_time_hours:.2f} hours, with free memory: {free_memory:.2f} GB')
                success = False
                error_count = 0
                while not success:
                    try:
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': current_loss,
                            'training_completed': training_completed
                        }, model_file_path)
                        success = True  # If it reaches this point, the block executed without errors, so we set success to True
                    except Exception as e:
                        error_count += 1
                        print(f"An error occurred (Attempt #{error_count}): {e}. Trying again...")
            previous_loss = current_loss

    # Capture end time
    end_time = time.time()
    total_elapsed_time = end_time - start_time
    total_elapsed_time_hours = total_elapsed_time / 3600
    print(f'Training completed in {total_elapsed_time_hours:.2f} hours')

    success = False
    error_count = 0
    while not success:
        try:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': current_loss,
                'training_completed': training_completed
            }, model_file_path)
            success = True  # If it reaches this point, the block executed without errors, so we set success to True
        except Exception as e:
            error_count += 1
            print(f"An error occurred (Attempt #{error_count}): {e}. Trying again...")

    return model