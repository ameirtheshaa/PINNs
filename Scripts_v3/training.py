from definitions import *
from boundary import *
from PINN import *
from weighting import *

def train_model(rank, world_size, model, activation_function, device, X_train_tensor, y_train_tensor, config, batch_size, model_file_path, log_folder, epochs):
    if config["train_test"]["distributed_training"]:
        # Initialize distributed training environment
        backend = config["distributed_training"]["backend"]
        if config["distributed_training"]["use_tcp"]:
            ip_address_master = config["distributed_training"]["ip_address_master"]
            port_number = config["distributed_training"]["port_number"]
            init_method = f'tcp://{ip_address_master}:{port_number}'
            if config["distributed_training"]["master_node"]:
                print("Master node is initializing the distributed training environment...")
            else:
                print(f"Worker node {rank} is initializing...")
            dist.init_process_group(backend=backend, init_method=init_method, rank=rank, world_size=world_size)
            if config["distributed_training"]["master_node"]:
                print("Master node has completed initialization!")
            else:
                print(f"Worker node {rank} has completed initialization!")
        else:
            shared_file_path = f"{log_folder}/sharedfile_for_distributed_training"
            if rank == 0 and Path(shared_file_path).exists():
                print ("removing current shared file path...")
                os.remove(shared_file_path)
            init_method = f'file://{shared_file_path}'
            print(f"Process {rank} is initializing the distributed training environment...")
            print(f"Process {rank} is waiting for all workers to join...")
            dist.init_process_group(backend=backend, init_method=init_method, rank=rank, world_size=world_size)     
            print(f"Process {rank} has completed initialization!")
        model = model.to(device)
        if torch.cuda.is_available():
            model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)
        else:
            model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    else:
        model = model.to(device)

    # Capture start time
    start_time = time.time()
    print (f'starting to train, time: {(time.time() - start_time):.2f} seconds') 
    model_file_path = Path(model_file_path)  # Convert the string to a Path object
    chosen_optimizer_key = config["chosen_optimizer"]
    chosen_optimizer_config = config[chosen_optimizer_key]
    if chosen_optimizer_key == "both_optimizers":
        optimizer_adam, optimizer_lbfgs = get_optimizer(model, chosen_optimizer_config)
    else:
        optimizer = get_optimizer(model, chosen_optimizer_config)    
    print (f'using {chosen_optimizer_key}, time: {(time.time() - start_time):.2f} seconds')  
    
    model.train()
    
    sphere_center, sphere_radius, cylinder_base_center, cylinder_radius, cylinder_height, cylinder_cap_height, cylinder_cap_radius, num_points_sphere, num_points_cylinder, x_range, y_range, z_value, inlet_velocity, num_points_boundary, wind_angles = parameters()

    def total_boundary_loss(config, sphere_center, sphere_radius, cylinder_base_center, cylinder_radius, cylinder_height, cylinder_cap_height, cylinder_cap_radius, num_points_sphere, num_points_cylinder, x_range, y_range, z_value, inlet_velocity, num_points_boundary, wind_angles):
        """
        Compute the total boundary loss by summing the losses from the no-slip condition and the inlet velocity condition.

        Parameters:
        - model: The neural network model.
        - X_no_slip: The coordinates of the no-slip boundary points.
        - y_no_slip: The known zero velocities at the no-slip boundary points.
        - X_inlet: The coordinates of the inlet boundary points.
        - y_inlet: The known velocities at the inlet boundary points.

        Returns:
        - total_loss: The total boundary loss.
        """
        boundary_losses = []
        for wind_direction in wind_angles:
            X_no_slip, y_no_slip = no_slip_boundary_conditions(device, sphere_center, sphere_radius, cylinder_base_center, cylinder_radius, cylinder_height, cylinder_cap_height, cylinder_cap_radius, num_points_sphere, num_points_cylinder, wind_direction)
            X_inlet, y_inlet = sample_domain_boundary(device, x_range, y_range, z_value, wind_direction, inlet_velocity, num_points_boundary)

            if config["train_test"]["distributed_training"]:
                no_slip_loss = model.module.compute_boundary_loss(X_no_slip, y_no_slip, activation_function)
                inlet_loss = model.module.compute_boundary_loss(X_inlet, y_inlet, activation_function)
            else:
                no_slip_loss = model.compute_boundary_loss(X_no_slip, y_no_slip, activation_function)
                inlet_loss = model.compute_boundary_loss(X_inlet, y_inlet, activation_function)
            
            total_loss = no_slip_loss + inlet_loss

            boundary_losses.append([wind_direction,total_loss])

        loss_values = [loss[1] for loss in boundary_losses]
        loss_tensor = torch.tensor(loss_values)
        total_averaged_boundary_loss = loss_tensor.mean().item()
    
        return total_averaged_boundary_loss

    def calculate_total_loss(X, y, config):
        total_loss = 0
        data_loss = 0
        momentum_loss = 0
        cont_loss = 0

        n = int(config["training"]["number_of_points_per_axis"])

        if config["loss_components"]["boundary_loss"]:
            wind_directions = extract_wind_angle_from_X_closest_match(X, wind_angles, tolerance=5)
            total_averaged_boundary_loss = total_boundary_loss(config, sphere_center, sphere_radius, cylinder_base_center, cylinder_radius, cylinder_height, cylinder_cap_height, cylinder_cap_radius, num_points_sphere, num_points_cylinder, x_range, y_range, z_value, inlet_velocity, num_points_boundary, wind_directions)
        else:
            total_averaged_boundary_loss = 0
        if config["train_test"]["distributed_training"]:
            if config["loss_components"]["data_loss"]:
                data_loss_ = model.module.compute_data_loss(X, y, activation_function)
                data_loss += data_loss_
            if config["loss_components"]["momentum_loss"]:
                if config["training"]["use_custom_points_for_physics_loss"]:
                    X_custom = generate_points_from_X(X, n, device)
                else:
                    X_custom = X
                momentum_loss_ = model.module.compute_physics_momentum_loss(X_custom, activation_function)
                momentum_loss += momentum_loss_
            if config["loss_components"]["cont_loss"]:
                if config["training"]["use_custom_points_for_physics_loss"]:
                    X_custom = generate_points_from_X(X, n, device)
                else:
                    X_custom = X
                cont_loss_ = model.module.compute_physics_cont_loss(X_custom, activation_function)
                cont_loss += cont_loss_
        else:
            if config["loss_components"]["data_loss"]:
                data_loss_ = model.compute_data_loss(X, y, activation_function)
                data_loss += data_loss_
            if config["loss_components"]["momentum_loss"]:
                if config["training"]["use_custom_points_for_physics_loss"]:
                    X_custom = generate_points_from_X(X, n, device)
                else:
                    X_custom = X
                momentum_loss_ = model.compute_physics_momentum_loss(X_custom, activation_function)
                momentum_loss += momentum_loss_
            if config["loss_components"]["cont_loss"]:
                if config["training"]["use_custom_points_for_physics_loss"]:
                    X_custom = generate_points_from_X(X, n, device)
                else:
                    X_custom = X
                cont_loss_ = model.compute_physics_cont_loss(X_custom, activation_function)
                cont_loss += cont_loss_
        if config["loss_components"]["use_weighting"]:
            total_loss = weighting(data_loss, total_averaged_boundary_loss, cont_loss, momentum_loss)
        else:
            total_loss = data_loss+total_averaged_boundary_loss+cont_loss+momentum_loss
        return total_loss

    def closure():
        optimizer.zero_grad()
        if config["train_test"]["distributed_training"]:
            predictions = model.module(X_batch, activation_function)
        else:
            predictions = model(X_batch, activation_function)
        total_loss = calculate_total_loss(X_batch, y_batch, config)
        total_loss.backward()
        return total_loss
    
    if config["training"]["use_batches"]:
        if not config["training"]["force"]:
            maximum_batch_size = batch_size
            train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
            while True:
                train_loader_temp = torch.utils.data.DataLoader(train_dataset, batch_size=maximum_batch_size, shuffle=True)
                X_batch, y_batch = next(iter(train_loader_temp))  # take a single batch
                try:
                    if config[chosen_optimizer_key] == "both_optimizers":
                        optimizer = optimizer_adam
                    else:
                        continue
                    loss = optimizer.step(closure)
                    maximum_batch_size += int(0.001 * len(X_train_tensor))  # Increase and try again
                    available_gpu_memory, _ = get_available_device_memory(device.index)
                    print(f"Determining maximum batch size: {maximum_batch_size} with free memory = {available_gpu_memory:.2f} GB, time: {(time.time() - start_time):.2f} seconds")
                except StopIteration:
                    break
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        maximum_batch_size -= int(0.001 * len(X_train_tensor))  # The last successful batch size
                        break
                    raise e 
            print(f"Maximum batch size found: {maximum_batch_size}, time: {(time.time() - start_time):.2f} seconds")
            available_gpu_memory, total_gpu_memory = get_available_device_memory(device.index)
            # alpha = calculate_alpha(available_gpu_memory, total_gpu_memory)
            alpha = 0.1
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
            print (f'starting forced training, time: {(time.time() - start_time):.2f} seconds') 
            train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
            optimal_batch_size = config["training"]["batch_size"]
            if config["train_test"]["distributed_training"]:
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=optimal_batch_size, sampler=train_sampler)
            else:
                train_loader = torch.utils.data.DataLoader(train_dataset, optimal_batch_size, shuffle=True)
            required_memory_forced = estimate_memory(model, X_train_tensor, optimal_batch_size)
            print(f"Using batches with forced batch size: {optimal_batch_size} with Estimated Memory: {required_memory_forced} GB, time: {(time.time() - start_time):.2f} seconds")
    else:
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        optimal_batch_size = len(X_train_tensor)
        if config["train_test"]["distributed_training"]:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=optimal_batch_size, sampler=train_sampler)
        else:
            train_loader = torch.utils.data.DataLoader(train_dataset, optimal_batch_size, shuffle=True)
        print(f"Using full tensor with size: {optimal_batch_size}, time: {(time.time() - start_time):.2f} seconds")

    previous_loss = None
    use_epoch = config["training"]["use_epochs"]
    loss_diff_threshold = config["training"]["loss_diff_threshold"]
    consecutive_count_threshold = config["training"]["consecutive_count_threshold"]
    consecutive_count = 0
    early_stop = False  # Flag to break out of both loops

    if os.path.exists(model_file_path):
        print(f"continuing from last checkpoint..., time: {(time.time() - start_time):.2f} seconds")
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
            print("Training has already been completed., time: {(time.time() - start_time):.2f} seconds")
            return model
        else:
            start_epoch = checkpoint['epoch'] + 1
            print(f"continuing from last checkpoint... starting from epoch = {start_epoch}, time: {(time.time() - start_time):.2f} seconds")
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
                    print(f"An error occurred (Attempt #{error_count}): {e}. Trying again..., time: {(time.time() - start_time):.2f} seconds")
            break
        else:
            training_completed = False
        for X_batch, y_batch in train_loader:
            if config[chosen_optimizer_key] == "both_optimizers":
                if epoch <= config[chosen_optimizer_key]["adam_epochs"]:
                    optimizer = optimizer_adam
                else:
                    optimizer = optimizer_lbfgs
            else:
                continue
            current_loss = optimizer.step(closure)
            if previous_loss is not None:
                loss_diff = abs(current_loss - previous_loss)
                if loss_diff < loss_diff_threshold:
                    consecutive_count += 1
                    if consecutive_count >= consecutive_count_threshold:
                        print(f"Consecutive loss difference less than {loss_diff_threshold} for {consecutive_count_threshold} epochs at epoch {epoch}. Stopping training..., time: {(time.time() - start_time):.2f} seconds")
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
            print(f"An error occurred (Attempt #{error_count}): {e}. Trying again..., time: {(time.time() - start_time):.2f} seconds")

    return model