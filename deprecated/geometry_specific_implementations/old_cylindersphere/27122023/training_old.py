from definitions import *
from boundary import *
from PINN import *
from weighting import *

def train_model(rank, world_size, model, device, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, X_test_tensor_skipped, y_test_tensor_skipped, config, feature_scaler, target_scaler, batch_size, model_file_path, log_folder, meteo_filenames, datafolder_path, epochs):
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
    
    num_points_sphere, num_points_cylinder, epsilon, min_x_range, max_x_range, x_range, min_y_range, max_y_range, y_range, min_z_range, max_z_range, z_range, sphere_faces_data, cylinder_faces_data, num_points_boundary = parameters(datafolder_path, config)

    def no_slip_loss(config, wind_angles, sphere_faces_data, cylinder_faces_data, num_points_sphere, num_points_cylinder, epsilon, feature_scaler, target_scaler):
        no_slip_losses = []
        for wind_direction in wind_angles:
            sphere_no_slip_points, sphere_no_slip_normals = no_slip_boundary_conditions(device, sphere_faces_data, num_points_sphere, wind_direction, feature_scaler, target_scaler)
            cylinder_no_slip_points, cylinder_no_slip_normals = no_slip_boundary_conditions(device, cylinder_faces_data, num_points_cylinder, wind_direction, feature_scaler, target_scaler)

            if config["train_test"]["distributed_training"]:
                sphere_no_slip_loss = model.module.compute_no_slip_loss(sphere_no_slip_points, sphere_no_slip_normals, epsilon, config["training"]["output_params"])
                cylinder_no_slip_loss = model.module.compute_no_slip_loss(cylinder_no_slip_points, cylinder_no_slip_normals, epsilon, config["training"]["output_params"])
            else:
                sphere_no_slip_loss = model.compute_no_slip_loss(sphere_no_slip_points, sphere_no_slip_normals, epsilon, config["training"]["output_params"])
                cylinder_no_slip_loss = model.compute_no_slip_loss(cylinder_no_slip_points, cylinder_no_slip_normals, epsilon, config["training"]["output_params"])

            total_no_slip_loss = sphere_no_slip_loss+cylinder_no_slip_loss

            no_slip_losses.append(total_no_slip_loss)

        return no_slip_losses

    def inlet_loss(config, wind_angles, feature_scaler, target_scaler, meteo_filenames, datafolder_path, x_range, y_range, num_points_boundary):
        inlet_losses = []
        for wind_direction in wind_angles:
            boundary_points, boundary_known_velocities = sample_domain_boundary(config, device, meteo_filenames, datafolder_path, min_x_range, max_x_range, x_range, min_y_range, max_y_range, y_range, min_z_range, max_z_range, z_range, wind_direction, num_points_boundary, feature_scaler, target_scaler)

            if config["train_test"]["distributed_training"]:
                inlet_loss = model.module.compute_inlet_loss(boundary_points, boundary_known_velocities, config["training"]["output_params"])
            else:
                inlet_loss = model.compute_inlet_loss(boundary_points, boundary_known_velocities, config["training"]["output_params"])

            inlet_losses.append(inlet_loss)

        return inlet_losses

    def total_boundary_loss(config, wind_angles):
        if config["loss_components"]["no_slip_loss"]:
            no_slip_losses = no_slip_loss(config, wind_angles, sphere_faces_data, cylinder_faces_data, num_points_sphere, num_points_cylinder, epsilon, feature_scaler, target_scaler)
            avg_no_slip_loss = torch.stack(no_slip_losses).mean()
        else:
            avg_no_slip_loss = 0
        if config["loss_components"]["inlet_loss"]:
            inlet_losses = inlet_loss(config, wind_angles, feature_scaler, target_scaler, meteo_filenames, datafolder_path, x_range, y_range, num_points_boundary)
            avg_inlet_loss = torch.stack(inlet_losses).mean()
        else:
            avg_inlet_loss = 0
        total_boundary_loss = avg_no_slip_loss+avg_inlet_loss
        return total_boundary_loss, avg_no_slip_loss, avg_inlet_loss

    def calculate_total_loss(X, y, config):
        input_params = config["training"]["input_params"]
        output_params = config["training"]["output_params"]
        total_loss = 0
        n = int(config["training"]["number_of_points_per_axis"])
        if config["training"]["use_custom_points_for_physics_loss"]:
            X_custom = generate_points_from_X(X, n, device)
        else:
            X_custom = X

        rho = config["data"]["density"]
        nu = config["data"]["kinematic_viscosity"]
        all_wind_angles = config["training"]["all_angles"]

        if config["loss_components"]["boundary_loss"]:
            wind_directions = extract_wind_angle_from_X_closest_match(X, all_wind_angles, tolerance=15)
            total_averaged_boundary_loss, avg_no_slip_loss, avg_inlet_loss = total_boundary_loss(config, wind_directions)
        else:
            total_averaged_boundary_loss = 0
            avg_no_slip_loss = 0
            avg_inlet_loss = 0

        total_loss += total_averaged_boundary_loss

        if config["train_test"]["distributed_training"]:
            if config["loss_components"]["data_loss"]:
                data_loss = model.module.compute_data_loss(X, y)
                total_loss += data_loss
            if config["loss_components"]["momentum_loss"]:
                momentum_loss = model.module.compute_physics_momentum_loss(X_custom, input_params, output_params, rho, nu)
                total_loss += momentum_loss
            if config["loss_components"]["cont_loss"]:
                cont_loss = model.module.compute_physics_cont_loss(X_custom, input_params, output_params)
                total_loss += cont_loss
        else:
            if config["loss_components"]["data_loss"]:
                data_loss = model.compute_data_loss(X, y)
                total_loss += data_loss
            if config["loss_components"]["momentum_loss"]:
                momentum_loss = model.compute_physics_momentum_loss(X_custom, input_params, output_params, rho, nu)
                total_loss += momentum_loss
            if config["loss_components"]["cont_loss"]:
                cont_loss = model.compute_physics_cont_loss(X_custom, input_params, output_params)
                total_loss += cont_loss
        
        data_loss = locals().get('data_loss', 0)
        total_averaged_boundary_loss = locals().get('total_averaged_boundary_loss', 0)
        cont_loss = locals().get('cont_loss', 0)
        momentum_loss = locals().get('momentum_loss', 0)
        if config["training"]["use_epochs"]:
            max_epochs = config["training"]["num_epochs"]
        else:
            max_epochs = 1e5
        total_loss_weighted = weighting(config, data_loss, avg_inlet_loss, avg_no_slip_loss, cont_loss, momentum_loss, epoch, max_epochs, model, optimizer=optimizer)
        losses = [total_loss, total_loss_weighted, data_loss, cont_loss, momentum_loss, total_averaged_boundary_loss, avg_no_slip_loss, avg_inlet_loss]

        if config["loss_components"]["use_weighting"]:
            return total_loss_weighted, losses
        else:
            return total_loss, losses 

    def closure():
        optimizer.zero_grad()
        if config["train_test"]["distributed_training"]:
            predictions = model.module(X_batch)
        else:
            predictions = model(X_batch)
        total_loss, losses = calculate_total_loss(X_batch, y_batch, config)
        total_loss.backward()
        return total_loss, losses
    
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
                    loss, losses = optimizer.step(closure)
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
        print(f"continuing from last checkpoint at {model_file_path}..., time: {(time.time() - start_time):.2f} seconds")
        try:
            checkpoint = torch.load(model_file_path, map_location=device)
        except RuntimeError as e:
            print(f"Error loading checkpoint: {e}")
        state_dict = checkpoint['model_state_dict']
        modified_state_dict = convert_state_dict(config, state_dict)
        model.load_state_dict(modified_state_dict)
        if chosen_optimizer_key == "both_optimizers":
            if checkpoint['epoch'] <= config[chosen_optimizer_key]["adam_epochs"]:
                optimizer = optimizer_adam
            else:
                optimizer = optimizer_lbfgs
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        training_completed = checkpoint.get('training_completed', False)  # Default to False if the flag is not in the saved state
        if training_completed:
            print(f"Training has already been completed., time: {(time.time() - start_time):.2f} seconds")
            return model
        else:
            start_epoch = checkpoint['epoch']
            start_time += checkpoint['save_time']
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
                        'training_completed': training_completed,
                        'save_time': (time.time() - start_time)
                    }, model_file_path)
                    success = True  # If it reaches this point, the block executed without errors, so we set success to True
                except Exception as e:
                    error_count += 1
                    print(f"An error occurred (Attempt #{error_count}): {e}. Trying again..., time: {(time.time() - start_time):.2f} seconds")
            break
        else:
            training_completed = False
        print_lines = 0
        for X_batch, y_batch in train_loader:
            if chosen_optimizer_key == "both_optimizers":
                if epoch <= config[chosen_optimizer_key]["adam_epochs"]:
                    optimizer = optimizer_adam
                else:
                    optimizer = optimizer_lbfgs
            current_loss, current_losses = optimizer.step(closure)
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
                if print_lines < 1:
                    current_elapsed_time = time.time() - start_time
                    current_elapsed_time_hours = current_elapsed_time / 3600
                    free_memory, _ = get_available_device_memory(device.index)
                    total_loss, weighted_total_loss, data_loss, cont_loss, momentum_loss, total_avg_boundary_loss, avg_no_slip_loss, avg_inlet_loss = current_losses
                    print(f'Epoch [{epoch}/{epochs if use_epoch else "infinity"}], Loss: {current_loss}, Total Time Elapsed: {current_elapsed_time_hours:.2f} hours, with free memory: {free_memory:.2f} GB; data_loss = {data_loss}, cont_loss = {cont_loss}, momentum_loss = {momentum_loss}, total_averaged_boundary_loss = {total_avg_boundary_loss}, averaged_no_slip_loss = {avg_no_slip_loss}, averaged_inlet_loss = {avg_inlet_loss},  total_loss = {total_loss}, total_loss_weighted = {weighted_total_loss} - {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
                    save_to_csv(epoch, epochs, use_epoch, current_loss, current_elapsed_time_hours, free_memory, data_loss, cont_loss, momentum_loss, total_avg_boundary_loss, avg_no_slip_loss, avg_inlet_loss, total_loss, weighted_total_loss, file_path=os.path.join(log_folder,'info.csv'))
                    success = False
                    error_count = 0
                    while not success:
                        try:
                            torch.save({
                                'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': current_loss,
                                'training_completed': training_completed,
                                'save_time': (time.time() - start_time)
                            }, model_file_path)
                            success = True  # If it reaches this point, the block executed without errors, so we set success to True
                        except Exception as e:
                            error_count += 1
                            print(f"An error occurred (Attempt #{error_count}): {e}. Trying again...")
                    print_lines += 1
                    
                    mses, r2s = evaluate_model_training(device, model, model_file_path, X_test_tensor, y_test_tensor)
                    output_params = config["training"]["output_params_modf"]
                    mse_strings = [f'MSE {param}: {mse}' for param, mse in zip(output_params, mses)]
                    mse_output = ', '.join(mse_strings)
                    r2_strings = [f'r2 {param}: {r2}' for param, r2 in zip(output_params, r2s)]
                    r2_output = ', '.join(r2_strings)
                    print(f'Epoch [{epoch}/{epochs if use_epoch else "infinity"}], {mse_output}, {r2_output} for Test Data - {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
                    save_evaluation_to_csv(epoch, epochs, use_epoch, output_params, mses, r2s, file_path=os.path.join(log_folder,'info_test.csv'))

                    mses, r2s = evaluate_model_training(device, model, model_file_path, X_test_tensor_skipped, y_test_tensor_skipped)
                    output_params = config["training"]["output_params_modf"]
                    mse_strings = [f'MSE {param}: {mse}' for param, mse in zip(output_params, mses)]
                    mse_output = ', '.join(mse_strings)
                    r2_strings = [f'r2 {param}: {r2}' for param, r2 in zip(output_params, r2s)]
                    r2_output = ', '.join(r2_strings)
                    print(f'Epoch [{epoch}/{epochs if use_epoch else "infinity"}], {mse_output}, {r2_output} for Skipped Data - {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
                    save_evaluation_to_csv(epoch, epochs, use_epoch, output_params, mses, r2s, file_path=os.path.join(log_folder,'info_skipped.csv'))
                else:
                    pass
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
                'training_completed': training_completed,
                'save_time': (time.time() - start_time)
            }, model_file_path)
            success = True  # If it reaches this point, the block executed without errors, so we set success to True
        except Exception as e:
            error_count += 1
            print(f"An error occurred (Attempt #{error_count}): {e}. Trying again..., time: {(time.time() - start_time):.2f} seconds")

    return model