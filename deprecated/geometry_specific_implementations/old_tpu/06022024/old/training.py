from definitions import *
from boundary import *
# from PINN import *
from weighting import *

def train_model(model, device, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, X_test_tensor_skipped, y_test_tensor_skipped, config, feature_scaler, target_scaler, batch_size, model_file_path, log_folder, meteo_filenames, datafolder_path, epochs):
    # Capture start time
    start_time = time.time()
    print (f'starting to train, time: {(time.time() - start_time):.2f} seconds') 
    # Define the path for the JSON file (change 'metadata.json' as needed)
    metadata_file_path = model_file_path + "_metadata.json"
    model_file_path = Path(model_file_path)  # Convert the string to a Path object

    # Set a default value for start_epoch
    start_epoch = 1
    training_completed = False
   
    # print (f'using {chosen_optimizer_key}, time: {(time.time() - start_time):.2f} seconds')  
    
    num_points_sphere, num_points_cylinder, epsilon, x_range, y_range, sphere_faces_data, cylinder_faces_data, num_points_boundary = parameters(datafolder_path, config)

    current_losses = []
    
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
            boundary_points, boundary_known_velocities = sample_domain_boundary(config, device, meteo_filenames, datafolder_path, x_range, y_range, wind_direction, num_points_boundary, feature_scaler, target_scaler)

            if config["train_test"]["distributed_training"]:
                inlet_loss = model.module.compute_boundary_loss(boundary_points, boundary_known_velocities, config["training"]["output_params"])
            else:
                inlet_loss = model.compute_boundary_loss(boundary_points, boundary_known_velocities, config["training"]["output_params"])

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
        total_loss_weighted = weighting(data_loss, total_averaged_boundary_loss, cont_loss, momentum_loss)
        losses = [total_loss, total_loss_weighted, data_loss, cont_loss, momentum_loss, total_averaged_boundary_loss, avg_no_slip_loss, avg_inlet_loss]
        if config["loss_components"]["use_weighting"]:
            return total_loss_weighted, losses
        else:
            return total_loss, losses 

    def closure(model, X_batch, y_batch):
        with tf.GradientTape() as tape:
            predictions = model(X_batch, training=True)
            total_loss, losses = calculate_total_loss(X_batch, y_batch, config)
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        current_losses = losses
        return total_loss

    # Assuming X_train_tensor and y_train_tensor are your training data and labels in TensorFlow format
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_tensor, y_train_tensor))

    if config["training"]["use_batches"]:
        optimal_batch_size = config["training"]["batch_size"]
    else:
        power, counter = nearest_power_of_2(len(X_train_tensor))
        optimal_batch_size = power

    train_dataset = train_dataset.batch(optimal_batch_size).shuffle(buffer_size=len(X_train_tensor))

    if config['TPU']:
        # Create strategy for TPU
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.experimental.TPUStrategy(resolver)

        # Distribute the dataset
        train_dataset = strategy.experimental_distribute_dataset(train_dataset)

    previous_loss = None
    use_epoch = config["training"]["use_epochs"]
    loss_diff_threshold = config["training"]["loss_diff_threshold"]
    consecutive_count_threshold = config["training"]["consecutive_count_threshold"]
    consecutive_count = 0
    early_stop = False  # Flag to break out of both loops

    # Load model
    if os.path.exists(model_file_path):
        model = tf.keras.models.load_model(model_file_path)
        print(f"Model loaded from {model_file_path}")

        # Load training metadata
        if os.path.exists(metadata_file_path):
            with open(metadata_file_path, 'r') as f:
                checkpoint = json.load(f)
            start_epoch = checkpoint.get('epoch', 1)
            training_completed = checkpoint.get('training_completed', False)

            if training_completed:
                print(f"Training has already been completed. Last trained epoch: {start_epoch}")
                return model
            else:
                print(f"Continuing from last checkpoint... starting from epoch = {start_epoch}")

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
            success = False
            error_count = 0
            while not success:
                try:
                    # Save the TensorFlow model
                    model.save(model_file_path)

                    # Prepare training metadata for saving
                    training_metadata = {
                        'epoch': epoch,
                        'loss': float(current_loss),  # Convert to float if it's a tensor
                        'training_completed': training_completed
                    }

                    # Save metadata as a JSON file
                    with open(metadata_file_path, 'w') as f:
                        json.dump(training_metadata, f)
                    success = True  # If it reaches this point, the block executed without errors, so we set success to True
                except Exception as e:
                    error_count += 1
                    print(f"An error occurred (Attempt #{error_count}): {e}. Trying again..., time: {(time.time() - start_time):.2f} seconds")
            break
        else:
            training_completed = False
        print_lines = 0
        for X_batch, y_batch in train_dataset:
            if config['TPU']:
                current_loss = xm.optimizer_step(optimizer, barrier=True, optimizer_args={'closure': closure})
                xm.mark_step()
            else:
                current_loss = closure(model, X_batch, y_batch)
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
                    if len(current_losses) > 0:
                        total_loss, weighted_total_loss, data_loss, cont_loss, momentum_loss, total_avg_boundary_loss, avg_no_slip_loss, avg_inlet_loss = current_losses
                        print(f'Epoch [{epoch}/{epochs if use_epoch else "infinity"}], Loss: {current_loss}, Total Time Elapsed: {current_elapsed_time_hours:.2f} hours, with free memory: {free_memory:.2f} GB; data_loss = {data_loss}, cont_loss = {cont_loss}, momentum_loss = {momentum_loss}, total_averaged_boundary_loss = {total_avg_boundary_loss}, averaged_no_slip_loss = {avg_no_slip_loss}, averaged_inlet_loss = {avg_inlet_loss},  total_loss = {total_loss}, total_loss_weighted = {weighted_total_loss}')
                        save_to_csv(epoch, epochs, use_epoch, current_loss, current_elapsed_time_hours, free_memory, data_loss, cont_loss, momentum_loss, total_avg_boundary_loss, avg_no_slip_loss, avg_inlet_loss, total_loss, weighted_total_loss, file_path=os.path.join(log_folder,'info.csv'))
                    success = False
                    error_count = 0
                    while not success:
                        try:
                            # Save the TensorFlow model
                            model.save(model_file_path)

                            # Prepare training metadata for saving
                            training_metadata = {
                                'epoch': epoch,
                                'loss': float(current_loss),  # Convert to float if it's a tensor
                                'training_completed': training_completed
                            }

                            # Define the path for the JSON file (change 'metadata.json' as needed)
                            metadata_file_path = model_file_path + "_metadata.json"

                            # Save metadata as a JSON file
                            with open(metadata_file_path, 'w') as f:
                                json.dump(training_metadata, f)
                            success = True  # Set success to True if saving is successful
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
                    print(f'Epoch [{epoch}/{epochs if use_epoch else "infinity"}], {mse_output}, {r2_output} for Test Data')
                    save_evaluation_to_csv(epoch, epochs, use_epoch, output_params, mses, r2s, file_path=os.path.join(log_folder,'info_test.csv'))

                    mses, r2s = evaluate_model_training(device, model, model_file_path, X_test_tensor_skipped, y_test_tensor_skipped)
                    output_params = config["training"]["output_params_modf"]
                    mse_strings = [f'MSE {param}: {mse}' for param, mse in zip(output_params, mses)]
                    mse_output = ', '.join(mse_strings)
                    r2_strings = [f'r2 {param}: {r2}' for param, r2 in zip(output_params, r2s)]
                    r2_output = ', '.join(r2_strings)
                    print(f'Epoch [{epoch}/{epochs if use_epoch else "infinity"}], {mse_output}, {r2_output} for Skipped Data')
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
                'training_completed': training_completed
            }, model_file_path)
            success = True  # If it reaches this point, the block executed without errors, so we set success to True
        except Exception as e:
            error_count += 1
            print(f"An error occurred (Attempt #{error_count}): {e}. Trying again..., time: {(time.time() - start_time):.2f} seconds")

    return model