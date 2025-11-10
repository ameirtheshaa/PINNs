from definitions import *
from boundary import *
from PINN import *
from weighting import *

def no_slip_loss(config, device, model, wind_angles, flattened_mesh, epsilon, feature_scaler, target_scaler):
    no_slip_losses = []
    for wind_direction in wind_angles:
        no_slip_points, no_slip_normals = no_slip_boundary_conditions(device, flattened_mesh, wind_direction, feature_scaler, target_scaler)
        no_slip_loss = model.compute_no_slip_loss(no_slip_points, no_slip_normals, epsilon, config["training"]["output_params"])
        no_slip_losses.append(no_slip_loss)
    return torch.stack(no_slip_losses).mean()

def inlet_loss(config, device, model, wind_angles, feature_scaler, target_scaler, meteo_filenames, datafolder_path, min_x_range, max_x_range, x_range, min_y_range, max_y_range, y_range, min_z_range, max_z_range, z_range, num_points_boundary):
    inlet_losses = []
    for wind_direction in wind_angles:
        boundary_points, boundary_known_velocities = sample_domain_boundary(config, device, meteo_filenames, datafolder_path, min_x_range, max_x_range, x_range, min_y_range, max_y_range, y_range, min_z_range, max_z_range, z_range, wind_direction, num_points_boundary, feature_scaler, target_scaler)
        inlet_loss = model.compute_inlet_loss(boundary_points, boundary_known_velocities, config["training"]["output_params"])
        inlet_losses.append(inlet_loss)
    return torch.stack(inlet_losses).mean()

def calculate_total_loss(X, y, X_data, y_data, epoch, config, device, model, optimizer, datafolder_path, meteo_filenames, feature_scaler, target_scaler):
    geometry_filename = os.path.join(datafolder_path,'scaled_cylinder_sphere.stl')
    epsilon, min_x_range, max_x_range, x_range, min_y_range, max_y_range, y_range, min_z_range, max_z_range, z_range, flattened_mesh, num_points_boundary = parameters(geometry_filename)
    input_params = config["training"]["input_params"]
    output_params = config["training"]["output_params"]
    rho = config["data"]["density"]
    nu = config["data"]["kinematic_viscosity"]
    all_wind_angles = config["training"]["all_angles"]
    training_wind_angles = config["training"]["angles_to_train"]
    total_loss = 0

    if config["loss_components"]["data_loss"]:
        data_loss = model.compute_data_loss(X_data, y_data)
    else:
        data_loss = 0

    if config["loss_components"]["cont_loss"]:
        cont_loss = model.compute_physics_cont_loss(X, input_params, output_params)
    else:
        cont_loss = 0

    if config["loss_components"]["no_slip_loss"]:
        no_slip_losses = no_slip_loss(config, device, model, training_wind_angles, flattened_mesh, epsilon, feature_scaler, target_scaler)
    else:
        no_slip_losses = 0

    if config["loss_components"]["inlet_loss"]:
        inlet_losses = inlet_loss(config, device, model, training_wind_angles, feature_scaler, target_scaler, meteo_filenames, datafolder_path, min_x_range, max_x_range, x_range, min_y_range, max_y_range, y_range, min_z_range, max_z_range, z_range, num_points_boundary)
    else:
        inlet_losses = 0

    if config["loss_components"]["momentum_loss"]:
        momentum_loss = model.compute_physics_momentum_loss(X, input_params, output_params, rho, nu)
    else:
        momentum_loss = 0

    total_loss += data_loss
    total_loss += cont_loss
    total_loss += no_slip_losses
    total_loss += inlet_losses
    total_loss += momentum_loss

    if config["loss_components"]["use_weighting"]:
        if config["training"]["use_epochs"]:
            max_epochs = config["training"]["num_epochs"]
        else:
            max_epochs = 1e5
        total_loss_weighted = weighting(config, data_loss, inlet_losses, no_slip_losses, cont_loss, momentum_loss, epoch, max_epochs, model, optimizer)
    else:
        total_loss_weighted = 0
    
    losses = [total_loss, total_loss_weighted, data_loss, cont_loss, momentum_loss, no_slip_losses, inlet_losses]

    if config["loss_components"]["use_weighting"]:
        return total_loss_weighted, losses
    else:
        return total_loss, losses

def save_model(model, model_file_path, epoch, current_loss, training_completed, optimizer):
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

def save_evaluation_results(config, device, model, model_file_path, log_folder, X_test_tensor, y_test_tensor, epoch, epochs, use_epoch, save_name):
    mses, r2s = evaluate_model_training(device, model, model_file_path, X_test_tensor, y_test_tensor)
    output_params = config["training"]["output_params_modf"]
    mse_strings = [f'MSE {param}: {mse}' for param, mse in zip(output_params, mses)]
    mse_output = ', '.join(mse_strings)
    r2_strings = [f'r2 {param}: {r2}' for param, r2 in zip(output_params, r2s)]
    r2_output = ', '.join(r2_strings)
    print(f'Epoch [{epoch}/{epochs if use_epoch else "infinity"}], {mse_output}, {r2_output} for {save_name} Data - {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    save_evaluation_to_csv(epoch, epochs, use_epoch, output_params, mses, r2s, file_path=os.path.join(log_folder,f'{save_name}'))