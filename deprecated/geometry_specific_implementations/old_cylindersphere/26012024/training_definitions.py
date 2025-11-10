from definitions import *
from boundary import *
from PINN import *
from weighting import *

def no_slip_loss(model, no_slip_features_normals, output_params):
    no_slip_losses = []
    for i in no_slip_features_normals:
        epsilon = i[0]
        no_slip_points = i[2]
        no_slip_normals = i[3]
        no_slip_loss = model.compute_no_slip_loss(no_slip_points, no_slip_normals, epsilon, output_params)
        no_slip_losses.append(no_slip_loss)
    return torch.stack(no_slip_losses).sum()

def inlet_loss(model, all_inlet_features_targets, output_params):
    inlet_losses = []
    for i in all_inlet_features_targets:
        boundary_points = i[1]
        boundary_known_velocities = i[2]
        inlet_loss = model.compute_inlet_loss(boundary_points, boundary_known_velocities, output_params)
        inlet_losses.append(inlet_loss)
    return torch.stack(inlet_losses).sum()

def calculate_total_loss(X, y, epoch, config, model, optimizer, closure, no_slip_features_normals=None, all_inlet_features_targets=None, y_divergences=None):
    
    input_params = config["training"]["input_params"]
    output_params = config["training"]["output_params"]
    rho = config["data"]["density"]
    nu = config["data"]["kinematic_viscosity"]
    all_wind_angles = config["training"]["all_angles"]
    training_wind_angles = config["training"]["angles_to_train"]
    loss_components = {}
    if config["loss_components"]["data_loss"]:
        if config["train_test"]["data_loss_test_size"] is None:
            loss_components['data_loss'] = model.compute_data_loss(X, y)
        else:
            n = config["train_test"]["data_loss_test_size"]
            if n > 1:
                n = min(n, len(X))
            else:
                n *= len(X)
            n = int(n)
            indices = torch.randperm(len(X))[:n]
            X_subset = X[indices]
            y_subset = y[indices]
            model.compute_data_loss(X_subset, y_subset)
    if config["loss_components"]["cont_loss"]:
        if y_divergences is not None:
            y_divergences = y_divergences.view(-1, 1)
            loss_components['cont_loss'] = model.compute_physics_cont_loss(X, input_params, output_params, y_divergences)
        else:
            loss_components['cont_loss'] = model.compute_physics_cont_loss(X, input_params, output_params)
    if config["loss_components"]["no_slip_loss"]:
        loss_components['no_slip_loss'] = no_slip_loss(model, no_slip_features_normals, output_params)
    if config["loss_components"]["inlet_loss"]:
        loss_components['inlet_loss'] = inlet_loss(model, all_inlet_features_targets, output_params)
    if config["loss_components"]["momentum_loss"]:
        loss_components['momentum_loss'] = model.compute_physics_momentum_loss(X, input_params, output_params, rho, nu)
    total_loss = sum(loss_components.values())
    loss_components['total_loss'] = total_loss
    loss_components['total_loss_weighted'] = total_loss
    if config["loss_components"]["use_weighting"]:
        if config["training"]["use_epochs"]:
            max_epochs = config["training"]["num_epochs"]
        else:
            max_epochs = 1e5
        total_loss_weighted = weighting(config, loss_components, epoch, max_epochs, model, optimizer)
        loss_components['total_loss_weighted'] = total_loss_weighted
        if closure:
            return total_loss_weighted
        else:
            return loss_components
    else:
        if closure:
            return total_loss
        else:
            return loss_components

def save_model(model, model_file_path, epoch, current_loss, training_completed, optimizer, start_time):
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
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': current_loss,
                'training_completed': training_completed
            }, f'{model_file_path}_{epoch}')
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