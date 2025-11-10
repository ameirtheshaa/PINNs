from definitions import *
from physics import *
from PINN import *

def extract_parameters(X, input_params):
    extracted_params = []
    for param in input_params:
        if param == 'Points:0':
            extracted_params.append(X[:, 0:1])
        elif param == 'Points:1':
            extracted_params.append(X[:, 1:2])
        elif param == 'Points:2':
            extracted_params.append(X[:, 2:3])
        elif param == 'cos(WindAngle)':
            extracted_params.append(X[:, 3:4])
        elif param == 'sin(WindAngle)':
            extracted_params.append(X[:, 4:5])
        # Add more conditions here for other potential parameters
        else:
            raise ValueError(f"Unknown parameter: {param}")
    return extracted_params

def compute_data_loss(model, X, y):
    criterion = nn.MSELoss()
    predictions = model(X)
    loss = criterion(predictions, y)
    return loss

def compute_inlet_loss(model, X_boundary, y_boundary_known, output_params):

    def find_velocity_indices(output_params):
        velocity_labels = ['Velocity:0', 'Velocity:1', 'Velocity:2']
        velocity_indices = [output_params.index(label) for label in velocity_labels]
        return velocity_indices

    velocity_indices = find_velocity_indices(output_params)
    start_idx, end_idx = min(velocity_indices), max(velocity_indices) + 1
    boundary_predictions = model(X_boundary)
    velocities = boundary_predictions[:, start_idx:end_idx]
    criterion = nn.MSELoss()
    inlet_loss = criterion(velocities, y_boundary_known)
    return inlet_loss

def compute_no_slip_loss(model, X_boundary, normals, epsilon, output_params):

    def find_velocity_indices(output_params):
        velocity_labels = ['Velocity:0', 'Velocity:1', 'Velocity:2']
        velocity_indices = [output_params.index(label) for label in velocity_labels]
        return velocity_indices

    def compute_tangential_velocity(velocities, normals):
        normal_velocities = torch.sum(velocities * normals, dim=1, keepdim=True) * normals
        tangential_velocities = velocities - normal_velocities
        return tangential_velocities, normal_velocities

    def compute_penalty_term(tangential_velocities, epsilon):
        penalty = torch.mean(tangential_velocities**2)  # (∫(v⋅τ)² ds) term
        penalty = penalty / (2*epsilon)
        return penalty

    velocity_indices = find_velocity_indices(output_params)
    start_idx, end_idx = min(velocity_indices), max(velocity_indices) + 1
    boundary_predictions = model(X_boundary)
    velocities = boundary_predictions[:, start_idx:end_idx]
    tangential_velocities, normal_velocities = compute_tangential_velocity(velocities, normals)
    penalty = compute_penalty_term(tangential_velocities, epsilon)
    criterion = nn.MSELoss()
    normal_loss = criterion(normal_velocities, torch.zeros_like(normal_velocities))
    no_slip_loss = penalty + normal_loss
    return no_slip_loss

def compute_physics_momentum_loss(model, X, input_params, output_params, rho, nu):
    extracted_inputs = extract_input_parameters(X, input_params)
    input_dict = {param: value for param, value in zip(input_params, extracted_inputs)}
    x = input_dict.get('Points:0')
    y = input_dict.get('Points:1')
    z = input_dict.get('Points:2')
    cos_wind_angle = input_dict.get('cos(WindAngle)')
    sin_wind_angle = input_dict.get('sin(WindAngle)')
    f, g, h = RANS(model, cos_wind_angle, sin_wind_angle, x, y, z, rho, nu, output_params)
    loss_f = nn.MSELoss()(f, torch.zeros_like(f))
    loss_g = nn.MSELoss()(g, torch.zeros_like(g))
    loss_h = nn.MSELoss()(h, torch.zeros_like(h))
    loss_physics_momentum = loss_f + loss_g + loss_h 
    return loss_physics_momentum

def compute_physics_cont_loss(model, X, input_params, output_params):
    extracted_inputs = extract_parameters(X, input_params)
    input_dict = {param: value for param, value in zip(input_params, extracted_inputs)}
    x = input_dict.get('Points:0')
    y = input_dict.get('Points:1')
    z = input_dict.get('Points:2')
    cos_wind_angle = input_dict.get('cos(WindAngle)')
    sin_wind_angle = input_dict.get('sin(WindAngle)')
    cont = continuity(model, cos_wind_angle, sin_wind_angle, x, y, z, output_params)
    loss_cont = nn.MSELoss()(cont, torch.zeros_like(cont))
    return loss_cont