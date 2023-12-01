from definitions import *
from training import *
from plotting import *

class PINN(nn.Module):
    def __init__(self, input_params, output_params, hidden_layers, neurons_per_layer, activation, use_batch_norm, dropout_rate):
        super(PINN, self).__init__()

        input_size = len(input_params)
        output_size = len(output_params)

        # Create a list to hold all layers
        layers = []

        # Input layer
        layers.append(nn.Linear(input_size, neurons_per_layer[0]))

        # Hidden layers
        for i in range(hidden_layers - 1):
            layers.append(activation())
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(neurons_per_layer[i]))
            if dropout_rate:
                layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.Linear(neurons_per_layer[i], neurons_per_layer[i + 1]))

        # Output layer
        layers.append(activation())
        layers.append(nn.Linear(neurons_per_layer[-1], output_size))

        # Combine all layers into a Sequential module
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def compute_data_loss(self, X, y):
        criterion = nn.MSELoss()
        predictions = self(X)
        loss = criterion(predictions, y)
        return loss

    def compute_boundary_loss(self, X_boundary, y_boundary_known):
        criterion = nn.MSELoss()
        boundary_predictions = self(X_boundary)
        
        # Only consider the velocity components (u, v, w) for the boundary loss
        boundary_loss = criterion(boundary_predictions[:, 1:4], y_boundary_known)
        
        return boundary_loss

    def compute_no_slip_loss(self, X_boundary, normals, output_params):

        def find_velocity_indices(output_params):
            velocity_labels = ['Velocity:0', 'Velocity:1', 'Velocity:2']
            velocity_indices = [output_params.index(label) for label in velocity_labels]
            return velocity_indices

        def compute_tangential_velocity(velocities, normals):
            normal_velocities = torch.sum(velocities * normals, dim=1, keepdim=True) * normals
            tangential_velocities = velocities - normal_component
            return tangential_velocities, normal_velocities

        def compute_penalty_term(tangential_velocities, epsilon):
            penalty = torch.mean(tangential_velocities**2)  # (∫(v⋅τ)² ds) term
            penalty = penalty / (2*epsilon)
            return penalty

        velocity_indices = find_velocity_indices(output_params)
        start_idx, end_idx = min(velocity_indices), max(velocity_indices) + 1

        boundary_predictions = self(X_boundary)
        velocities = boundary_predictions[:, start_idx:end_idx]
        
        tangential_velocities, normal_velocities = compute_tangential_velocity(velocities, normals)
        penalty = compute_penalty_term(tangential_velocities, epsilon)

        criterion = nn.MSELoss()
        normal_loss = criterion(normal_velocities, torch.zeros_like(normal_velocities))
        
        no_slip_loss = penalty + normal_loss

        return no_slip_loss
    
    def extract_parameters(self, X, input_params):
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

    def extract_output_parameters(self, Y, output_params):
        extracted_output_params = []
        for param in output_params:
            if param == 'Pressure':
                extracted_output_params.append(Y[:, 0:1])
            elif param == 'Velocity:0':
                extracted_output_params.append(Y[:, 1:2])
            elif param == 'Velocity:1':
                extracted_output_params.append(Y[:, 2:3])
            elif param == 'Velocity:2':
                extracted_output_params.append(Y[:, 3:4])
            elif param == 'TurbVisc':
                extracted_output_params.append(Y[:, 4:5])
            # Add more conditions here for other potential output parameters
            else:
                raise ValueError(f"Unknown output parameter: {param}")

        return extracted_output_params

    def continuity(self, cos_wind_angle, sin_wind_angle, x, y, z, output_params):
        # Ensure x, y, and z have requires_grad set to True
        x.requires_grad_(True)
        y.requires_grad_(True)
        z.requires_grad_(True)

        # Stack the input data
        input_data = torch.hstack((x, y, z, cos_wind_angle, sin_wind_angle))

        output = self(input_data)  # Model output
        extracted_outputs = self.extract_output_parameters(output, output_params)

        # Create a dictionary to hold the extracted outputs
        output_dict = {param: value for param, value in zip(output_params, extracted_outputs)}

        u = output_dict.get('Velocity:0')
        v = output_dict.get('Velocity:1')
        w = output_dict.get('Velocity:2')

        def compute_gradients_and_second_order_gradients(tensor, coord):
            # Compute the gradients and second order gradients of tensor with respect to coord
            grad_first = torch.autograd.grad(tensor, coord, grad_outputs=torch.ones_like(tensor), create_graph=True)[0]
            grad_second = torch.autograd.grad(grad_first, coord, grad_outputs=torch.ones_like(grad_first), create_graph=True)[0]
            return grad_first, grad_second

        # Compute gradients and second order gradients
        u_x, _ = compute_gradients_and_second_order_gradients(u, x)
        v_y, _ = compute_gradients_and_second_order_gradients(v, y)
        w_z, _ = compute_gradients_and_second_order_gradients(w, z)

        cont = u_x + v_y + w_z

        return cont

    def RANS(self, cos_wind_angle, sin_wind_angle, x, y, z, rho, nu, output_params):
        # Ensure x, y, and z have requires_grad set to True
        x.requires_grad_(True)
        y.requires_grad_(True)
        z.requires_grad_(True)

        # Stack the input data
        input_data = torch.hstack((x, y, z, cos_wind_angle, sin_wind_angle))

        output = self(input_data)  # Model output
        extracted_outputs = self.extract_output_parameters(output, output_params)

        # Create a dictionary to hold the extracted outputs
        output_dict = {param: value for param, value in zip(output_params, extracted_outputs)}

        # Access the outputs as needed
        p = output_dict.get('Pressure')
        u = output_dict.get('Velocity:0')
        v = output_dict.get('Velocity:1')
        w = output_dict.get('Velocity:2')
        nu_t = output_dict.get('TurbVisc')

        nu_eff = nu_t + nu

        def compute_gradients_and_second_order_gradients(tensor, coord):
            # Compute the gradients and second order gradients of tensor with respect to coord
            grad_first = torch.autograd.grad(tensor, coord, grad_outputs=torch.ones_like(tensor), create_graph=True)[0]
            grad_second = torch.autograd.grad(grad_first, coord, grad_outputs=torch.ones_like(grad_first), create_graph=True)[0]
            return grad_first, grad_second

        # Compute gradients and second order gradients
        u_x, u_xx = compute_gradients_and_second_order_gradients(u, x)
        u_y, u_yy = compute_gradients_and_second_order_gradients(u, y)
        u_z, u_zz = compute_gradients_and_second_order_gradients(u, z)

        v_x, v_xx = compute_gradients_and_second_order_gradients(v, x)
        v_y, v_yy = compute_gradients_and_second_order_gradients(v, y)
        v_z, v_zz = compute_gradients_and_second_order_gradients(v, z)

        w_x, w_xx = compute_gradients_and_second_order_gradients(w, x)
        w_y, w_yy = compute_gradients_and_second_order_gradients(w, y)
        w_z, w_zz = compute_gradients_and_second_order_gradients(w, z)

        # Compute the gradients of p
        p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        p_z = torch.autograd.grad(p, z, grad_outputs=torch.ones_like(p), create_graph=True)[0]

        f = (u * u_x + v * u_y + w * u_z - (1 / rho) * p_x + nu_eff * (2 * u_xx) + nu_eff * (u_yy + v_xy) + nu_eff * (u_zz + w_xz))
        g = (u * v_x + v * v_y + w * v_z - (1 / rho) * p_y + nu_eff * (v_xx + u_xy) + nu_eff * (2 * v_yy) + nu_eff * (v_zz + w_yz))
        h = (u * w_x + v * w_y + w * w_z - (1 / rho) * p_z + nu_eff * (w_xx + u_xz) + nu_eff * (w_yy + v_yz) + nu_eff * (2 * w_zz))

        return f, g, h

    def compute_physics_momentum_loss(self, X, input_params, output_params):
        extracted_inputs = self.extract_parameters(X, input_params)
        input_dict = {param: value for param, value in zip(input_params, extracted_inputs)}

        # Access the inputs using keys
        x = input_dict.get('Points:0')
        y = input_dict.get('Points:1')
        z = input_dict.get('Points:2')
        cos_wind_angle = input_dict.get('cos(WindAngle)')
        sin_wind_angle = input_dict.get('sin(WindAngle)')

        f, g, h = self.RANS(cos_wind_angle, sin_wind_angle, x, y, z, rho, nu)
        
        loss_f = nn.MSELoss()(f, torch.zeros_like(f))
        loss_g = nn.MSELoss()(g, torch.zeros_like(g))
        loss_h = nn.MSELoss()(h, torch.zeros_like(h))
        
        loss_physics_momentum = loss_f + loss_g + loss_h 
        
        return loss_physics_momentum

    def compute_physics_cont_loss(self, X, input_params, output_params):
        extracted_inputs = self.extract_parameters(X, input_params)
        input_dict = {param: value for param, value in zip(input_params, extracted_inputs)}

        # Access the inputs using keys
        x = input_dict.get('Points:0')
        y = input_dict.get('Points:1')
        z = input_dict.get('Points:2')
        cos_wind_angle = input_dict.get('cos(WindAngle)')
        sin_wind_angle = input_dict.get('sin(WindAngle)')

        cont = self.continuity(cos_wind_angle, sin_wind_angle, x, y, z, output_params)
        
        loss_cont = nn.MSELoss()(cont, torch.zeros_like(cont))

        return loss_cont