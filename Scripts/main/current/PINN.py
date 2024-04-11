import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, input_params, output_params, hidden_layers, neurons_per_layer, activation, use_batch_norm, dropout_rate):
        super(PINN, self).__init__()
        input_size = len(input_params)
        output_size = len(output_params)
        layers = []
        layers.append(nn.Linear(input_size, neurons_per_layer[0]))
        for i in range(hidden_layers - 1):
            layers.append(activation())
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(neurons_per_layer[i]))
            if dropout_rate:
                layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.Linear(neurons_per_layer[i], neurons_per_layer[i + 1]))
        layers.append(activation())
        layers.append(nn.Linear(neurons_per_layer[-1], output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def compute_data_loss(self, X, y):
        criterion = nn.MSELoss()
        predictions = self(X)
        loss = criterion(predictions, y)
        return loss

    def compute_inlet_loss(self, X_boundary, y_boundary_known, output_params):

        def find_velocity_indices(output_params):
            velocity_labels = ['Velocity:0', 'Velocity:1', 'Velocity:2']
            velocity_indices = [output_params.index(label) for label in velocity_labels]
            return velocity_indices

        velocity_indices = find_velocity_indices(output_params)
        start_idx, end_idx = min(velocity_indices), max(velocity_indices) + 1
        boundary_predictions = self(X_boundary)
        velocities = boundary_predictions[:, start_idx:end_idx]
        criterion = nn.MSELoss()
        inlet_loss = criterion(velocities, y_boundary_known)
        return inlet_loss

    def compute_no_slip_loss(self, X_boundary, normals, epsilon, output_params):

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
        boundary_predictions = self(X_boundary)
        velocities = boundary_predictions[:, start_idx:end_idx]
        tangential_velocities, normal_velocities = compute_tangential_velocity(velocities, normals)
        penalty = compute_penalty_term(tangential_velocities, epsilon)
        criterion = nn.MSELoss()
        normal_loss = criterion(normal_velocities, torch.zeros_like(normal_velocities))
        no_slip_loss = penalty + normal_loss
        return no_slip_loss
    
    def extract_input_parameters(self, Y, input_params):
        # Create a dictionary mapping parameter names to their column indices
        param_to_index = {param: i for i, param in enumerate(input_params)}

        extracted_input_params = []
        for param in input_params:
            if param in param_to_index:
                idx = param_to_index[param]
                extracted_input_params.append(Y[:, idx:idx+1])
            else:
                raise ValueError(f"Unknown input parameter: {param}")
        
        return extracted_input_params

    def extract_output_parameters(self, Y, output_params):
        # Create a dictionary mapping parameter names to their column indices
        param_to_index = {param: i for i, param in enumerate(output_params)}

        extracted_output_params = []
        for param in output_params:
            if param in param_to_index:
                idx = param_to_index[param]
                extracted_output_params.append(Y[:, idx:idx+1])
            else:
                raise ValueError(f"Unknown output parameter: {param}")
        
        return extracted_output_params

    def continuity(self, cos_wind_angle, sin_wind_angle, x, y, z, output_params):
        x.requires_grad_(True)
        y.requires_grad_(True)
        z.requires_grad_(True)
        input_data = torch.hstack((x, y, z, cos_wind_angle, sin_wind_angle))
        output = self(input_data)  # Model output
        extracted_outputs = self.extract_output_parameters(output, output_params)
        output_dict = {param: value for param, value in zip(output_params, extracted_outputs)}
        u = output_dict.get('Velocity:0')
        v = output_dict.get('Velocity:1')
        w = output_dict.get('Velocity:2')

        def compute_gradients(tensor, coord):
            grad_first = torch.autograd.grad(tensor, coord, grad_outputs=torch.ones_like(tensor), create_graph=True)[0]
            return grad_first

        u_x = compute_gradients(u, x)
        v_y = compute_gradients(v, y)
        w_z = compute_gradients(w, z)
        cont = u_x + v_y + w_z
        return cont

    def compute_derivatives(self, cos_wind_angle, sin_wind_angle, x, y, z, output_params):
        x.requires_grad_(True)
        y.requires_grad_(True)
        z.requires_grad_(True)
        input_data = torch.hstack((x, y, z, cos_wind_angle, sin_wind_angle))
        output = self(input_data)
        extracted_outputs = self.extract_output_parameters(output, output_params)
        output_dict = {param: value for param, value in zip(output_params, extracted_outputs)}
        p = output_dict.get('Pressure')
        u = output_dict.get('Velocity:0')
        v = output_dict.get('Velocity:1')
        w = output_dict.get('Velocity:2')
        nu_t = output_dict.get('TurbVisc')

        def compute_gradients_and_second_order_gradients(tensor, coord1, coord2):
            grad_first = torch.autograd.grad(tensor, coord1, grad_outputs=torch.ones_like(tensor), create_graph=True)[0]
            grad_second = torch.autograd.grad(grad_first, coord2, grad_outputs=torch.ones_like(grad_first), create_graph=True)[0]
            grad_first = grad_first.cpu().detach().numpy().flatten()
            grad_second = grad_second.cpu().detach().numpy().flatten()
            return grad_first, grad_second

        def compute_gradients(tensor, coord):
            grad_first = torch.autograd.grad(tensor, coord, grad_outputs=torch.ones_like(tensor), create_graph=True)[0]
            grad_first = grad_first.cpu().detach().numpy().flatten()
            return grad_first

        u_x, u_xx = compute_gradients_and_second_order_gradients(u, x, x)
        u_y, u_yy = compute_gradients_and_second_order_gradients(u, y, y)
        u_z, u_zz = compute_gradients_and_second_order_gradients(u, z, z)
        v_x, v_xx = compute_gradients_and_second_order_gradients(v, x, x)
        v_y, v_yy = compute_gradients_and_second_order_gradients(v, y, y)
        v_z, v_zz = compute_gradients_and_second_order_gradients(v, z, z)
        w_x, w_xx = compute_gradients_and_second_order_gradients(w, x, x)
        w_y, w_yy = compute_gradients_and_second_order_gradients(w, y, y)
        w_z, w_zz = compute_gradients_and_second_order_gradients(w, z, z)
        
        _, u_xy = compute_gradients_and_second_order_gradients(u, x, y)
        _, u_xz = compute_gradients_and_second_order_gradients(u, x, z)
        _, v_xy = compute_gradients_and_second_order_gradients(v, x, y)
        _, v_yz = compute_gradients_and_second_order_gradients(v, y, z)
        _, w_xz = compute_gradients_and_second_order_gradients(w, x, z)
        _, w_yz = compute_gradients_and_second_order_gradients(w, y, z)

        p_x = compute_gradients(p, x)
        p_y = compute_gradients(p, y)
        p_z = compute_gradients(p, z)

        derivatives_dict = {
            "u": u.cpu().detach().numpy().flatten(),
            "v": v.cpu().detach().numpy().flatten(),
            "w": w.cpu().detach().numpy().flatten(),
            "p": p.cpu().detach().numpy().flatten(),
            "TurbVisc": nu_t.cpu().detach().numpy().flatten(),
            "u_x": u_x,
            "u_xx": u_xx,
            "u_y": u_y,
            "u_yy": u_yy,
            "u_z": u_z,
            "u_zz": u_zz,
            "u_xy": u_xy,
            "u_xz": u_xz,
            "v_x": v_x,
            "v_xx": v_xx,
            "v_y": v_y,
            "v_yy": v_yy,
            "v_z": v_z,
            "v_zz": v_zz,
            "v_xy": v_xy,
            "v_yz": v_yz,
            "w_x": w_x,
            "w_xx": w_xx,
            "w_y": w_y,
            "w_yy": w_yy,
            "w_z": w_z,
            "w_zz": w_zz,
            "w_xz": w_xz,
            "w_yz": w_yz,
            "p_x": p_x,
            "p_y": p_y,
            "p_z": p_z
        }

        return derivatives_dict

    def RANS(self, cos_wind_angle, sin_wind_angle, x, y, z, rho, nu, output_params):
        x.requires_grad_(True)
        y.requires_grad_(True)
        z.requires_grad_(True)
        input_data = torch.hstack((x, y, z, cos_wind_angle, sin_wind_angle))
        output = self(input_data)
        extracted_outputs = self.extract_output_parameters(output, output_params)
        output_dict = {param: value for param, value in zip(output_params, extracted_outputs)}
        p = output_dict.get('Pressure')
        u = output_dict.get('Velocity:0')
        v = output_dict.get('Velocity:1')
        w = output_dict.get('Velocity:2')
        nu_t = output_dict.get('TurbVisc')
        nu_eff = nu_t + nu

        def compute_gradients_and_second_order_gradients(tensor, coord):
            grad_first = torch.autograd.grad(tensor, coord, grad_outputs=torch.ones_like(tensor), create_graph=True)[0]
            grad_second = torch.autograd.grad(grad_first, coord, grad_outputs=torch.ones_like(grad_first), create_graph=True)[0]
            return grad_first, grad_second

        def compute_gradients(tensor, coord):
            grad_first = torch.autograd.grad(tensor, coord, grad_outputs=torch.ones_like(tensor), create_graph=True)[0]
            return grad_first

        def compute_second_order_gradient(first_derivative, coord):
            grad_second = torch.autograd.grad(first_derivative, coord, grad_outputs=torch.ones_like(first_derivative), create_graph=True)[0]
            return grad_second

        u_x, u_xx = compute_gradients_and_second_order_gradients(u, x)
        u_y, u_yy = compute_gradients_and_second_order_gradients(u, y)
        u_z, u_zz = compute_gradients_and_second_order_gradients(u, z)
        u_xy = compute_second_order_gradient(u_x, y)
        u_xz = compute_second_order_gradient(u_x, z)
        v_x, v_xx = compute_gradients_and_second_order_gradients(v, x)
        v_y, v_yy = compute_gradients_and_second_order_gradients(v, y)
        v_z, v_zz = compute_gradients_and_second_order_gradients(v, z)
        v_xy = compute_second_order_gradient(v_x, y)
        v_yz = compute_second_order_gradient(v_y, z)
        w_x, w_xx = compute_gradients_and_second_order_gradients(w, x)
        w_y, w_yy = compute_gradients_and_second_order_gradients(w, y)
        w_z, w_zz = compute_gradients_and_second_order_gradients(w, z)
        w_xz = compute_second_order_gradient(w_x, z)
        w_yz = compute_second_order_gradient(w_y, z)

        p_x = compute_gradients(p, x)
        p_y = compute_gradients(p, y)
        p_z = compute_gradients(p, z)

        f = (u * u_x + v * u_y + w * u_z - (1 / rho) * p_x + nu_eff * (2 * u_xx) + nu_eff * (u_yy + v_xy) + nu_eff * (u_zz + w_xz))
        g = (u * v_x + v * v_y + w * v_z - (1 / rho) * p_y + nu_eff * (v_xx + u_xy) + nu_eff * (2 * v_yy) + nu_eff * (v_zz + w_yz))
        h = (u * w_x + v * w_y + w * w_z - (1 / rho) * p_z + nu_eff * (w_xx + u_xz) + nu_eff * (w_yy + v_yz) + nu_eff * (2 * w_zz))
        
        return f, g, h

    def compute_physics_momentum_loss(self, X, input_params, output_params, rho, nu):
        extracted_inputs = self.extract_input_parameters(X, input_params)
        input_dict = {param: value for param, value in zip(input_params, extracted_inputs)}
        x = input_dict.get('Points:0')
        y = input_dict.get('Points:1')
        z = input_dict.get('Points:2')
        cos_wind_angle = input_dict.get('cos(WindAngle)')
        sin_wind_angle = input_dict.get('sin(WindAngle)')
        f, g, h = self.RANS(cos_wind_angle, sin_wind_angle, x, y, z, rho, nu, output_params)
        loss_f = nn.MSELoss()(f, torch.zeros_like(f))
        loss_g = nn.MSELoss()(g, torch.zeros_like(g))
        loss_h = nn.MSELoss()(h, torch.zeros_like(h))
        loss_physics_momentum = loss_f + loss_g + loss_h 
        return loss_physics_momentum

    def compute_physics_cont_loss(self, X, input_params, output_params, data=None):
        criterion = nn.MSELoss()
        extracted_inputs = self.extract_input_parameters(X, input_params)
        input_dict = {param: value for param, value in zip(input_params, extracted_inputs)}
        x = input_dict.get('Points:0')
        y = input_dict.get('Points:1')
        z = input_dict.get('Points:2')
        cos_wind_angle = input_dict.get('cos(WindAngle)')
        sin_wind_angle = input_dict.get('sin(WindAngle)')
        cont = self.continuity(cos_wind_angle, sin_wind_angle, x, y, z, output_params)
        if data is None:
            loss_cont = criterion(cont, torch.zeros_like(cont))
        else:
            loss_cont = criterion(cont, data)
        return loss_cont