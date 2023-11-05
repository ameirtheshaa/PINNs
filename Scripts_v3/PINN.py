from definitions import *
from training import *
from plotting import *

class PINN(nn.Module):
    def __init__(self, neurons):
        super(PINN, self).__init__()
        self.fc1 = nn.Linear(5, neurons)
        self.fc2 = nn.Linear(neurons, neurons)
        self.fc3 = nn.Linear(neurons, neurons)
        self.fc4 = nn.Linear(neurons, neurons)
        self.output_layer = nn.Linear(neurons, 5)
        
    def forward_relu(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        output = self.output_layer(x)

        return output

    def forward_tanh(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        output = self.output_layer(x)

        return output

    def forward(self, x, activation):
        if activation == 'relu':
            return self.forward_relu(x)
        elif activation == 'tanh':
            return self.forward_tanh(x)
        else:
            return self.forward_relu(x)

    def compute_data_loss(self, X, y, activation):
        criterion = nn.MSELoss()
        predictions = self(X, activation)
        loss = criterion(predictions, y)
        return loss

    def compute_boundary_loss(self, X_boundary, y_boundary_known, activation):
        criterion = nn.MSELoss()
        boundary_predictions = self(X_boundary, activation)
        
        # Only consider the velocity components (u, v, w) for the boundary loss
        boundary_loss = criterion(boundary_predictions[:, :3], y_boundary_known[:, :3])
        
        return boundary_loss
    
    def extract_parameters(self, X):
        x = X[:, 0:1]  # Points:0
        y = X[:, 1:2]  # Points:1
        z = X[:, 2:3]  # Points:2
        wind_angle = X[:, 3:4]  # WindAngle

        return wind_angle, x, y, z

    def RANS(self, wind_angle, x, y, z, rho, nu, activation):
        # Ensure x, y, and z have requires_grad set to True
        x.requires_grad_(True)
        y.requires_grad_(True)
        z.requires_grad_(True)

        # Compute absolute values of sin and cos of wind_angle
        abs_cos_theta = torch.abs(torch.cos(torch.deg2rad(wind_angle)))
        abs_sin_theta = torch.abs(torch.sin(torch.deg2rad(wind_angle)))

        # Stack the input data
        input_data = torch.hstack((x, y, z, abs_cos_theta, abs_sin_theta))

        # Get the output and stress_tensor from the model
        output = self(input_data, activation)

        # Extract u, v, w, and nut_t from the output
        u, v, w, p, nu_t = output[:, 0], output[:, 1], output[:, 2], output[:, 3], output[:, 4]

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

        # # Define Stress Tensor

        # # Compute f, g, h, and cont using the obtained gradients and second order gradients
        # f = rho * (u * u_x + v * u_y + w * u_z) + p_x - nu * (u_xx + u_yy + u_zz) - stress_tensor[:, 0] - stress_tensor[:, 3] - stress_tensor[:, 4]
        # g = rho * (u * v_x + v * v_y + w * v_z) + p_y - nu * (v_xx + v_yy + v_zz) - stress_tensor[:, 3] - stress_tensor[:, 1] - stress_tensor[:, 5]
        # h = rho * (u * w_x + v * w_y + w * w_z) + p_z - nu * (w_xx + w_yy + w_zz) - stress_tensor[:, 4] - stress_tensor[:, 5] - stress_tensor[:, 2]

        cont = u_x + v_y + w_z

        # return u, v, w, p, f, g, h, cont

        return u, v, w, p, nu_t, cont

    # def compute_physics_momentum_loss(self, X, activation):
    #     wind_angle, x, y, z = self.extract_parameters(X)  # Extract the required input parameters for the RANS function
    #     u_pred, v_pred, w_pred, p_pred, f, g, h, cont = self.RANS(wind_angle, x, y, z, nu_t, rho, nu)
        
    #     loss_f = nn.MSELoss()(f, torch.zeros_like(f))
    #     loss_g = nn.MSELoss()(g, torch.zeros_like(g))
    #     loss_h = nn.MSELoss()(h, torch.zeros_like(h))
        
    #     loss_physics_momentum = loss_f + loss_g + loss_h 
    #     return loss_physics_momentum

    def compute_physics_cont_loss(self, X, rho, nu, activation):
        wind_angle, x, y, z = self.extract_parameters(X)  # Extract the required input parameters for the RANS function
        u_pred, v_pred, w_pred, p_pred, nu_t_pred, cont = self.RANS(wind_angle, x, y, z, rho, nu, activation)
        
        loss_cont = nn.MSELoss()(cont, torch.zeros_like(cont))

        return loss_cont