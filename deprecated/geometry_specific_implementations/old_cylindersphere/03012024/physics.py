from definitions import *
from PINN import *

def extract_output_parameters(Y, output_params):
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

def compute_gradients_and_second_order_gradients(tensor, coord):
    grad_first = torch.autograd.grad(tensor, coord, grad_outputs=torch.ones_like(tensor), create_graph=True)[0]
    grad_second = torch.autograd.grad(grad_first, coord, grad_outputs=torch.ones_like(grad_first), create_graph=True)[0]
    return grad_first, grad_second

def compute_gradients(tensor, coord):
    grad_first = torch.autograd.grad(tensor, coord, grad_outputs=torch.ones_like(tensor), create_graph=True)[0]
    return grad_first

def compute_divergence(model, cos_wind_angle, sin_wind_angle, x, y, z, output_params):
    x.requires_grad_(True)
    y.requires_grad_(True)
    z.requires_grad_(True)
    input_data = torch.hstack((x, y, z, cos_wind_angle, sin_wind_angle))
    output = model(input_data)
    extracted_outputs = extract_output_parameters(output, output_params)
    output_dict = {param: value for param, value in zip(output_params, extracted_outputs)}
    u = output_dict.get('Velocity:0')
    v = output_dict.get('Velocity:1')
    w = output_dict.get('Velocity:2')

    u_x = compute_gradients(u, x)
    v_y = compute_gradients(v, y)
    w_z = compute_gradients(w, z)

    return u_x, v_y, w_z

def continuity(model, cos_wind_angle, sin_wind_angle, x, y, z, output_params):
    u_x, v_y, w_z = compute_divergence(model, cos_wind_angle, sin_wind_angle, x, y, z, output_params)
    cont = u_x + v_y + w_z
    return cont

def RANS(model, cos_wind_angle, sin_wind_angle, x, y, z, rho, nu, output_params):
    x.requires_grad_(True)
    y.requires_grad_(True)
    z.requires_grad_(True)
    input_data = torch.hstack((x, y, z, cos_wind_angle, sin_wind_angle))
    output = model(input_data)
    extracted_outputs = extract_output_parameters(output, output_params)
    output_dict = {param: value for param, value in zip(output_params, extracted_outputs)}
    p = output_dict.get('Pressure')
    u = output_dict.get('Velocity:0')
    v = output_dict.get('Velocity:1')
    w = output_dict.get('Velocity:2')
    nu_t = output_dict.get('TurbVisc')
    nu_eff = nu_t + nu

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
    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_z = torch.autograd.grad(p, z, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    f = (u * u_x + v * u_y + w * u_z - (1 / rho) * p_x + nu_eff * (2 * u_xx) + nu_eff * (u_yy + v_xy) + nu_eff * (u_zz + w_xz))
    g = (u * v_x + v * v_y + w * v_z - (1 / rho) * p_y + nu_eff * (v_xx + u_xy) + nu_eff * (2 * v_yy) + nu_eff * (v_zz + w_yz))
    h = (u * w_x + v * w_y + w * w_z - (1 / rho) * p_z + nu_eff * (w_xx + u_xz) + nu_eff * (w_yy + v_yz) + nu_eff * (2 * w_zz))
    return f, g, h