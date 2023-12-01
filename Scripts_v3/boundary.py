from definitions import *
from PINN import *

###################################################################################################
# 1) Dirichlet Boundary Conditions: 
# These specify the value of the solution at the boundary. 
# For example, you might know the wind velocity at the domain boundaries.
# 2) Neumann Boundary Conditions: 
# These specify the derivative of the solution at the boundary. 
# For instance, you might have information about how the wind velocity changes 
# as you approach the domain boundaries.
# 3) Slip/No-Slip Conditions: On the surface of the sphere and cylinder, 
# you might have no-slip conditions (velocity is zero) or 
# slip conditions (velocity is not zero, but its normal component is zero).
# 4) Inflow/Outflow Conditions: At the domain boundaries, you might specify which boundaries are 
# inflow (wind entering the domain) and which are outflow (wind exiting the domain).
# Initial Conditions: If this is a time-dependent problem, you'd also need initial conditions, 
# which specify the state of the system at the starting time.
###################################################################################################

def parameters(config):
	sphere_center = [500, 500, 50]
	sphere_radius = 5
	cylinder_base_center = [500, 570, 0]
	cylinder_radius = 7.5
	cylinder_height = 64
	cylinder_cap_height = 1
	cylinder_cap_radius = 6.5
	num_points_sphere = 1000
	num_points_cylinder = 1000

	x_range = [0,1000]
	y_range = [0,1000]
	z_value = 100
	inlet_velocity = 4.043564066
	num_points_boundary = 1000

	wind_angles = config["training"]["all_angles"]

	return sphere_center, sphere_radius, cylinder_base_center, cylinder_radius, cylinder_height, cylinder_cap_height, cylinder_cap_radius, num_points_sphere, num_points_cylinder, x_range, y_range, z_value, inlet_velocity, num_points_boundary, wind_angles

def sample_sphere_surface(center, radius, num_points, wind_direction):
	# Convert wind direction to radians
    theta = np.deg2rad(wind_direction)
    cos_theta = np.ones(num_points)
    sin_theta = np.ones(num_points)
    cos_theta *= np.abs(np.cos(theta))
    sin_theta *= np.abs(np.sin(theta))

    # Sample points using spherical coordinates and convert to Cartesian
    phi = np.random.uniform(0, np.pi, num_points)
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    x = center[0] + radius * np.sin(phi) * np.cos(theta)
    y = center[1] + radius * np.sin(phi) * np.sin(theta)
    z = center[2] + radius * np.cos(phi)
    return np.vstack((x, y, z, cos_theta, sin_theta)).T

def sample_cylinder_surface(base_center, radius, height, cap_height, cap_radius, num_points, wind_direction):
	# Convert wind direction to radians
    theta = np.deg2rad(wind_direction)
    cos_theta = np.ones(num_points*2)
    sin_theta = np.ones(num_points*2)
    cos_theta *= np.abs(np.cos(theta))
    sin_theta *= np.abs(np.sin(theta))

    # Sample points on the lateral surface
    z_lateral = np.random.uniform(0, height - cap_height, num_points)
    theta_lateral = np.random.uniform(0, 2 * np.pi, num_points)
    x_lateral = base_center[0] + radius * np.cos(theta_lateral)
    y_lateral = base_center[1] + radius * np.sin(theta_lateral)
    
    # Sample points on the rounded cap
    phi_max = np.arcsin(cap_height / cap_radius)
    phi_cap = np.random.uniform(0, phi_max, num_points)
    theta_cap = np.random.uniform(0, 2 * np.pi, num_points)
    x_cap = base_center[0] + cap_radius * np.sin(phi_cap) * np.cos(theta_cap)
    y_cap = base_center[1] + cap_radius * np.sin(phi_cap) * np.sin(theta_cap)
    z_cap = base_center[2] + height - cap_height + cap_radius * (1 - np.cos(phi_cap))
    
    # Combine lateral and cap points
    x = np.concatenate([x_lateral, x_cap])
    y = np.concatenate([y_lateral, y_cap])
    z = np.concatenate([z_lateral, z_cap])
    
    return np.vstack((x, y, z, cos_theta, sin_theta)).T

def sample_domain_boundary(device, x_range, y_range, z_value, wind_direction, inlet_velocity, num_points):
    # Convert wind direction to radians
    theta = np.deg2rad(wind_direction)
    cos_theta = np.ones(num_points)
    sin_theta = np.ones(num_points)
    cos_theta *= np.abs(np.cos(theta))
    sin_theta *= np.abs(np.sin(theta))
    
    # Compute the x and y components of the inlet velocity
    u = np.ones(num_points)
    v = np.ones(num_points)
    w = np.ones(num_points)
    u *= inlet_velocity * np.cos(theta)
    v *= inlet_velocity * np.sin(theta)
    w *= inlet_velocity*0  # Assuming no vertical wind component
    
    if 45 <= wind_direction < 135:  # Positive y-direction
        y = np.full(num_points, y_range[1])
        x = np.random.uniform(x_range[0], x_range[1], num_points)
    elif 135 <= wind_direction < 225:  # Negative x-direction
        x = np.full(num_points, x_range[0])
        y = np.random.uniform(y_range[0], y_range[1], num_points)
    elif 225 <= wind_direction < 315:  # Negative y-direction
        y = np.full(num_points, y_range[0])
        x = np.random.uniform(x_range[0], x_range[1], num_points)
    else:  # Positive x-direction
        x = np.full(num_points, x_range[1])
        y = np.random.uniform(y_range[0], y_range[1], num_points)
    
    z = np.full(num_points, z_value)

    input_X = np.vstack((x, y, z, cos_theta, sin_theta)).T
    output_y = np.vstack((u, v, w)).T

    input_X = torch.tensor(input_X, dtype=torch.float32)
    output_y = torch.tensor(output_y, dtype=torch.float32)

    input_X = input_X.to(device)
    output_y = output_y.to(device)

    return input_X, output_y

def no_slip_boundary_conditions(device, sphere_center, sphere_radius, cylinder_base_center, cylinder_radius, cylinder_height, cylinder_cap_height, cylinder_cap_radius, num_points_sphere, num_points_cylinder, wind_direction):
    """
    Returns the sampled points on the sphere and cylinder surfaces along with their corresponding no-slip velocities.

    Parameters:
    - sphere_center: Center coordinates of the sphere [x, y, z].
    - sphere_radius: Radius of the sphere.
    - cylinder_base_center: Base center coordinates of the cylinder [x, y, z].
    - cylinder_radius: Radius of the cylinder.
    - cylinder_height: Height of the cylinder.
    - num_points_sphere: Number of points to sample on the sphere surface.
    - num_points_cylinder: Number of points to sample on the cylinder surface.

    Returns:
    - all_no_slip_points: Combined coordinates of the sampled points on the sphere and cylinder surfaces.
    - no_slip_velocities: Velocities (all zeros) at the sampled points, representing the no-slip condition.
    """

    # Sample points on the sphere and cylinder
    sphere_points = sample_sphere_surface(sphere_center, sphere_radius, num_points_sphere, wind_direction)
    cylinder_points = sample_cylinder_surface(cylinder_base_center, cylinder_radius, cylinder_height, cylinder_cap_height, cylinder_cap_radius, num_points_cylinder, wind_direction)

    # Combine the points
    all_no_slip_points = np.vstack([sphere_points, cylinder_points])

    # Set the velocity at these points to zero
    no_slip_velocities = np.zeros((len(all_no_slip_points), 3))  # Assuming 3D velocities (u, v, w)

    all_no_slip_points = torch.tensor(all_no_slip_points, dtype=torch.float32)
    no_slip_velocities = torch.tensor(no_slip_velocities, dtype=torch.float32)

    all_no_slip_points = all_no_slip_points.to(device)
    no_slip_velocities = no_slip_velocities.to(device)

    return all_no_slip_points, no_slip_velocities