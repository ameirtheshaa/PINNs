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

def parameters(datafolder_path, config):
    num_points_sphere = 0.01
    num_points_cylinder = 0.01
    epsilon = 100
    x_range = 1000
    y_range = 1000

    df_sphere = pd.read_csv(os.path.join(datafolder_path,'sphere.csv'))
    df_cylinder = pd.read_csv(os.path.join(datafolder_path,'cylinder.csv'))
    faces_sphere = df_sphere.values.tolist()
    faces_cylinder = df_cylinder.values.tolist()

    return num_points_sphere, num_points_cylinder, epsilon, x_range, y_range, faces_sphere, faces_cylinder

def no_slip_boundary_conditions(device, faces_data, num_points, wind_angle, feature_scaler, target_scaler):
    """
    Returns the sampled points on the geometry surface along with their corresponding no-slip velocities and the normals.

    Parameters:
    - geometry: List of points on the surface of the geometry
    - num_points: Percentage of points to sample on the geometry (0,1]
    - wind_angle: Angle of the inlet wind direction

    Returns:
    - no_slip_points: Coordinates of the sampled points on the geometry's surface in the representation expected by the NN.
    - no_slip_velocities: Velocities (all zeros) at the sampled points, representing the no-slip condition.
    - no_slip_normals: Normal vectors corresponding to the coordinates of the sampled points on the geometry's surface.
    """

    def compute_centroid(face):
        return [sum(coord) / 3 for coord in zip(*face)]

    faces = restructure_data(faces_data)
    subfaces = random.sample(faces, int(len(faces) * num_points))
    no_slip_normals = [compute_normal(*face) for face in subfaces]
    no_slip_normals = np.array(no_slip_normals)

    # Compute centroids for each face
    centroids = [compute_centroid(face) for face in subfaces]

    # Separate x, y, and z coordinates
    x_coords = [point[0] for point in centroids]
    y_coords = [point[1] for point in centroids]
    z_coords = [point[2] for point in centroids]

    # Convert to NumPy arrays
    x_array = np.array(x_coords)
    y_array = np.array(y_coords)
    z_array = np.array(z_coords)

    # Create the DataFrame
    df_features = pd.DataFrame({
        'Points:0': x_array,
        'Points:1': y_array,
        'Points:2': z_array
    })

    # Add cos(WindAngle) and sin(WindAngle) as new columns
    df_features['cos(WindAngle)'] = np.cos(np.deg2rad(wind_angle))
    df_features['sin(WindAngle)'] = np.sin(np.deg2rad(wind_angle))

    df_targets = pd.DataFrame(index=range(len(df_features)))

    # # Add zero velocities
    # df_targets['Pressure'] = 0.0
    # df_targets['Velocity:0'] = 0.0
    # df_targets['Velocity:1'] = 0.0
    # df_targets['Velocity:2'] = 0.0
    # df_targets['TurbVisc'] = 0.0

    # # Scale the features accordingly to the distribution of the original dataset
    # normalized_features, normalized_targets = transform_data_with_scalers(df_features, df_targets, feature_scaler, target_scaler)

    normalized_features = feature_scaler.transform(df_features)

    # Convert to Torch tensor
    no_slip_points = torch.tensor(normalized_features, dtype=torch.float32)
    # no_slip_velocities = torch.tensor(normalized_targets, dtype=torch.float32)
    no_slip_normals = torch.tensor(no_slip_normals, dtype=torch.float32)

    no_slip_points = no_slip_points.to(device)
    # no_slip_velocities = no_slip_velocities.to(device)
    no_slip_normals = no_slip_normals.to(device)

    return no_slip_points, no_slip_normals

def sample_domain_boundary(device, filenames, datafolder_path, x_range, y_range, wind_direction):
    num_points = 10
    for filename in sorted(filenames):
        wind_angle = int(filename.split('_')[1].split('dd')[1])
        if wind_direction == wind_angle:
            print ('yes', wind_direction, wind_angle)
            df = pd.read_csv(os.path.join(datafolder_path, filename))
            for i in df.values:
                altitude = int(i[0])
                u_value = float(i[1])
                v_value = float(i[2])
                w_value = 0
                k = float(i[3])
                eps = float(i[4])
                u = np.ones(num_points)*u_value
                v = np.ones(num_points)*v_value
                w = np.ones(num_points)*w_value

                # Side 1
                side_1 = [(i / (num_points - 1) * 1000, 0, altitude) for i in range(num_points)]
                # Side 2
                side_2 = [(1000, i / (num_points - 1) * 1000, altitude) for i in range(num_points)]
                # Side 3
                side_3 = [(1000 - i / (num_points - 1) * 1000, 1000, altitude) for i in range(num_points)]
                # Side 4
                side_4 = [(0, 1000 - i / (num_points - 1) * 1000, altitude) for i in range(num_points)]


                print (side_1)
                print (side_2)
                print (side_3)
                print (side_4)

                print (u)
                print (v)
                print (w)

                # Create a figure and a set of subplots
                fig, axs = plt.subplots(2, 2, figsize=(10, 10))  # Adjust the size as needed

                # Plotting for each side
                # Side 1
                x1, y1, z1 = zip(*side_1)
                axs[0, 0].quiver(x1, y1, u, v, scale=5)
                axs[0, 0].set_title('Side 1')
                axs[0, 0].set_xlim([0, 1000])
                axs[0, 0].set_ylim([0, 1000])

                # Side 2
                x2, y2, z2 = zip(*side_2)
                axs[0, 1].quiver(x2, y2, u, v, scale=5)
                axs[0, 1].set_title('Side 2')
                axs[0, 1].set_xlim([0, 1000])
                axs[0, 1].set_ylim([0, 1000])

                # Side 3
                x3, y3, z3 = zip(*side_3)
                axs[1, 0].quiver(x3, y3, u, v, scale=5)
                axs[1, 0].set_title('Side 3')
                axs[1, 0].set_xlim([0, 1000])
                axs[1, 0].set_ylim([0, 1000])

                # Side 4
                x4, y4, z4 = zip(*side_4)
                axs[1, 1].quiver(x4, y4, u, v, scale=5)
                axs[1, 1].set_title('Side 4')
                axs[1, 1].set_xlim([0, 1000])
                axs[1, 1].set_ylim([0, 1000])

                # Display the plot
                plt.savefig(f'wtf_{wind_direction}')








    # # Convert wind direction to radians
    # theta = np.deg2rad(wind_direction)
    # cos_theta = np.ones(num_points)
    # sin_theta = np.ones(num_points)
    # cos_theta *= np.abs(np.cos(theta))
    # sin_theta *= np.abs(np.sin(theta))
    
    
    # if 45 <= wind_direction < 135:  # Positive y-direction
    #     y = np.full(num_points, y_range[1])
    #     x = np.random.uniform(x_range[0], x_range[1], num_points)
    # elif 135 <= wind_direction < 225:  # Negative x-direction
    #     x = np.full(num_points, x_range[0])
    #     y = np.random.uniform(y_range[0], y_range[1], num_points)
    # elif 225 <= wind_direction < 315:  # Negative y-direction
    #     y = np.full(num_points, y_range[0])
    #     x = np.random.uniform(x_range[0], x_range[1], num_points)
    # else:  # Positive x-direction
    #     x = np.full(num_points, x_range[1])
    #     y = np.random.uniform(y_range[0], y_range[1], num_points)
    
    # z = np.full(num_points, z_value)

    # input_X = np.vstack((x, y, z, cos_theta, sin_theta)).T
    # output_y = np.vstack((u, v, w)).T

    # input_X = torch.tensor(input_X, dtype=torch.float32)
    # output_y = torch.tensor(output_y, dtype=torch.float32)

    # input_X = input_X.to(device)
    # output_y = output_y.to(device)

    # return input_X, output_y