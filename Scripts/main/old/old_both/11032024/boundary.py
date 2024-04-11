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

def no_slip_loss_params(geometry_filename):
    epsilon = 100

    your_mesh = mesh.Mesh.from_file(geometry_filename)
    flattened_mesh = your_mesh.vectors.reshape(-1, 3)

    return epsilon, flattened_mesh

def inlet_loss_params(config):
    min_x_range = config["training"]["boundary"][0][0]
    max_x_range = config["training"]["boundary"][0][1]
    min_y_range = config["training"]["boundary"][0][2]
    max_y_range = config["training"]["boundary"][0][3]
    min_z_range = config["training"]["boundary"][0][4]
    max_z_range = config["training"]["boundary"][0][5]
    num_points_boundary = config["training"]["boundary"][1]

    x_range = max_x_range - min_x_range
    y_range = max_y_range - min_y_range
    z_range = max_z_range - min_z_range

    return min_x_range, max_x_range, x_range, min_y_range, max_y_range, y_range, min_z_range, max_z_range, z_range, num_points_boundary

def no_slip_boundary_conditions(config, device, feature_scaler):
    """
    Returns the sampled points on the geometry surface along with their corresponding no-slip velocities and the normals.

    Parameters:
    - geometry: List of points on the surface of the geometry
    - wind_angle: Angle of the inlet wind direction

    Returns:
    - no_slip_points: Coordinates of the sampled points on the geometry's surface in the representation expected by the NN.
    - no_slip_velocities: Velocities (all zeros) at the sampled points, representing the no-slip condition.
    - no_slip_normals: Normal vectors corresponding to the coordinates of the sampled points on the geometry's surface.
    """

    chosen_machine_key = config["chosen_machine"]
    datafolder_path = os.path.join(config["machine"][chosen_machine_key], config["data"]["data_folder_name"])
    wind_angles = config["training"]["angles_to_train"]
    geometry_filename = os.path.join(datafolder_path,'scaled_cylinder_sphere.stl')

    epsilon, flattened_mesh = no_slip_loss_params(geometry_filename)
    faces = restructure_data(flattened_mesh)
    no_slip_features_normals = []

    for wind_angle in wind_angles:

        subfaces = random.sample(faces, 10000)
        no_slip_normals = [compute_normal(*face) for face in subfaces]
        no_slip_normals = np.array(no_slip_normals)

        x_coords = []
        y_coords = []
        z_coords = []

        for a,b,c in subfaces:
            x_coord_a, y_coord_a, z_coord_a = a[0], a[1], a[2]
            x_coord_b, y_coord_b, z_coord_b = b[0], b[1], b[2]
            x_coord_c, y_coord_c, z_coord_c = c[0], c[1], c[2]
            x_coord = np.mean([x_coord_a,x_coord_b,x_coord_c])
            y_coord = np.mean([y_coord_a,y_coord_b,y_coord_c])
            z_coord = np.mean([z_coord_a,z_coord_b,z_coord_c])
            x_coords.append(x_coord)
            y_coords.append(y_coord)
            z_coords.append(z_coord)

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

        normalized_features = feature_scaler.transform(df_features)

        # Convert to Torch tensor
        no_slip_points = torch.tensor(normalized_features, dtype=torch.float32)
        no_slip_normals = torch.tensor(no_slip_normals, dtype=torch.float32)

        no_slip_points = no_slip_points.to(device)
        no_slip_normals = no_slip_normals.to(device)

        no_slip_features_normals.append([epsilon, wind_angle, no_slip_points, no_slip_normals])

    return no_slip_features_normals

def sample_domain_boundary(config, device, feature_scaler, target_scaler):

    chosen_machine_key = config["chosen_machine"]
    datafolder_path = os.path.join(config["machine"][chosen_machine_key], config["data"]["data_folder_name"])
    meteo_filenames = get_filenames_from_folder(datafolder_path, config["data"]["extension"], config["data"]["startname_meteo"])
    wind_angles = config["training"]["angles_to_train"]
    
    min_x_range, max_x_range, x_range, min_y_range, max_y_range, y_range, min_z_range, max_z_range, z_range, num_points = inlet_loss_params(config)

    all_inlet_features_targets = []

    for wind_direction in wind_angles:
        df_all = []
        altitudes = np.arange(min_z_range+1,max_z_range+1,20)
        for filename in sorted(meteo_filenames):
            wind_angle = int(filename.split('_')[1].split('dd')[1])
            if wind_direction == wind_angle:
                df = pd.read_csv(os.path.join(datafolder_path, filename))

                altitude = df['Altitude']
                u = df['u']
                v = df['v']

                # Function to calculate trend line and R² for a given series
                def calculate_trend_and_r2(altitude, series):
                    # Transform altitude values for logarithmic trend line
                    log_altitude = np.log(altitude)
                    coefficients = np.polyfit(log_altitude, series, 1)
                    trend_line = np.polyval(coefficients, log_altitude)

                    # Calculate R2 value
                    slope, intercept, r_value, _, _ = linregress(log_altitude, series)
                    r2 = r_value**2

                    return trend_line, coefficients, r2

                # Calculate for 'u' and 'v'
                u_trend_line, u_coefficients, u_r2 = calculate_trend_and_r2(altitude, u)
                v_trend_line, v_coefficients, v_r2 = calculate_trend_and_r2(altitude, v)

                # Trend line equations
                u_slope, u_intercept = u_coefficients
                v_slope, v_intercept = v_coefficients
                u_trend_eq_ = f"u = {u_slope:.2f} * log(Altitude) + {u_intercept:.2f}"
                v_trend_eq_ = f"v = {v_slope:.2f} * log(Altitude) + {v_intercept:.2f}"

                for altitude_ in altitudes:
                    u_value = u_slope*np.log(altitude_) + u_intercept
                    v_value = v_slope*np.log(altitude_) + v_intercept
                    w_value = 0

                    # Side 1
                    side_1 = [(np.random.uniform(min_x_range, max_x_range), min_y_range, altitude_, np.cos(np.deg2rad(wind_angle)), np.sin(np.deg2rad(wind_angle)), u_value, v_value, w_value) for _ in range(num_points)]

                    # Side 2
                    side_2 = [(max_x_range, np.random.uniform(min_y_range, max_y_range), altitude_, np.cos(np.deg2rad(wind_angle)), np.sin(np.deg2rad(wind_angle)), u_value, v_value, w_value) for _ in range(num_points)]

                    # Side 3
                    side_3 = [(np.random.uniform(min_x_range, max_x_range), max_y_range, altitude_, np.cos(np.deg2rad(wind_angle)), np.sin(np.deg2rad(wind_angle)), u_value, v_value, w_value) for _ in range(num_points)]

                    # Side 4
                    side_4 = [(min_x_range, np.random.uniform(min_y_range, max_y_range), altitude_, np.cos(np.deg2rad(wind_angle)), np.sin(np.deg2rad(wind_angle)), u_value, v_value, w_value) for _ in range(num_points)]

                    # Concatenate the triplets
                    all_data = np.concatenate([side_1, side_2, side_3, side_4])

                    # Create a DataFrame
                    df_ = pd.DataFrame(all_data, columns=['Points:0', 'Points:1', 'Points:2', 'cos(WindAngle)', 'sin(WindAngle)', 'Velocity:0', 'Velocity:1', 'Velocity:2'])
                    df_all.append(df_)

        if len(df_all) != 0:
            boundary_data = pd.concat(df_all)
            
            features = boundary_data[config["training"]["input_params"]]
            targets = boundary_data[['Velocity:0', 'Velocity:1', 'Velocity:2']]

            normalized_features = feature_scaler.transform(features)
            normalized_targets = pd.DataFrame()
            stds_means_dict = extract_stds_means(target_scaler, config["training"]["output_params"])
            
            for col in ['Velocity:0', 'Velocity:1', 'Velocity:2']:
                col_std = col + "_std"
                col_mean = col + "_mean"
                normalized_targets[col] = (targets[col] - stds_means_dict[col_mean]) / (stds_means_dict[col_std])
                
            normalized_features_tensor = torch.tensor(normalized_features, dtype=torch.float32)
            normalized_targets_tensor = torch.tensor(normalized_targets.values, dtype=torch.float32)

            normalized_features_tensor = normalized_features_tensor.to(device)
            normalized_targets_tensor = normalized_targets_tensor.to(device)

            all_inlet_features_targets.append([wind_angle, normalized_features_tensor, normalized_targets_tensor])

    return all_inlet_features_targets