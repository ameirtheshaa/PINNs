from definitions import * 
from plotting import *
from physics import *

def evaluation_boundary_physics(model, device, config, data_dict, model_file_path, output_folder, today, overall_start_time, physics):
    testing_type = f"Boundary Evaluation {physics}"
    chosen_machine_key = config["chosen_machine"]
    datafolder_path = os.path.join(config["machine"][chosen_machine_key], config["data"]["data_folder_name"])
    geometry_filename = os.path.join(datafolder_path, config["data"]["geometry"])
    rho = config["data"]["density"]
    nu = config["data"]["kinematic_viscosity"]
    checkpoint = open_model_file(model_file_path, device)
    wind_angles = config["training"]["all_angles"]
    print (f'Model {testing_type} at Epoch = {checkpoint["epoch"]} and Training Completed = {checkpoint["training_completed"]}, Time: {(time.time() - overall_start_time):.2f} Seconds')
    model.load_state_dict(checkpoint['model_state_dict'])
    feature_scaler = data_dict["feature_scaler"]
    target_scaler = data_dict["target_scaler"]
    for wind_angle in wind_angles:
        X_test_tensor = load_boundary_testing_data_per_angle(config, device, feature_scaler, wind_angle)
        df_torch = evaluate_model_physics(config, model, [wind_angle], X_test_tensor, feature_scaler, target_scaler, output_folder, physics)
        print (f'Model Evaluated and Starting to Plot for Wind Angle = {wind_angle}, Time: {(time.time() - overall_start_time):.2f} Seconds')
        plot_physics_boundary_2d(df_torch,physics,wind_angle,config,output_folder)
        print (f'Plotting Done for Wind Angle = {wind_angle}, Time: {(time.time() - overall_start_time):.2f} Seconds')

def evaluate_model_boundary(config, model, wind_angles, X_test_tensor, feature_scaler, target_scaler):
    model.eval()
    dfs = []
    with torch.no_grad():
        predictions_tensor = model(X_test_tensor)
        X_test_tensor_cpu = X_test_tensor.cpu()
        X_test_tensor_cpu = inverse_transform_features(X_test_tensor_cpu, feature_scaler)
        X_test_column_names = config["training"]["input_params_modf"]
        X_test_dataframe = pd.DataFrame(X_test_tensor_cpu, columns=X_test_column_names)
        predictions_tensor_cpu = predictions_tensor.cpu()
        predictions_tensor_cpu = inverse_transform_targets(predictions_tensor_cpu, target_scaler)
        predictions_column_names = config["training"]["output_params_modf"]
        predictions_dataframe = pd.DataFrame(predictions_tensor_cpu, columns=predictions_column_names)
        predictions_dataframe['Velocity_Magnitude'] = np.sqrt(predictions_dataframe['Velocity_X']**2 + predictions_dataframe['Velocity_Y']**2 + predictions_dataframe['Velocity_Z']**2)
        for wind_angle in wind_angles:
            lower_bound = wind_angle - 2
            upper_bound = wind_angle + 2
            X_test_dataframe['WindAngle_rad'] = np.arctan2(X_test_dataframe['sin(WindAngle)'], X_test_dataframe['cos(WindAngle)'])
            X_test_dataframe['WindAngle'] = X_test_dataframe['WindAngle_rad'].apply(lambda x: int(np.ceil(np.degrees(x))))
            mask = X_test_dataframe['WindAngle'].between(lower_bound, upper_bound)
            filtered_X_test_dataframe = X_test_dataframe.loc[mask]
            filtered_predictions = predictions_dataframe.loc[mask]
            if len(filtered_predictions)!= 0:
                combined_df = pd.concat([filtered_X_test_dataframe, filtered_predictions], axis=1)
                combined_df['WindAngle'] = (wind_angle)
                dfs.append(combined_df)
    data = pd.concat(dfs)
    return data

def concatenate_boundary_data(filenames, datafolder_path, wind_angles):
    dfs = []
    for filename in filenames:
        for wind_angle in wind_angles:
            df = pd.read_csv(os.path.join(datafolder_path, filename))
            df['cos(WindAngle)'] = (np.cos(np.deg2rad(wind_angle)))
            df['sin(WindAngle)'] = (np.sin(np.deg2rad(wind_angle)))
            df['WindAngle'] = (wind_angle)
            dfs.append(df)
    data = pd.concat(dfs)
    return data 

def load_boundary_testing_data(config, device, feature_scaler, filename):
    chosen_machine_key = config["chosen_machine"]
    datafolder_path = os.path.join(config["machine"][chosen_machine_key], config["data"]["data_folder_name"])
    filenames = get_filenames_from_folder(datafolder_path, config["data"]["extension"], filename)
    all_wind_angles = config["training"]["all_angles"]
    data = concatenate_boundary_data(filenames, datafolder_path, all_wind_angles)
    features = data[config["training"]["input_params"]]
    normalized_features = feature_scaler.transform(features)
    X_test_tensor = torch.tensor(np.array(normalized_features), dtype=torch.float32)
    X_test_tensor = X_test_tensor.to(device)
    return X_test_tensor

def load_boundary_testing_data_per_angle(config, device, feature_scaler, wind_angle):
    chosen_machine_key = config["chosen_machine"]
    datafolder_path = os.path.join(config["machine"][chosen_machine_key], config["data"]["data_folder_name"])
    filenames = get_filenames_from_folder(datafolder_path, config["data"]["extension"], 'geometry_test_points')
    data = concatenate_boundary_data(filenames, datafolder_path, [wind_angle])
    features = data[config["training"]["input_params"]]
    normalized_features = feature_scaler.transform(features)
    X_test_tensor = torch.tensor(np.array(normalized_features), dtype=torch.float32)
    X_test_tensor = X_test_tensor.to(device)

    return X_test_tensor

def plot_boundary_2d(config,df,wind_angles,geometry_filename,plot_folder):
    params = config["plotting"]["plotting_params"]
    os.makedirs(plot_folder, exist_ok=True)
    for wind_angle in wind_angles:
        for j in params:
            plane = j[0]
            cut = j[1]
            tolerance = j[2]
            savenames_scatter_single = [['Velocity_Magnitude', 'Velocity Magnitude', os.path.join(plot_folder,f'{plane}_totalvelocity_scatter_{wind_angle}.png')],
            ['Velocity_X', 'Velocity X', os.path.join(plot_folder,f'{plane}_vx_scatter_{wind_angle}.png')],
            ['Velocity_Y', 'Velocity Y', os.path.join(plot_folder,f'{plane}_vy_scatter_{wind_angle}.png')],
            ['Velocity_Z', 'Velocity Z', os.path.join(plot_folder,f'{plane}_vz_scatter_{wind_angle}.png')],
            ['Pressure', 'Pressure', os.path.join(plot_folder,f'{plane}_pressure_scatter_{wind_angle}.png')],
            ['TurbVisc', 'TurbVisc', os.path.join(plot_folder,f'{plane}_turbvisc_scatter_{wind_angle}.png')]]
            plot_data_scatter(df, plane, wind_angle, cut, tolerance, geometry_filename, savenames_scatter_single)

def plot_physics_boundary(df,wind_angle,config,datafolder_path,savename,plane,cut,tolerance, physics):
    coordinate1, coordinate2, coordinate3, lim_min1, lim_max1, lim_min2, lim_max2 = get_plane_config(plane)
    filtered_df = filter_dataframe(df, wind_angle, coordinate3, cut, tolerance)
    grid1, grid2 = define_scatter_grid(filtered_df, coordinate1, coordinate2)
    grid1, grid2 = normalize_grids(plane, grid1, grid2)
    cut, tolerance = normalize_cut_tolerance(plane, cut, tolerance)
    grid3 = filtered_df[physics]

    fig, ax = plt.subplots(figsize=(32,16), sharey=True)
    fig.suptitle(f'Comparison of Actual vs. Predicted values with Wind Angle = {wind_angle} in the {plane} Plane with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')
    vmin, vmax, levels, cmap, scatter_size, ticks = plotting_details(grid3)
    scatter_actual = ax.scatter(grid1, grid2, c=grid3, s=scatter_size, vmin=np.min(grid3), vmax=np.max(grid3), cmap=cmap)
    scatter_cbar_actual = fig.colorbar(scatter_actual, ax=ax)
    scatter_cbar_actual.set_label(f'{physics} Residual - Actual', rotation=270, labelpad=15)
    ax.set_xlabel(f'{coordinate1} Coordinate')
    ax.set_ylabel(f'{coordinate2} Coordinate')
    ax.set_xlim(lim_min1, lim_max1) 
    ax.set_ylim(lim_min2, lim_max2)
    ax.set_title(f'Boundary {physics} Residual in the {plane} Plane for Wind Angle = {wind_angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f} \n Mean = {np.mean(grid3):.2f} and Standard Deviation = {np.std(grid3):.2f}')
    plt.tight_layout()
    plt.savefig(savename)
    plt.close()

def plot_physics_boundary_2d(df_torch,physics,angle,config,plot_folder):
    params = config["plotting"]["plotting_params"]
    chosen_machine_key = config["chosen_machine"]
    datafolder_path = os.path.join(config["machine"][chosen_machine_key], config["data"]["data_folder_name"])
    for j in params:
        plane = j[0]
        cut = j[1]
        tolerance = j[2]
        savename = os.path.join(plot_folder,f'{plane}_{physics}_{angle}.png')
        plot_physics_boundary(df_torch,angle,config,datafolder_path,savename,plane,cut,tolerance,physics)