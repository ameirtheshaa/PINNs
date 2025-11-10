from definitions import *
from PINN import *
from plotting_definitions import *
from physics import *
from losses import *

def get_divergence(model, X, input_params, output_params):
    extracted_inputs = extract_parameters(X, input_params)
    input_dict = {param: value for param, value in zip(input_params, extracted_inputs)}   
    x = input_dict.get('Points:0')
    y = input_dict.get('Points:1')
    z = input_dict.get('Points:2')
    cos_wind_angle = input_dict.get('cos(WindAngle)')
    sin_wind_angle = input_dict.get('sin(WindAngle)')  
    u_x, v_y, w_z = compute_divergence(model, cos_wind_angle, sin_wind_angle, x, y, z, output_params)
    return u_x, v_y, w_z

def evaluate_model_new(config, model, wind_angles, X_test_tensor, feature_scaler, target_scaler):
	model.eval()
	dfs = []
	input_params = config["training"]["input_params"]
	output_params = config["training"]["output_params"]
	u_x, v_y, w_z = get_divergence(model, X_test_tensor, input_params, output_params)
	div_dict = {
	"u_x": u_x.cpu().detach().numpy().flatten(),
	"v_y": v_y.cpu().detach().numpy().flatten(),
	"w_z": w_z.cpu().detach().numpy().flatten()
	}
	div_dataframe = pd.DataFrame(div_dict)
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
			filtered_div = div_dataframe.loc[mask]
			if len(filtered_predictions)!= 0:
				combined_df = pd.concat([filtered_X_test_dataframe, filtered_predictions, filtered_div], axis=1)
				combined_df['WindAngle'] = (wind_angle)
				dfs.append(combined_df)
	data = pd.concat(dfs)
	return data

def get_divergence_from_torch(df, plane, coordinate1, coordinate2, grid1, grid2):
	dvx_dx = griddata(df[[coordinate1, coordinate2]].values, df['u_x'].values, (grid1, grid2), method='linear', fill_value=0)
	dvy_dy = griddata(df[[coordinate1, coordinate2]].values, df['v_y'].values, (grid1, grid2), method='linear', fill_value=0)
	dvz_dz = griddata(df[[coordinate1, coordinate2]].values, df['w_z'].values, (grid1, grid2), method='linear', fill_value=0)
	if plane == 'X-Z':
		divergence = dvx_dx + dvz_dz
	elif plane == 'Y-Z':
		divergence = dvy_dy + dvz_dz
	elif plane == 'X-Y':
		divergence = dvx_dx + dvy_dy
	divergence_flat = divergence.flatten()
	return divergence, divergence_flat

def get_divergence_from_grid(plane, grid1, grid2, grid_vx, grid_vy, grid_vz):
	def calculate_derivatives(grid_v, d, axis):
		return (np.roll(grid_v, -1, axis=axis) - np.roll(grid_v, 1, axis=axis)) / (2 * d)
	dx = grid1[1, 0] - grid1[0, 0]
	dy = grid2[0, 1] - grid2[0, 0]
	if plane == 'X-Z':
		dvx_dx = calculate_derivatives(grid_vx, dx, axis=0)  # Derivative with respect to X
		dvz_dz = calculate_derivatives(grid_vz, dy, axis=1)  # Derivative with respect to Z
		divergence = dvx_dx + dvz_dz
	elif plane == 'Y-Z':
		dvy_dy = calculate_derivatives(grid_vy, dx, axis=0)  # Derivative with respect to Y
		dvz_dz = calculate_derivatives(grid_vz, dy, axis=1)  # Derivative with respect to Z
		divergence = dvy_dy + dvz_dz
	elif plane == 'X-Y':
		dvx_dx = calculate_derivatives(grid_vx, dx, axis=0)  # Derivative with respect to X
		dvy_dy = calculate_derivatives(grid_vy, dy, axis=1)  # Derivative with respect to Y
		divergence = dvx_dx + dvy_dy
	divergence_flat = divergence.flatten()
	return divergence, divergence_flat

def plot_data_predictions(df,df_data,wind_angle,config,datafolder_path,savename_total_div, savename_total_velocity,plane,cut,tolerance):
	coordinate1, coordinate2, coordinate3, lim_min1, lim_max1, lim_min2, lim_max2 = get_plane_config(plane)
	filtered_df = filter_dataframe(df, wind_angle, coordinate3, cut, tolerance)
	filtered_df_data = filter_dataframe(df_data, wind_angle, coordinate3, cut, tolerance)
	grid1, grid2 = define_grid(filtered_df_data, coordinate1, coordinate2)
	grid_vx_actual, grid_vy_actual, grid_vz_actual, grid_magnitude_actual = get_velocity_grids(grid1, grid2, coordinate1, coordinate2, filtered_df_data)
	grid_vx_pred, grid_vy_pred, grid_vz_pred, grid_magnitude_pred = get_velocity_grids(grid1, grid2, coordinate1, coordinate2, filtered_df)
	geometry_filename = os.path.join(datafolder_path, config["data"]["geometry"])
	geometry1, geometry2 = get_geometry(plane, geometry_filename)
	grid1, grid2 = normalize_grids(plane, grid1, grid2)
	geometry1, geometry2 = normalize_grids(plane, geometry1, geometry2)
	cut, tolerance = normalize_cut_tolerance(plane, cut, tolerance)
	# quiver_points1, quiver_points2, quiver_v1, quiver_v2 = make_quiver_grid(plane, grid1, grid2, grid_vx, grid_vy, grid_vz, stride=100)

	divergence_actual, divergence_flat_actual = get_divergence_from_grid(plane, grid1, grid2, grid_vx_actual, grid_vy_actual, grid_vz_actual)
	divergence_pred, divergence_flat_pred = get_divergence_from_grid(plane, grid1, grid2, grid_vx_pred, grid_vy_pred, grid_vz_pred)

	fig, axs = plt.subplots(1, 2, figsize=(16,8), sharey=True)
	fig.suptitle(f'Comparison of Actual vs. Predicted values with Wind Angle = {wind_angle} in the {plane} Plane with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')
	vmin, vmax, levels, cmap, scatter_size, ticks = plotting_details(divergence_flat_actual)

	contour_actual = axs[0].contourf(grid1, grid2, divergence_actual, levels=128, vmin=vmin, vmax=vmax, cmap=cmap)
	axs[0].scatter(geometry1, geometry2, c='black', s=scatter_size, label='Geometry')
	cbar = fig.colorbar(contour_actual, ax=axs[0])
	cbar.set_label('Div Velocity - Actual', rotation=270, labelpad=15)
	axs[0].set_xlabel(f'{coordinate1} Coordinate')
	axs[0].set_ylabel(f'{coordinate2} Coordinate')
	axs[0].set_xlim(lim_min1, lim_max1) 
	axs[0].set_ylim(lim_min2, lim_max2)
	axs[0].set_title(f'Actual Div V in the {plane} Plane for Wind Angle = {wind_angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')

	contour_pred = axs[1].contourf(grid1, grid2, divergence_pred, levels=128, vmin=vmin, vmax=vmax, cmap=cmap)
	axs[1].scatter(geometry1, geometry2, c='black', s=scatter_size, label='Geometry')
	cbar = fig.colorbar(contour_pred, ax=axs[1])
	cbar.set_label('Div Velocity - Predicted', rotation=270, labelpad=15)
	axs[1].set_xlabel(f'{coordinate1} Coordinate')
	axs[1].set_ylabel(f'{coordinate2} Coordinate')
	axs[1].set_xlim(lim_min1, lim_max1) 
	axs[1].set_ylim(lim_min2, lim_max2)
	axs[1].set_title(f'Predicted Div V in the {plane} Plane for Wind Angle = {wind_angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')

	plt.tight_layout()
	plt.savefig(savename_total_div)
	plt.close()

def plot_torch_predictions(df,df_data,wind_angle,config,datafolder_path,savename_total_div, savename_total_velocity,plane,cut,tolerance):
	coordinate1, coordinate2, coordinate3, lim_min1, lim_max1, lim_min2, lim_max2 = get_plane_config(plane)
	filtered_df = filter_dataframe(df, wind_angle, coordinate3, cut, tolerance)
	filtered_df_data = filter_dataframe(df_data, wind_angle, coordinate3, cut, tolerance)
	grid1, grid2 = define_grid(filtered_df_data, coordinate1, coordinate2)
	grid_vx_actual, grid_vy_actual, grid_vz_actual, grid_magnitude_actual = get_velocity_grids(grid1, grid2, coordinate1, coordinate2, filtered_df_data)
	grid_vx_pred, grid_vy_pred, grid_vz_pred, grid_magnitude_pred = get_velocity_grids(grid1, grid2, coordinate1, coordinate2, filtered_df)
	geometry_filename = os.path.join(datafolder_path, config["data"]["geometry"])
	geometry1, geometry2 = get_geometry(plane, geometry_filename)
	grid1, grid2 = normalize_grids(plane, grid1, grid2)
	geometry1, geometry2 = normalize_grids(plane, geometry1, geometry2)
	cut, tolerance = normalize_cut_tolerance(plane, cut, tolerance)
	divergence_actual, divergence_flat_actual = get_divergence_from_grid(plane, grid1, grid2, grid_vx_actual, grid_vy_actual, grid_vz_actual)
	divergence_pred, divergence_flat_pred = get_divergence_from_torch(filtered_df, plane, coordinate1, coordinate2, grid1, grid2)

	fig, axs = plt.subplots(1, 2, figsize=(16,8), sharey=True)
	fig.suptitle(f'Comparison of Actual vs. Predicted values with Wind Angle = {wind_angle} in the {plane} Plane with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')
	vmin, vmax, levels, cmap, scatter_size, ticks = plotting_details(divergence_flat_actual)

	contour_actual = axs[0].contourf(grid1, grid2, divergence_actual, levels=128, vmin=vmin, vmax=vmax, cmap=cmap)
	axs[0].scatter(geometry1, geometry2, c='black', s=scatter_size, label='Geometry')
	cbar = fig.colorbar(contour_actual, ax=axs[0])
	cbar.set_label('Div Velocity - Actual', rotation=270, labelpad=15)
	axs[0].set_xlabel(f'{coordinate1} Coordinate')
	axs[0].set_ylabel(f'{coordinate2} Coordinate')
	axs[0].set_xlim(lim_min1, lim_max1) 
	axs[0].set_ylim(lim_min2, lim_max2)
	axs[0].set_title(f'Actual Div V in the {plane} Plane for Wind Angle = {wind_angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')

	contour_pred = axs[1].contourf(grid1, grid2, divergence_pred, levels=128, vmin=vmin, vmax=vmax, cmap=cmap)
	axs[1].scatter(geometry1, geometry2, c='black', s=scatter_size, label='Geometry')
	cbar = fig.colorbar(contour_pred, ax=axs[1])
	cbar.set_label('Div Velocity - Predicted', rotation=270, labelpad=15)
	axs[1].set_xlabel(f'{coordinate1} Coordinate')
	axs[1].set_ylabel(f'{coordinate2} Coordinate')
	axs[1].set_xlim(lim_min1, lim_max1) 
	axs[1].set_ylim(lim_min2, lim_max2)
	axs[1].set_title(f'Predicted Div V in the {plane} Plane for Wind Angle = {wind_angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')

	plt.tight_layout()
	plt.savefig(savename_total_div)
	plt.close()

def plot_div_new_angles(df,wind_angle,config,datafolder_path,savename_total_div, savename_total_velocity,plane,cut,tolerance):
    coordinate1, coordinate2, coordinate3, lim_min1, lim_max1, lim_min2, lim_max2 = get_plane_config(plane)
    filtered_df = filter_dataframe(df, wind_angle, coordinate3, cut, tolerance)
    grid1, grid2 = define_grid(filtered_df, coordinate1, coordinate2)
    grid_vx, grid_vy, grid_vz, grid_magnitude = get_velocity_grids(grid1, grid2, coordinate1, coordinate2, filtered_df)
    geometry_filename = os.path.join(datafolder_path, config["data"]["geometry"])
    geometry1, geometry2 = get_geometry(plane, geometry_filename)
    grid1, grid2 = normalize_grids(plane, grid1, grid2)
    geometry1, geometry2 = normalize_grids(plane, geometry1, geometry2)
    cut, tolerance = normalize_cut_tolerance(plane, cut, tolerance)
    quiver_points1, quiver_points2, quiver_v1, quiver_v2 = make_quiver_grid(plane, grid1, grid2, grid_vx, grid_vy, grid_vz, stride=100)
    divergence, divergence_flat = get_divergence_from_grid(plane, grid1, grid2, grid_vx, grid_vy, grid_vz)

    # For Div Velocity
    fig, ax = plt.subplots(figsize=(16, 8), sharey=True)
    vmin, vmax, levels, cmap, scatter_size, ticks = plotting_details(divergence_flat)
    contour = ax.contourf(grid1, grid2, divergence, levels=128, vmin=vmin, vmax=vmax, cmap=cmap)
    # ax.quiver(quiver_points1, quiver_points2, quiver_v1, quiver_v2, scale=200, color='white')
    ax.scatter(geometry1, geometry2, c='black', s=scatter_size, label='Geometry')
    # cbar = fig.colorbar(contour, ax=ax, ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label('Div Velocity', rotation=270, labelpad=15)
    ax.set_xlabel(f'{coordinate1} Coordinate')
    ax.set_ylabel(f'{coordinate2} Coordinate')
    ax.set_xlim(lim_min1, lim_max1) 
    ax.set_ylim(lim_min2, lim_max2)
    ax.set_title(f'Div V in the {plane} Plane for Wind Angle = {wind_angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')
    plt.tight_layout()
    plt.savefig(savename_total_div)
    plt.close()

    # # For Total Velocity
    # fig, ax = plt.subplots(figsize=(16, 8), sharey=True)
    # vmin, vmax, cmap, scatter_size, ticks = plotting_details(filtered_df['Velocity_Magnitude'].values)
    # contour = ax.contourf(grid1, grid2, grid_magnitude, levels=128, vmin=vmin, vmax=vmax, cmap=cmap)
    # ax.scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
    # ax.tricontourf(tri.Triangulation(cylinderfill1, cylinderfill2), np.zeros(len(cylinderfill1)), colors='black', alpha=1)  # Fill color set to black
    # cylinder_patch = patches.Patch(color='black', label='Cylinder Fill')
    # cbar = fig.colorbar(contour, ax=ax, ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
    # cbar.set_label('Velocity Magnitude', rotation=270, labelpad=15)
    # ax.set_xlabel(f'{coordinate1} Coordinate')
    # ax.set_ylabel(f'{coordinate2} Coordinate')
    # ax.set_xlim(lim_min1, lim_max1) 
    # ax.set_ylim(lim_min2, lim_max2)
    # ax.set_title(f'Total Velocity in the {plane} Plane for Wind Angle = {angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')
    # plt.tight_layout()
    # plt.savefig(savename_total_velocity)
    # plt.close()

def plot_div_angles_2d(df,df_data,df_torch,angle,config,plot_folder):
    params = [['X-Z',570,5],['Y-Z',500,5],['X-Y',50,5]]
    chosen_machine_key = config["chosen_machine"]
    datafolder_path = os.path.join(config["machine"][chosen_machine_key], config["data"]["data_folder_name"])
    for j in params:
        plane = j[0]
        cut = j[1]
        tolerance = j[2]
        savename_total_velocity = os.path.join(plot_folder,f'{plane}_totalvelocity_{angle}.png')
        savename_total_div = os.path.join(plot_folder,f'{plane}_divvelocity_{angle}.png')
        savename_total_div_double = os.path.join(plot_folder,f'{plane}_divvelocity_{angle}_double.png')
        savename_total_div_doubletorch = os.path.join(plot_folder,f'{plane}_divvelocity_{angle}_doubletorch.png')
        plot_div_new_angles(df,angle,config,datafolder_path,savename_total_div,savename_total_velocity,plane,cut,tolerance)
        plot_data_predictions(df,df_data,angle,config,datafolder_path,savename_total_div_double, savename_total_velocity,plane,cut,tolerance)
        plot_torch_predictions(df_torch,df_data,angle,config,datafolder_path,savename_total_div_doubletorch, savename_total_velocity,plane,cut,tolerance)

def evaluation_divergence(config, device, output_folder, model_file_path, model, today, overall_start_time):
	testing_type = "Evaluation Divergence"
	chosen_machine_key = config["chosen_machine"]
	datafolder_path = os.path.join(config["machine"][chosen_machine_key], config["data"]["data_folder_name"])
	geometry_filename = os.path.join(datafolder_path, config["data"]["geometry"])
	checkpoint = open_model_file(model_file_path, device)
	div_wind_angles = config["training"]["all_angles"]
	div_plot_folder = os.path.join(output_folder, f'plots_output_for_div_{today}_{checkpoint["epoch"]}')
	os.makedirs(div_plot_folder, exist_ok=True)
	print (f'Model {testing_type} at Epoch = {checkpoint["epoch"]} and Training Completed = {checkpoint["training_completed"]}, Time: {(time.time() - overall_start_time):.2f} Seconds')
	model.load_state_dict(checkpoint['model_state_dict'])
	X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, labels_train_tensor, X_train_tensor_skipped, y_train_tensor_skipped, X_test_tensor_skipped, y_test_tensor_skipped, X_train_tensor_data, y_train_tensor_data, X_test_tensor_data, y_test_tensor_data, feature_scaler, target_scaler = load_data(config, device)
	for wind_angle in div_wind_angles:
		div_wind_angle = [wind_angle]
		X_test_tensor_new = load_data_new_angles(device, config, feature_scaler, target_scaler, div_wind_angle)
		df = evaluate_model(config, model, div_wind_angle, X_test_tensor_new, feature_scaler, target_scaler)
		df_torch = evaluate_model_new(config, model, div_wind_angle, X_test_tensor_new, feature_scaler, target_scaler)
		df_data = load_plotting_data(config, div_wind_angle)
		print (f'Model Evaluated and Starting to Plot for Wind Angle = {wind_angle}, Time: {(time.time() - overall_start_time):.2f} Seconds')
		plot_div_angles_2d(df,df_data,df_torch,wind_angle,config,div_plot_folder)
		print (f'Plotting Done for Wind Angle = {wind_angle}, Time: {(time.time() - overall_start_time):.2f} Seconds')