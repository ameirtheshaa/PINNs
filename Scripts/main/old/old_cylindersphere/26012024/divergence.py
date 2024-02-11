from definitions import *
from PINN import *
from plotting_definitions import *

def load_data_new_angles(device, config, feature_scaler, target_scaler, wind_angles=None):
    chosen_machine_key = config["chosen_machine"]
    if wind_angles is None:
        wind_angles = config["train_test"]["new_angles"]
    datafolder_path = os.path.join(config["machine"][chosen_machine_key], config["data"]["data_folder_name"])
    filenames = get_filenames_from_folder(datafolder_path, config["data"]["extension"], config["data"]["startname_data"])
    dfs = []
    for wind_angle in wind_angles:
        for filename in sorted(filenames):
        	wind_dir = int(filename.split('_')[-1].split('.')[0])
        	if wind_angle == wind_dir:
	            df = pd.read_csv(os.path.join(datafolder_path, filename))
	            df['cos(WindAngle)'] = (np.cos(np.deg2rad(wind_angle)))
	            df['sin(WindAngle)'] = (np.sin(np.deg2rad(wind_angle)))
	            dfs.append(df)
    data = pd.concat(dfs)
    features = data[config["training"]["input_params"]]
    targets = data[config["training"]["output_params"]]
    normalized_features, normalized_targets = transform_data_with_scalers(features, targets, feature_scaler, target_scaler)
    X_train, X_test, y_train, y_test = train_test_split(normalized_features, normalized_targets,test_size=config["train_test"]["new_angles_test_size"], random_state=config["train_test"]["random_state"])
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    X_test_tensor = X_test_tensor.to(device)
    return X_test_tensor

def concatenate_data_files(filenames, datafolder_path, wind_angles):
	dfs = []
	for filename in sorted(filenames):
		df = pd.read_csv(os.path.join(datafolder_path, filename))
		wind_angle = int(filename.split('_')[-1].split('.')[0])
		df['cos(WindAngle)'] = (np.cos(np.deg2rad(wind_angle)))
		df['sin(WindAngle)'] = (np.sin(np.deg2rad(wind_angle)))
		if wind_angle in wind_angles:
			df['WindAngle'] = (wind_angle)
			dfs.append(df)
	data = pd.concat(dfs)
	return data

def get_gradients(model, X, input_params, output_params):
    extracted_inputs = model.extract_input_parameters(X, input_params)
    input_dict = {param: value for param, value in zip(input_params, extracted_inputs)}   
    x = input_dict.get('Points:0')
    y = input_dict.get('Points:1')
    z = input_dict.get('Points:2')
    cos_wind_angle = input_dict.get('cos(WindAngle)')
    sin_wind_angle = input_dict.get('sin(WindAngle)')  
    u_x, v_y, w_z = model.compute_divergence(cos_wind_angle, sin_wind_angle, x, y, z, output_params)
    return u_x, v_y, w_z

def get_cont(model, X, input_params, output_params):
    extracted_inputs = model.extract_input_parameters(X, input_params)
    input_dict = {param: value for param, value in zip(input_params, extracted_inputs)}   
    x = input_dict.get('Points:0')
    y = input_dict.get('Points:1')
    z = input_dict.get('Points:2')
    cos_wind_angle = input_dict.get('cos(WindAngle)')
    sin_wind_angle = input_dict.get('sin(WindAngle)')  
    cont = model.continuity(cos_wind_angle, sin_wind_angle, x, y, z, output_params)
    return cont

def evaluate_model_new(config, model, wind_angles, X_test_tensor, feature_scaler, target_scaler, output_folder):
	model.eval()
	dfs = []
	input_params = config["training"]["input_params"]
	output_params = config["training"]["output_params"]
	u_x, v_y, w_z = get_gradients(model, X_test_tensor, input_params, output_params)
	cont = get_cont(model, X_test_tensor, input_params, output_params)
	extracted_stds = extract_stds(target_scaler, output_params)
	extracted_input_stds = extract_input_stds(feature_scaler, input_params)
	div_dict = {
	"u_x": u_x.cpu().detach().numpy().flatten()*extracted_stds[0]/extracted_input_stds[0],
	"v_y": v_y.cpu().detach().numpy().flatten()*extracted_stds[1]/extracted_input_stds[1],
	"w_z": w_z.cpu().detach().numpy().flatten()*extracted_stds[2]/extracted_input_stds[2],
	"Divergence": u_x.cpu().detach().numpy().flatten()*extracted_stds[0]/extracted_input_stds[0] + v_y.cpu().detach().numpy().flatten()*extracted_stds[1]/extracted_input_stds[1] + w_z.cpu().detach().numpy().flatten()*extracted_stds[2]/extracted_input_stds[2]
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
				combined_df.to_csv(os.path.join(output_folder, f'divergence_output_{wind_angle}.csv'), index=False)
				dfs.append(combined_df)
	data = pd.concat(dfs)
	return data

def get_divergence(df, plane, coordinate1, coordinate2, grid1, grid2):
	divergence = griddata(df[[coordinate1, coordinate2]].values, df['Divergence'].values, (grid1, grid2), method='linear', fill_value=0)
	return divergence

def load_line_div_data(device, config, wind_angles, feature_scaler, target_scaler):
    chosen_machine_key = config["chosen_machine"]
    datafolder_path = os.path.join(config["machine"][chosen_machine_key], config["data"]["data_folder_name"])
    filenames = get_filenames_from_folder(datafolder_path, config["data"]["extension"], "gradV_line_")
    data = concatenate_data_files(filenames, datafolder_path, wind_angles)
    data.rename(columns={'gradV_0': 'u_x', 'gradV_1': 'u_y', 'gradV_2': 'u_z', 'gradV_3': 'v_x', 'gradV_4': 'v_y', 'gradV_5': 'v_z', 'gradV_6': 'w_x', 'gradV_7': 'w_y', 'gradV_7': 'w_z' }, inplace=True)
    data.rename(columns={'Points_0': 'Points:0', 'Points_1': 'Points:1', 'Points_2': 'Points:2', 'Velocity_0': 'Velocity:0', 'Velocity_1': 'Velocity:1', 'Velocity_2': 'Velocity:2'}, inplace=True)
    features = data[config["training"]["input_params"]]
    targets = data[config["training"]["output_params"]]
    normalized_features, normalized_targets = transform_data_with_scalers(features, targets, feature_scaler, target_scaler)
    X_train, X_test, y_train, y_test = train_test_split(normalized_features, normalized_targets,test_size=0.99, random_state=config["train_test"]["random_state"])
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    X_test_tensor = X_test_tensor.to(device)
    return data, X_test_tensor

def symlog(x):
    x = np.array(x)
    symlog_x = np.where(x != 0, np.sign(x) * np.log1p(np.abs(x) - 1), 0)
    return symlog_x

def base_line_plots(wind_angle, coord_values_actual, coord_values_pred, grad_actual, velocity_actual, grad_pred, velocity_pred, savename_total_div, label_grad, label_velocity):
	fig, axs = plt.subplots(1, 2, figsize=(16,8), sharey=True)
	fig.suptitle(f'Comparison of Actual vs. Predicted values with Wind Angle = {wind_angle}')
	
	# axs[0].scatter(coord_values_actual, grad_actual, c='black', s=1, label=f'{label_grad} actual')
	# axs[0].scatter(coord_values_actual, velocity_actual, c='red', s=1, label=f'{label_velocity} actual')
	axs[0].plot(coord_values_actual, grad_actual, c='black', label=f'{label_grad} actual')
	axs[0].plot(coord_values_actual, velocity_actual, c='red', label=f'{label_velocity} actual')
	axs[0].set_xlabel(f'Y Coordinate')
	axs[0].set_ylabel(f'Value')
	axs[0].set_title(f'Actual {label_velocity} and {label_grad} for Wind Angle = {wind_angle} with a cut at Z=50m and X=480m')
	axs[0].legend()

	# axs[1].scatter(coord_values_pred, grad_pred, c='black', s=1, label=f'{label_grad} predicted')
	# axs[1].scatter(coord_values_pred, velocity_pred, c='red', s=1, label=f'{label_velocity} predicted')
	axs[1].plot(coord_values_pred, grad_pred, c='black', label=f'{label_grad} predicted')
	axs[1].plot(coord_values_pred, velocity_pred, c='red', label=f'{label_velocity} predicted')
	axs[1].set_xlabel(f'Y Coordinate')
	axs[1].set_ylabel(f'Value')
	axs[1].set_title(f'Predicted {label_velocity} and {label_grad} for Wind Angle = {wind_angle} with a cut at Z=50m and X=480m')
	axs[1].legend()

	plt.tight_layout()
	plt.savefig(savename_total_div+f'div_{label_grad}.png')
	plt.close()

def base_line_plots_modf(wind_angle, coord_values_actual, coord_values_pred, grad_actual, velocity_actual, grad_pred, velocity_pred, savename_total_div, label_grad, label_velocity):
	fig, axs = plt.subplots(1, 2, figsize=(16,8), sharey=False)
	fig.suptitle(f'Comparison of Actual vs. Predicted values with Wind Angle = {wind_angle}')
	
	# axs[0].scatter(coord_values_actual, grad_actual, c='black', s=1, label=f'{label_grad} actual')
	# axs[0].scatter(coord_values_actual, velocity_actual, c='red', s=1, label=f'{label_velocity} actual')
	axs[0].plot(coord_values_actual, velocity_actual, c='red', label=f'{label_velocity} actual')
	axs[0].plot(coord_values_pred, velocity_pred, c='black', label=f'{label_velocity} predicted')
	axs[0].set_xlabel(f'Y Coordinate')
	axs[0].set_ylabel(f'Value')
	axs[0].set_title(f'Actual and Predicted {label_velocity} for Wind Angle = {wind_angle} with a cut at Z=50m and X=480m')
	axs[0].legend()

	# axs[1].scatter(coord_values_pred, grad_pred, c='black', s=1, label=f'{label_grad} predicted')
	# axs[1].scatter(coord_values_pred, velocity_pred, c='red', s=1, label=f'{label_velocity} predicted')
	axs[1].plot(coord_values_actual, grad_actual, c='red', label=f'{label_grad} actual')
	axs[1].plot(coord_values_pred, grad_pred, c='black', label=f'{label_grad} predicted')
	axs[1].set_xlabel(f'Y Coordinate')
	axs[1].set_ylabel(f'Value')
	axs[1].set_title(f'Actual and Predicted {label_grad} for Wind Angle = {wind_angle} with a cut at Z=50m and X=480m')
	axs[1].legend()

	plt.tight_layout()
	plt.savefig(savename_total_div+f'div_{label_grad}.png')
	plt.close()

def base_line_plots_modf_log(wind_angle, coord_values_actual, coord_values_pred, grad_actual, velocity_actual, grad_pred, velocity_pred, savename_total_div, label_grad, label_velocity):
	fig, axs = plt.subplots(1, 2, figsize=(16,8), sharey=False)
	fig.suptitle(f'Comparison of Actual vs. Predicted values with Wind Angle = {wind_angle}')
	
	# axs[0].scatter(coord_values_actual, grad_actual, c='black', s=1, label=f'{label_grad} actual')
	# axs[0].scatter(coord_values_actual, velocity_actual, c='red', s=1, label=f'{label_velocity} actual')
	axs[0].plot(coord_values_actual, velocity_actual, c='red', label=f'{label_velocity} actual')
	axs[0].plot(coord_values_pred, velocity_pred, c='black', label=f'{label_velocity} predicted')
	axs[0].set_xlabel(f'Y Coordinate')
	axs[0].set_ylabel(f'Value')
	axs[0].set_title(f'Actual and Predicted {label_velocity} for Wind Angle = {wind_angle} with a cut at Z=50m and X=480m')
	axs[0].legend()

	# axs[1].scatter(coord_values_pred, grad_pred, c='black', s=1, label=f'{label_grad} predicted')
	# axs[1].scatter(coord_values_pred, velocity_pred, c='red', s=1, label=f'{label_velocity} predicted')
	axs[1].plot(coord_values_actual, symlog(grad_actual), c='red', label=f'{label_grad} actual')
	axs[1].plot(coord_values_pred, symlog(grad_pred), c='black', label=f'{label_grad} predicted')
	axs[1].set_xlabel(f'Y Coordinate')
	axs[1].set_ylabel(f'Value')
	axs[1].set_title(f'Actual and Predicted (Symmetric Log) {label_grad} for Wind Angle = {wind_angle} with a cut at Z=50m and X=480m')
	axs[1].legend()

	plt.tight_layout()
	plt.savefig(savename_total_div+f'div_{label_grad}.png')
	plt.close()

def plot_line_predictions(df_data,df,wind_angle,savename_total_div):
	
	df_data.rename(columns={'Points:0': 'X', 'Points:1': 'Y', 'Points:2': 'Z', 'Velocity:0': 'Velocity_X', 'Velocity:1': 'Velocity_Y', 'Velocity:2': 'Velocity_Z'}, inplace=True)
	dvx_dx_actual = df_data['u_x'].values
	dvy_dy_actual = df_data['v_y'].values
	dvz_dz_actual = df_data['w_z'].values
	div_actual = df_data['Divergence'].values
	coord_values_actual = df_data['Y'].values
	u_actual = df_data['Velocity_X'].values
	v_actual = df_data['Velocity_Y'].values
	w_actual = df_data['Velocity_Z'].values
	velocity_actual = df_data['Velocity_Magnitude'].values

	df = df.sort_values(by='Y')
	dvx_dx_pred = df['u_x'].values
	dvy_dy_pred = df['v_y'].values
	dvz_dz_pred = df['w_z'].values
	div_pred = df['Divergence'].values
	coord_values_pred = df['Y'].values
	u_pred = df['Velocity_X'].values
	v_pred = df['Velocity_Y'].values
	w_pred = df['Velocity_Z'].values
	velocity_pred = df['Velocity_Magnitude'].values

	# base_line_plots(wind_angle, coord_values_actual, coord_values_pred, dvx_dx_actual, u_actual, dvx_dx_pred, u_pred, savename_total_div, label_grad='u_x', label_velocity='u')
	# base_line_plots(wind_angle, coord_values_actual, coord_values_pred, dvy_dy_actual, v_actual, dvy_dy_pred, v_pred, savename_total_div, label_grad='v_y', label_velocity='v')
	# base_line_plots(wind_angle, coord_values_actual, coord_values_pred, dvz_dz_actual, w_actual, dvz_dz_pred, w_pred, savename_total_div, label_grad='w_z', label_velocity='w')
	# base_line_plots(wind_angle, coord_values_actual, coord_values_pred, div_actual, velocity_actual, div_pred, velocity_pred, savename_total_div, label_grad='div', label_velocity='V')

	base_line_plots_modf(wind_angle, coord_values_actual, coord_values_pred, dvx_dx_actual, u_actual, dvx_dx_pred, u_pred, savename_total_div, label_grad='u_x', label_velocity='u')
	base_line_plots_modf(wind_angle, coord_values_actual, coord_values_pred, dvy_dy_actual, v_actual, dvy_dy_pred, v_pred, savename_total_div, label_grad='v_y', label_velocity='v')
	base_line_plots_modf(wind_angle, coord_values_actual, coord_values_pred, dvz_dz_actual, w_actual, dvz_dz_pred, w_pred, savename_total_div, label_grad='w_z', label_velocity='w')
	base_line_plots_modf(wind_angle, coord_values_actual, coord_values_pred, div_actual, velocity_actual, div_pred, velocity_pred, savename_total_div, label_grad='div', label_velocity='V')

	# base_line_plots_modf_log(wind_angle, coord_values_actual, coord_values_pred, dvx_dx_actual, u_actual, dvx_dx_pred, u_pred, savename_total_div, label_grad='u_x', label_velocity='u')
	# base_line_plots_modf_log(wind_angle, coord_values_actual, coord_values_pred, dvy_dy_actual, v_actual, dvy_dy_pred, v_pred, savename_total_div, label_grad='v_y', label_velocity='v')
	# base_line_plots_modf_log(wind_angle, coord_values_actual, coord_values_pred, dvz_dz_actual, w_actual, dvz_dz_pred, w_pred, savename_total_div, label_grad='w_z', label_velocity='w')
	# base_line_plots_modf_log(wind_angle, coord_values_actual, coord_values_pred, div_actual, velocity_actual, div_pred, velocity_pred, savename_total_div, label_grad='div', label_velocity='V')

def plot_torch_predictions(df,df_data,wind_angle,config,datafolder_path,savename_total_div,plane,cut,tolerance):
	coordinate1, coordinate2, coordinate3, lim_min1, lim_max1, lim_min2, lim_max2 = get_plane_config(plane)
	filtered_df = filter_dataframe(df, wind_angle, coordinate3, cut, tolerance)
	filtered_df_data = filter_dataframe(df_data, wind_angle, coordinate3, cut, tolerance)
	# grid1, grid2 = define_grid(filtered_df_data, coordinate1, coordinate2)
	grid1_actual, grid2_actual = define_scatter_grid(filtered_df_data, coordinate1, coordinate2)
	grid1_pred, grid2_pred = define_scatter_grid(filtered_df, coordinate1, coordinate2)
	geometry_filename = os.path.join(datafolder_path, config["data"]["geometry"])
	geometry1, geometry2 = get_geometry(plane, geometry_filename)
	grid1_actual, grid2_actual = normalize_grids(plane, grid1_actual, grid2_actual)
	grid1_pred, grid2_pred = normalize_grids(plane, grid1_pred, grid2_pred)
	geometry1, geometry2 = normalize_grids(plane, geometry1, geometry2)
	cut, tolerance = normalize_cut_tolerance(plane, cut, tolerance)
	# divergence_actual = get_divergence(filtered_df_data, plane, coordinate1, coordinate2, grid1, grid2)
	# divergence_pred = get_divergence(filtered_df, plane, coordinate1, coordinate2, grid1, grid2)
	divergence_actual = filtered_df_data['Divergence']
	divergence_pred = filtered_df['Divergence']

	fig, axs = plt.subplots(1, 2, figsize=(32,16), sharey=True)
	fig.suptitle(f'Comparison of Actual vs. Predicted values with Wind Angle = {wind_angle} in the {plane} Plane with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')
	vmin, vmax, levels, cmap, scatter_size, ticks = plotting_details(divergence_actual)
	scatter_actual = axs[0].scatter(grid1_actual, grid2_actual, c=divergence_actual, s=scatter_size, vmin=np.min(divergence_actual), vmax=np.max(divergence_actual), cmap=cmap)
	scatter_cbar_actual = fig.colorbar(scatter_actual, ax=axs[0])
	scatter_cbar_actual.set_label('Div Velocity - Actual', rotation=270, labelpad=15)
	# contour_actual = axs[0].contourf(grid1, grid2, divergence_actual, levels=levels, vmin=vmin, vmax=vmax, cmap=cmap)
	# contour_actual = axs[0].contourf(grid1, grid2, divergence_actual, cmap=cmap)
	# cbar = fig.colorbar(contour_actual, ax=axs[0], ticks=ticks)
	# cbar.set_label('Div Velocity - Actual', rotation=270, labelpad=15)
	axs[0].scatter(geometry1, geometry2, c='black', s=scatter_size, label='Geometry')
	axs[0].set_xlabel(f'{coordinate1} Coordinate')
	axs[0].set_ylabel(f'{coordinate2} Coordinate')
	axs[0].set_xlim(lim_min1, lim_max1) 
	axs[0].set_ylim(lim_min2, lim_max2)
	axs[0].set_title(f'Actual Div V in the {plane} Plane for Wind Angle = {wind_angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f} with mean = {np.mean(divergence_actual):.2f} and std dev = {np.std(divergence_actual):.2f}')

	vmin, vmax, levels, cmap, scatter_size, ticks = plotting_details(divergence_pred)
	scatter_pred = axs[1].scatter(grid1_pred, grid2_pred, c=divergence_pred, s=scatter_size, vmin=np.min(divergence_pred), vmax=np.max(divergence_pred), cmap=cmap)
	scatter_cbar_pred = fig.colorbar(scatter_pred, ax=axs[1])
	scatter_cbar_pred.set_label('Div Velocity - Predicted', rotation=270, labelpad=15)
	# contour_pred = axs[1].contourf(grid1, grid2, divergence_pred, levels=levels, vmin=vmin, vmax=vmax, cmap=cmap)
	# contour_pred = axs[1].contourf(grid1, grid2, divergence_pred, cmap=cmap)
	# cbar = fig.colorbar(contour_pred, ax=axs[1], ticks=ticks)
	# cbar = fig.colorbar(contour_actual, ax=axs[1])
	# cbar.set_label('Div Velocity - Predicted', rotation=270, labelpad=15)
	axs[1].scatter(geometry1, geometry2, c='black', s=scatter_size, label='Geometry')
	axs[1].set_xlabel(f'{coordinate1} Coordinate')
	axs[1].set_ylabel(f'{coordinate2} Coordinate')
	axs[1].set_xlim(lim_min1, lim_max1) 
	axs[1].set_ylim(lim_min2, lim_max2)
	axs[1].set_title(f'Predicted Div V in the {plane} Plane for Wind Angle = {wind_angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f} with mean = {np.mean(divergence_pred):.2f} and std dev = {np.std(divergence_pred):.2f}')

	plt.tight_layout()
	plt.savefig(savename_total_div)
	plt.close()

def plot_div_angles_2d(df_data,df_torch,angle,config,plot_folder):
    params = config["plotting"]["plotting_params"]
    chosen_machine_key = config["chosen_machine"]
    datafolder_path = os.path.join(config["machine"][chosen_machine_key], config["data"]["data_folder_name"])
    for j in params:
        plane = j[0]
        cut = j[1]
        tolerance = j[2]
        savename_total_div_doubletorch = os.path.join(plot_folder,f'{plane}_divvelocity_{angle}.png')
        plot_torch_predictions(df_torch,df_data,angle,config,datafolder_path,savename_total_div_doubletorch,plane,cut,tolerance)

def evaluation_divergence(model, device, config, data_dict, model_file_path, output_folder, today, overall_start_time):
	testing_type = "Evaluation Divergence"
	chosen_machine_key = config["chosen_machine"]
	datafolder_path = os.path.join(config["machine"][chosen_machine_key], config["data"]["data_folder_name"])
	geometry_filename = os.path.join(datafolder_path, config["data"]["geometry"])
	checkpoint = open_model_file(model_file_path, device)
	div_wind_angles = config["training"]["all_angles"]
	div_plot_folder = os.path.join(output_folder, f'plots_output_for_div_{today}_{checkpoint["epoch"]}')
	os.makedirs(div_plot_folder, exist_ok=True)
	div_line_plot_folder = os.path.join(output_folder, f'plots_output_for_div_line_{today}_{checkpoint["epoch"]}')
	os.makedirs(div_line_plot_folder, exist_ok=True)
	print (f'Model {testing_type} at Epoch = {checkpoint["epoch"]} and Training Completed = {checkpoint["training_completed"]}, Time: {(time.time() - overall_start_time):.2f} Seconds')
	model.load_state_dict(checkpoint['model_state_dict'])
	X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, labels_train_tensor, X_train_tensor_skipped, y_train_tensor_skipped, X_test_tensor_skipped, y_test_tensor_skipped, feature_scaler, target_scaler, X_train_divergences_tensor, X_test_divergences_tensor, y_train_divergences_tensor, y_test_divergences_tensor = data_dict.values()
	for wind_angle in div_wind_angles:
		div_wind_angle = [wind_angle]
		df_data_line, X_test_tensor_line = load_line_div_data(device, config, div_wind_angle, feature_scaler, target_scaler)
		df_torch_line = evaluate_model_new(config, model, div_wind_angle, X_test_tensor_line, feature_scaler, target_scaler, div_line_plot_folder)
		X_test_tensor_new = load_data_new_angles(device, config, feature_scaler, target_scaler, div_wind_angle)
		df_torch = evaluate_model_new(config, model, div_wind_angle, X_test_tensor_new, feature_scaler, target_scaler,div_plot_folder)
		df_data = load_div_data(config, div_wind_angle)

		print (f'Model Evaluated and Starting to Plot for Wind Angle = {wind_angle}, Time: {(time.time() - overall_start_time):.2f} Seconds')
		savename_total_div_doubletorch_line = os.path.join(div_line_plot_folder,f'div_line_{wind_angle}')
		plot_line_predictions(df_data_line,df_torch_line,wind_angle,savename_total_div_doubletorch_line)
		plot_div_angles_2d(df_data,df_torch,wind_angle,config,div_plot_folder)
		print (f'Plotting Done for Wind Angle = {wind_angle}, Time: {(time.time() - overall_start_time):.2f} Seconds')