from definitions import *
from PINN import *
from plotting_definitions import *

def evaluate_RANS_data(df, rho, nu):
	df['nu_eff'] = df['TurbVisc'] + nu
	df['f'] = (df['u'] * df['u_x'] + df['v'] * df['u_y'] + df['w'] * df['u_z'] - (1 / rho) * df['p_x'] + df['nu_eff'] * (2 * df['u_xx']) + df['nu_eff'] * (df['u_yy'] + df['v_xy']) + df['nu_eff'] * (df['u_zz'] + df['w_xz']))
	df['g'] = (df['u'] * df['v_x'] + df['v'] * df['v_y'] + df['w'] * df['v_z'] - (1 / rho) * df['p_y'] + df['nu_eff'] * (df['v_xx'] + df['u_xy']) + df['nu_eff'] * (2 * df['v_yy']) + df['nu_eff'] * (df['v_zz'] + df['w_yz']))
	df['h'] = (df['u'] * df['w_x'] + df['v'] * df['w_y'] + df['w'] * df['w_z'] - (1 / rho) * df['p_z'] + df['nu_eff'] * (df['w_xx'] + df['u_xz']) + df['nu_eff'] * (df['w_yy'] + df['v_yz']) + df['nu_eff'] * (2 * df['w_zz']))
	df['RANS'] = df['f'] + df['g'] + df['h']
	return df

def evaluate_div_data(df):
	df['Div'] = df['u_x'] + df['v_y'] * df['w_z']
	return df

def scale_derivatives_NN(config, feature_scaler, target_scaler, der_dict):
	input_params_modf = config["training"]["input_params_modf"]
	output_params_modf = config["training"]["output_params_modf"]
	input_stds_means = extract_stds_means(feature_scaler, input_params_modf)
	output_stds_means = extract_stds_means(target_scaler, output_params_modf)
	stds_means_dict = {**input_stds_means, **output_stds_means}

	der_dict["u_x"] = der_dict["u_x"]*(stds_means_dict['Velocity_X_std']/stds_means_dict['X_std'])
	der_dict["u_y"] = der_dict["u_y"]*(stds_means_dict['Velocity_X_std']/stds_means_dict['Y_std'])
	der_dict["u_z"] = der_dict["u_z"]*(stds_means_dict['Velocity_X_std']/stds_means_dict['Z_std'])

	der_dict["u_xx"] = der_dict["u_xx"]*(stds_means_dict['Velocity_X_std']/(stds_means_dict['X_std']*stds_means_dict['X_std']))
	der_dict["u_yy"] = der_dict["u_yy"]*(stds_means_dict['Velocity_X_std']/(stds_means_dict['Y_std']*stds_means_dict['Y_std']))
	der_dict["u_zz"] = der_dict["u_zz"]*(stds_means_dict['Velocity_X_std']/(stds_means_dict['Z_std']*stds_means_dict['Z_std']))

	der_dict["v_x"] = der_dict["v_x"]*(stds_means_dict['Velocity_Y_std']/stds_means_dict['X_std'])
	der_dict["v_y"] = der_dict["v_y"]*(stds_means_dict['Velocity_Y_std']/stds_means_dict['Y_std'])
	der_dict["v_z"] = der_dict["v_z"]*(stds_means_dict['Velocity_Y_std']/stds_means_dict['Z_std'])

	der_dict["v_xx"] = der_dict["v_xx"]*(stds_means_dict['Velocity_Y_std']/(stds_means_dict['X_std']*stds_means_dict['X_std']))
	der_dict["v_yy"] = der_dict["v_yy"]*(stds_means_dict['Velocity_Y_std']/(stds_means_dict['Y_std']*stds_means_dict['Y_std']))
	der_dict["v_zz"] = der_dict["v_zz"]*(stds_means_dict['Velocity_Y_std']/(stds_means_dict['Z_std']*stds_means_dict['Z_std']))

	der_dict["w_x"] = der_dict["w_x"]*(stds_means_dict['Velocity_Z_std']/stds_means_dict['X_std'])
	der_dict["w_y"] = der_dict["w_y"]*(stds_means_dict['Velocity_Z_std']/stds_means_dict['Y_std'])
	der_dict["w_z"] = der_dict["w_z"]*(stds_means_dict['Velocity_Z_std']/stds_means_dict['Z_std'])

	der_dict["w_xx"] = der_dict["w_xx"]*(stds_means_dict['Velocity_Z_std']/(stds_means_dict['X_std']*stds_means_dict['X_std']))
	der_dict["w_yy"] = der_dict["w_yy"]*(stds_means_dict['Velocity_Z_std']/(stds_means_dict['Y_std']*stds_means_dict['Y_std']))
	der_dict["w_zz"] = der_dict["w_zz"]*(stds_means_dict['Velocity_Z_std']/(stds_means_dict['Z_std']*stds_means_dict['Z_std']))

	der_dict["u_xy"] = der_dict["u_xy"]*(stds_means_dict['Velocity_X_std']/(stds_means_dict['X_std']*stds_means_dict['Y_std']))
	der_dict["u_xz"] = der_dict["u_xz"]*(stds_means_dict['Velocity_X_std']/(stds_means_dict['X_std']*stds_means_dict['Z_std']))	

	der_dict["v_xy"] = der_dict["v_xy"]*(stds_means_dict['Velocity_Y_std']/(stds_means_dict['X_std']*stds_means_dict['Y_std']))
	der_dict["v_yz"] = der_dict["v_yz"]*(stds_means_dict['Velocity_Y_std']/(stds_means_dict['Y_std']*stds_means_dict['Z_std']))

	der_dict["w_xz"] = der_dict["w_xz"]*(stds_means_dict['Velocity_Z_std']/(stds_means_dict['X_std']*stds_means_dict['Z_std']))
	der_dict["w_yz"] = der_dict["w_yz"]*(stds_means_dict['Velocity_Z_std']/(stds_means_dict['Y_std']*stds_means_dict['Z_std']))

	der_dict["p_x"] = der_dict["p_x"]*(stds_means_dict['Pressure_std']/stds_means_dict['X_std'])
	der_dict["p_y"] = der_dict["p_y"]*(stds_means_dict['Pressure_std']/stds_means_dict['Y_std'])
	der_dict["p_z"] = der_dict["p_z"]*(stds_means_dict['Pressure_std']/stds_means_dict['Z_std'])

	der_dict["u"] = der_dict["u"]*stds_means_dict['Velocity_X_std'] + stds_means_dict['Velocity_X_mean']
	der_dict["v"] = der_dict["v"]*stds_means_dict['Velocity_Y_std'] + stds_means_dict['Velocity_Y_mean']
	der_dict["w"] = der_dict["w"]*stds_means_dict['Velocity_Z_std'] + stds_means_dict['Velocity_Z_mean']
	der_dict["p"] = der_dict["p"]*stds_means_dict['Pressure_std'] + stds_means_dict['Pressure_mean']
	der_dict["TurbVisc"] = der_dict["TurbVisc"]*stds_means_dict['TurbVisc_std'] + stds_means_dict['TurbVisc_mean']

	return der_dict

def load_derivatives_NN(config, model, X, feature_scaler, target_scaler):
	input_params = config["training"]["input_params"]
	output_params = config["training"]["output_params"]
		
	extracted_inputs = model.extract_input_parameters(X, input_params)
	input_dict = {param: value for param, value in zip(input_params, extracted_inputs)}   
	x = input_dict.get('Points:0')
	y = input_dict.get('Points:1')
	z = input_dict.get('Points:2')
	cos_wind_angle = input_dict.get('cos(WindAngle)')
	sin_wind_angle = input_dict.get('sin(WindAngle)')

	derivatives_dict = model.compute_derivatives(cos_wind_angle, sin_wind_angle, x, y, z, output_params)
	derivatives_dict = scale_derivatives_NN(config, feature_scaler, target_scaler, derivatives_dict)

	derivatives_df = pd.DataFrame(derivatives_dict)

	return derivatives_df

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

def base_line_plots(wind_angle, coord_values_actual, coord_values_pred, grad_actual, velocity_actual, grad_pred, velocity_pred, savename_total_div, label_grad, label_velocity):
	fig, axs = plt.subplots(1, 2, figsize=(32,16), sharey=False)
	fig.suptitle(f'Comparison of Actual vs. Predicted values with Wind Angle = {wind_angle}')

	axs[0].plot(coord_values_actual, velocity_actual, c='red', label=f'{label_velocity} actual')
	axs[0].plot(coord_values_pred, velocity_pred, c='black', label=f'{label_velocity} predicted')
	axs[0].set_xlabel(f'Y Coordinate')
	axs[0].set_ylabel(f'Value')
	axs[0].set_title(f'Actual and Predicted {label_velocity} for Wind Angle = {wind_angle} with a cut at Z=50m and X=480m \n Actual {label_velocity} Mean = {np.mean(velocity_actual):.2f} and Standard Deviation = {np.std(velocity_actual):.2f} \n Predicted {label_velocity} Mean = {np.mean(velocity_pred):.2f} and Standard Deviation = {np.std(velocity_pred):.2f}')
	axs[0].legend()

	axs[1].plot(coord_values_actual, grad_actual, c='red', label=f'{label_grad} actual')
	axs[1].plot(coord_values_pred, grad_pred, c='black', label=f'{label_grad} predicted')
	axs[1].set_xlabel(f'Y Coordinate')
	axs[1].set_ylabel(f'Value')
	axs[1].set_title(f'Actual and Predicted {label_grad} for Wind Angle = {wind_angle} with a cut at Z=50m and X=480m\n Actual {label_grad} Mean = {np.mean(grad_actual):.2f} and Standard Deviation = {np.std(grad_actual):.2f} \n Predicted {label_grad} Mean = {np.mean(grad_pred):.2f} and Standard Deviation = {np.std(grad_pred):.2f}')
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
	div_pred = df['Div'].values
	coord_values_pred = df['Y'].values
	u_pred = df['Velocity_X'].values
	v_pred = df['Velocity_Y'].values
	w_pred = df['Velocity_Z'].values
	velocity_pred = df['Velocity_Magnitude'].values

	base_line_plots(wind_angle, coord_values_actual, coord_values_pred, dvx_dx_actual, u_actual, dvx_dx_pred, u_pred, savename_total_div, label_grad='u_x', label_velocity='u')
	base_line_plots(wind_angle, coord_values_actual, coord_values_pred, dvy_dy_actual, v_actual, dvy_dy_pred, v_pred, savename_total_div, label_grad='v_y', label_velocity='v')
	base_line_plots(wind_angle, coord_values_actual, coord_values_pred, dvz_dz_actual, w_actual, dvz_dz_pred, w_pred, savename_total_div, label_grad='w_z', label_velocity='w')
	base_line_plots(wind_angle, coord_values_actual, coord_values_pred, div_actual, velocity_actual, div_pred, velocity_pred, savename_total_div, label_grad='div', label_velocity='V')

def evaluate_model_physics(config, model, wind_angles, X_test_tensor, feature_scaler, target_scaler, output_folder, physics):
	model.eval()
	dfs = []
	df = load_derivatives_NN(config, model, X_test_tensor, feature_scaler, target_scaler)
	if physics == 'RANS':
		rho = config["data"]["density"]
		nu = config["data"]["kinematic_viscosity"]
		df = evaluate_RANS_data(df, rho, nu)
	elif physics == 'Div':
		df = evaluate_div_data(df)
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
			filtered_df = df.loc[mask]
			if len(filtered_predictions)!= 0:
				combined_df = pd.concat([filtered_X_test_dataframe, filtered_predictions, filtered_df], axis=1)
				combined_df['WindAngle'] = (wind_angle)
				combined_df.to_csv(os.path.join(output_folder, f'{physics}_output_{wind_angle}.csv'), index=False)
				dfs.append(combined_df)
	data = pd.concat(dfs)
	return data

def plot_physics_predictions(df,df_data,wind_angle,config,datafolder_path,savename,plane,cut,tolerance, physics):
	coordinate1, coordinate2, coordinate3, lim_min1, lim_max1, lim_min2, lim_max2 = get_plane_config(plane)
	filtered_df = filter_dataframe(df, wind_angle, coordinate3, cut, tolerance)
	filtered_df_data = filter_dataframe(df_data, wind_angle, coordinate3, cut, tolerance)
	grid1, grid2 = define_grid(filtered_df_data, coordinate1, coordinate2)
	grid1_actual, grid2_actual = define_scatter_grid(filtered_df_data, coordinate1, coordinate2)
	grid1_pred, grid2_pred = define_scatter_grid(filtered_df, coordinate1, coordinate2)
	geometry_filename = os.path.join(datafolder_path, config["data"]["geometry"])
	geometry1, geometry2 = get_geometry(plane, geometry_filename)
	grid1_actual, grid2_actual = normalize_grids(plane, grid1_actual, grid2_actual)
	grid1_pred, grid2_pred = normalize_grids(plane, grid1_pred, grid2_pred)
	geometry1, geometry2 = normalize_grids(plane, geometry1, geometry2)
	cut, tolerance = normalize_cut_tolerance(plane, cut, tolerance)
	_actual = filtered_df_data[physics]
	_pred = filtered_df[physics]

	fig, axs = plt.subplots(1, 2, figsize=(32,16), sharey=True)
	fig.suptitle(f'Comparison of Actual vs. Predicted values with Wind Angle = {wind_angle} in the {plane} Plane with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')
	vmin, vmax, levels, cmap, scatter_size, ticks = plotting_details(_actual)
	scatter_actual = axs[0].scatter(grid1_actual, grid2_actual, c=_actual, s=scatter_size, vmin=np.min(_actual), vmax=np.max(_actual), cmap=cmap)
	scatter_cbar_actual = fig.colorbar(scatter_actual, ax=axs[0])
	scatter_cbar_actual.set_label(f'{physics} Residual - Actual', rotation=270, labelpad=15)
	axs[0].scatter(geometry1, geometry2, c='black', s=scatter_size, label='Geometry')
	axs[0].set_xlabel(f'{coordinate1} Coordinate')
	axs[0].set_ylabel(f'{coordinate2} Coordinate')
	axs[0].set_xlim(lim_min1, lim_max1) 
	axs[0].set_ylim(lim_min2, lim_max2)
	axs[0].set_title(f'Actual {physics} Residual in the {plane} Plane for Wind Angle = {wind_angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f} \n Mean = {np.mean(_actual):.2f} and Standard Deviation = {np.std(_actual):.2f}')

	vmin, vmax, levels, cmap, scatter_size, ticks = plotting_details(_pred)
	scatter_pred = axs[1].scatter(grid1_pred, grid2_pred, c=_pred, s=scatter_size, vmin=np.min(_actual), vmax=np.max(_actual), cmap=cmap)
	scatter_cbar_pred = fig.colorbar(scatter_pred, ax=axs[1])
	scatter_cbar_pred.set_label(f'{physics} Residual - Predicted', rotation=270, labelpad=15)
	axs[1].scatter(geometry1, geometry2, c='black', s=scatter_size, label='Geometry')
	axs[1].set_xlabel(f'{coordinate1} Coordinate')
	axs[1].set_ylabel(f'{coordinate2} Coordinate')
	axs[1].set_xlim(lim_min1, lim_max1) 
	axs[1].set_ylim(lim_min2, lim_max2)
	axs[1].set_title(f'Predicted {physics} Residual in the {plane} Plane for Wind Angle = {wind_angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f} \n Mean = {np.mean(_pred):.2f} and Standard Deviation = {np.std(_pred):.2f}')

	plt.tight_layout()
	plt.savefig(savename)
	plt.close()

def plot_angles_2d(df_data,df_torch,physics,angle,config,plot_folder):
    params = config["plotting"]["plotting_params"]
    chosen_machine_key = config["chosen_machine"]
    datafolder_path = os.path.join(config["machine"][chosen_machine_key], config["data"]["data_folder_name"])
    for j in params:
        plane = j[0]
        cut = j[1]
        tolerance = j[2]
        savename = os.path.join(plot_folder,f'{plane}_{physics}_{angle}.png')
        plot_physics_predictions(df_torch,df_data,angle,config,datafolder_path,savename,plane,cut,tolerance,physics)

def evaluation_physics(model, device, config, data_dict, model_file_path, output_folder, today, overall_start_time, physics):
	testing_type = f"Evaluation {physics}"
	chosen_machine_key = config["chosen_machine"]
	datafolder_path = os.path.join(config["machine"][chosen_machine_key], config["data"]["data_folder_name"])
	geometry_filename = os.path.join(datafolder_path, config["data"]["geometry"])
	rho = config["data"]["density"]
	nu = config["data"]["kinematic_viscosity"]
	checkpoint = open_model_file(model_file_path, device)
	wind_angles = config["training"]["all_angles"]
	plot_folder = os.path.join(output_folder, f'plots_output_for_{physics}_{today}_{checkpoint["epoch"]}')
	os.makedirs(plot_folder, exist_ok=True)
	print (f'Model {testing_type} at Epoch = {checkpoint["epoch"]} and Training Completed = {checkpoint["training_completed"]}, Time: {(time.time() - overall_start_time):.2f} Seconds')
	model.load_state_dict(checkpoint['model_state_dict'])
	X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, labels_train_tensor, X_train_tensor_skipped, y_train_tensor_skipped, X_test_tensor_skipped, y_test_tensor_skipped, feature_scaler, target_scaler, X_train_divergences_tensor, X_test_divergences_tensor, y_train_divergences_tensor, y_test_divergences_tensor = data_dict.values()
	for wind_angle in wind_angles:
		X_test_tensor_new = load_data_new_angles(device, config, feature_scaler, target_scaler, [wind_angle])
		df_torch = evaluate_model_physics(config, model, [wind_angle], X_test_tensor_new, feature_scaler, target_scaler, plot_folder, physics)
		df_data = load_derivative_data(config, [wind_angle])
		df_data = evaluate_RANS_data(df_data, rho, nu)
		df_data = evaluate_div_data(df_data)
		print (f'Model Evaluated and Starting to Plot for Wind Angle = {wind_angle}, Time: {(time.time() - overall_start_time):.2f} Seconds')
		plot_angles_2d(df_data,df_torch,physics,wind_angle,config,plot_folder)
		if physics == 'Div':
			div_line_plot_folder = os.path.join(output_folder, f'plots_output_for_Div_line_{today}_{checkpoint["epoch"]}')
			os.makedirs(div_line_plot_folder, exist_ok=True)
			df_data_line, X_test_tensor_line = load_line_div_data(device, config, [wind_angle], feature_scaler, target_scaler)
			df_torch_line = evaluate_model_physics(config, model, [wind_angle], X_test_tensor_line, feature_scaler, target_scaler, div_line_plot_folder, physics)
			savename_div_line = os.path.join(div_line_plot_folder,f'div_line_{wind_angle}')
			plot_line_predictions(df_data_line,df_torch_line,wind_angle,savename_div_line)
		print (f'Plotting Done for Wind Angle = {wind_angle}, Time: {(time.time() - overall_start_time):.2f} Seconds')