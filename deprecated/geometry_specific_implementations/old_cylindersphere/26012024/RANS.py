from definitions import *
from PINN import *
from plotting_definitions import *

def extract_stds_means(scaler, params):
	stds = scaler.scale_  
	means = scaler.mean_  
	stds_dict = {param + "_std": std for param, std in zip(params, stds)}
	means_dict = {param + "_mean": mean for param, mean in zip(params, means)}
	stds_means_dict = {**stds_dict, **means_dict}
	return stds_means_dict

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

def evaluate_RANS_data(df, rho, nu):
	df['nu_eff'] = df['TurbVisc'] + nu
	df['f'] = (df['u'] * df['u_x'] + df['v'] * df['u_y'] + df['w'] * df['u_z'] - (1 / rho) * df['p_x'] + df['nu_eff'] * (2 * df['u_xx']) + df['nu_eff'] * (df['u_yy'] + df['v_xy']) + df['nu_eff'] * (df['u_zz'] + df['w_xz']))
	df['g'] = (df['u'] * df['v_x'] + df['v'] * df['v_y'] + df['w'] * df['v_z'] - (1 / rho) * df['p_y'] + df['nu_eff'] * (df['v_xx'] + df['u_xy']) + df['nu_eff'] * (2 * df['v_yy']) + df['nu_eff'] * (df['v_zz'] + df['w_yz']))
	df['h'] = (df['u'] * df['w_x'] + df['v'] * df['w_y'] + df['w'] * df['w_z'] - (1 / rho) * df['p_z'] + df['nu_eff'] * (df['w_xx'] + df['u_xz']) + df['nu_eff'] * (df['w_yy'] + df['v_yz']) + df['nu_eff'] * (2 * df['w_zz']))
	df['RANS'] = df['f'] + df['g'] + df['h']
	return df

def load_RANS_data(config, wind_angles):
	chosen_machine_key = config["chosen_machine"]
	datafolder_path = os.path.join(config["machine"][chosen_machine_key], config["data"]["data_folder_name"])
	filenames = get_filenames_from_folder(datafolder_path, config["data"]["extension"], "all_data")
	data = concatenate_data_files(filenames, datafolder_path, wind_angles)
	data.rename(columns={'Points:0': 'X', 'Points:1': 'Y', 'Points:2': 'Z', 
	'GradientVelocity:0': 'u_x', 'GradientVelocity:1': 'u_y', 'GradientVelocity:2': 'u_z', 
	'GradientVelocity:3': 'v_x', 'GradientVelocity:4': 'v_y', 'GradientVelocity:5': 'v_z', 
	'GradientVelocity:6': 'w_x', 'GradientVelocity:7': 'w_y', 'GradientVelocity:8': 'w_z',
	'SecondGradientVelocity:0': 'u_xx','SecondGradientVelocity:1': 'u_xy','SecondGradientVelocity:2': 'u_xz',
	'SecondGradientVelocity:3': 'u_yx','SecondGradientVelocity:4': 'u_yy','SecondGradientVelocity:5': 'u_yz',
	'SecondGradientVelocity:6': 'u_zx','SecondGradientVelocity:7': 'u_zy','SecondGradientVelocity:8': 'u_zz',
	'SecondGradientVelocity:9': 'v_xx','SecondGradientVelocity:10': 'v_xy','SecondGradientVelocity:11': 'v_xz',
	'SecondGradientVelocity:12': 'v_yx','SecondGradientVelocity:13': 'v_yy','SecondGradientVelocity:14': 'v_yz',
	'SecondGradientVelocity:15': 'v_zx','SecondGradientVelocity:16': 'v_zy','SecondGradientVelocity:17': 'v_zz',
	'SecondGradientVelocity:18': 'w_xx','SecondGradientVelocity:19': 'w_xy','SecondGradientVelocity:20': 'w_xz',
	'SecondGradientVelocity:21': 'w_yx','SecondGradientVelocity:22': 'w_yy','SecondGradientVelocity:23': 'w_yz',
	'SecondGradientVelocity:24': 'w_zx','SecondGradientVelocity:25': 'w_zy','SecondGradientVelocity:26': 'w_zz',
	'Velocity:0': 'u', 'Velocity:1': 'v', 'Velocity:2': 'w', 
	'GradientPressure:0': 'p_x', 'GradientPressure:1': 'p_y', 'GradientPressure:2': 'p_z',
	},inplace=True)
	rho = config["data"]["density"]
	nu = config["data"]["kinematic_viscosity"]
	data = evaluate_RANS_data(data, rho, nu)
	return data

def scale_RANS_NN(config, feature_scaler, target_scaler, der_dict):
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

def load_RANS_NN(config, model, X, feature_scaler, target_scaler):
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
	derivatives_dict = scale_RANS_NN(config, feature_scaler, target_scaler, derivatives_dict)

	derivatives_df = pd.DataFrame(derivatives_dict)

	rho = config["data"]["density"]
	nu = config["data"]["kinematic_viscosity"]

	derivatives_df = evaluate_RANS_data(derivatives_df, rho, nu)
	
	return derivatives_df

def evaluate_model_new(config, model, wind_angles, X_test_tensor, feature_scaler, target_scaler, output_folder):
	model.eval()
	dfs = []
	RANS_dataframe = load_RANS_NN(config, model, X_test_tensor, feature_scaler, target_scaler)
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
			filtered_RANS = RANS_dataframe.loc[mask]
			if len(filtered_predictions)!= 0:
				combined_df = pd.concat([filtered_X_test_dataframe, filtered_predictions, filtered_RANS], axis=1)
				combined_df['WindAngle'] = (wind_angle)
				combined_df.to_csv(os.path.join(output_folder, f'RANS_output_{wind_angle}.csv'), index=False)
				dfs.append(combined_df)
	data = pd.concat(dfs)
	return data

def plot_torch_predictions(df,df_data,wind_angle,config,datafolder_path,savename_total_RANS,plane,cut,tolerance):
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
	RANS_actual = filtered_df_data['RANS']
	RANS_pred = filtered_df['RANS']

	fig, axs = plt.subplots(1, 2, figsize=(32,16), sharey=True)
	fig.suptitle(f'Comparison of Actual vs. Predicted values with Wind Angle = {wind_angle} in the {plane} Plane with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')
	vmin, vmax, levels, cmap, scatter_size, ticks = plotting_details(RANS_actual)
	scatter_actual = axs[0].scatter(grid1_actual, grid2_actual, c=RANS_actual, s=scatter_size, vmin=np.min(RANS_actual), vmax=np.max(RANS_actual), cmap=cmap)
	scatter_cbar_actual = fig.colorbar(scatter_actual, ax=axs[0])
	scatter_cbar_actual.set_label('RANS Residual - Actual', rotation=270, labelpad=15)
	# contour_actual = axs[0].contourf(grid1, grid2, RANS_actual, levels=levels, vmin=vmin, vmax=vmax, cmap=cmap)
	# contour_actual = axs[0].contourf(grid1, grid2, RANS_actual, cmap=cmap)
	# cbar = fig.colorbar(contour_actual, ax=axs[0], ticks=ticks)
	# cbar.set_label('RANS Residual - Actual', rotation=270, labelpad=15)
	axs[0].scatter(geometry1, geometry2, c='black', s=scatter_size, label='Geometry')
	axs[0].set_xlabel(f'{coordinate1} Coordinate')
	axs[0].set_ylabel(f'{coordinate2} Coordinate')
	axs[0].set_xlim(lim_min1, lim_max1) 
	axs[0].set_ylim(lim_min2, lim_max2)
	axs[0].set_title(f'Actual RANS Residual in the {plane} Plane for Wind Angle = {wind_angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f} with mean = {np.mean(RANS_actual):.2f} and std dev = {np.std(RANS_actual):.2f}')

	vmin, vmax, levels, cmap, scatter_size, ticks = plotting_details(RANS_pred)
	scatter_pred = axs[1].scatter(grid1_pred, grid2_pred, c=RANS_pred, s=scatter_size, vmin=np.min(RANS_pred), vmax=np.max(RANS_pred), cmap=cmap)
	scatter_cbar_pred = fig.colorbar(scatter_pred, ax=axs[1])
	scatter_cbar_pred.set_label('RANS Residual - Predicted', rotation=270, labelpad=15)
	# contour_pred = axs[1].contourf(grid1, grid2, RANS_pred, levels=levels, vmin=vmin, vmax=vmax, cmap=cmap)
	# contour_pred = axs[1].contourf(grid1, grid2, RANS_pred, cmap=cmap)
	# cbar = fig.colorbar(contour_pred, ax=axs[1], ticks=ticks)
	# cbar = fig.colorbar(contour_actual, ax=axs[1])
	# cbar.set_label('Div Velocity - Predicted', rotation=270, labelpad=15)
	axs[1].scatter(geometry1, geometry2, c='black', s=scatter_size, label='Geometry')
	axs[1].set_xlabel(f'{coordinate1} Coordinate')
	axs[1].set_ylabel(f'{coordinate2} Coordinate')
	axs[1].set_xlim(lim_min1, lim_max1) 
	axs[1].set_ylim(lim_min2, lim_max2)
	axs[1].set_title(f'Predicted RANS Residual in the {plane} Plane for Wind Angle = {wind_angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f} with mean = {np.mean(RANS_pred):.2f} and std dev = {np.std(RANS_pred):.2f}')

	plt.tight_layout()
	plt.savefig(savename_total_RANS)
	plt.close()

def plot_RANS_angles_2d(df_data,df_torch,angle,config,plot_folder):
    params = config["plotting"]["plotting_params"]
    chosen_machine_key = config["chosen_machine"]
    datafolder_path = os.path.join(config["machine"][chosen_machine_key], config["data"]["data_folder_name"])
    for j in params:
        plane = j[0]
        cut = j[1]
        tolerance = j[2]
        savename_total_RANS_doubletorch = os.path.join(plot_folder,f'{plane}_RANS_{angle}.png')
        savename_total_RANS_doubletorch_diff = os.path.join(plot_folder,f'{plane}_RANS_diff_{angle}.png')
        plot_torch_predictions(df_torch,df_data,angle,config,datafolder_path,savename_total_RANS_doubletorch,plane,cut,tolerance)

def evaluation_RANS(model, device, config, data_dict, model_file_path, output_folder, today, overall_start_time):
	testing_type = "Evaluation RANS"
	chosen_machine_key = config["chosen_machine"]
	datafolder_path = os.path.join(config["machine"][chosen_machine_key], config["data"]["data_folder_name"])
	geometry_filename = os.path.join(datafolder_path, config["data"]["geometry"])
	checkpoint = open_model_file(model_file_path, device)
	RANS_wind_angles = config["training"]["all_angles"]
	RANS_plot_folder = os.path.join(output_folder, f'plots_output_for_RANS_{today}_{checkpoint["epoch"]}')
	os.makedirs(RANS_plot_folder, exist_ok=True)
	print (f'Model {testing_type} at Epoch = {checkpoint["epoch"]} and Training Completed = {checkpoint["training_completed"]}, Time: {(time.time() - overall_start_time):.2f} Seconds')
	model.load_state_dict(checkpoint['model_state_dict'])
	X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, labels_train_tensor, X_train_tensor_skipped, y_train_tensor_skipped, X_test_tensor_skipped, y_test_tensor_skipped, feature_scaler, target_scaler, X_train_divergences_tensor, X_test_divergences_tensor, y_train_divergences_tensor, y_test_divergences_tensor = data_dict.values()
	for wind_angle in RANS_wind_angles:
		RANS_wind_angle = [wind_angle]
		X_test_tensor_new = load_data_new_angles(device, config, feature_scaler, target_scaler, RANS_wind_angle)
		df_torch = evaluate_model_new(config, model, RANS_wind_angle, X_test_tensor_new, feature_scaler, target_scaler,RANS_plot_folder)
		df_data = load_RANS_data(config, RANS_wind_angle)

		print (f'Model Evaluated and Starting to Plot for Wind Angle = {wind_angle}, Time: {(time.time() - overall_start_time):.2f} Seconds')
		# savename_total_div_doubletorch_line = os.path.join(div_line_plot_folder,f'div_line_{wind_angle}')
		# plot_line_predictions(df_data_line,df_torch_line,wind_angle,savename_total_div_doubletorch_line)
		plot_RANS_angles_2d(df_data,df_torch,wind_angle,config,RANS_plot_folder)
		print (f'Plotting Done for Wind Angle = {wind_angle}, Time: {(time.time() - overall_start_time):.2f} Seconds')