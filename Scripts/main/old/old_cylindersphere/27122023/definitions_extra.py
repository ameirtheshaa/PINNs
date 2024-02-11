from definitions import *

def evaluate_model_div(config, model, X_test_tensor, y_test_tensor, feature_scaler, target_scaler):
    model.eval()

    print ('lets do this')

    predictions_tensor = model(X_test_tensor)

    wind_angles = config["training"]["angle_to_leave_out"]

    X_test_tensor_cpu = X_test_tensor.cpu()
    X_test_tensor_cpu_inv = inverse_transform_features(X_test_tensor_cpu, feature_scaler)

    x_tensor, y_tensor, z_tensor = X_test_tensor_cpu_inv[:, 0], X_test_tensor_cpu_inv[:, 1], X_test_tensor_cpu_inv[:, 2]
    
    y_test_tensor_cpu = y_test_tensor.cpu()
    y_test_tensor_cpu_inv = inverse_transform_targets(y_test_tensor_cpu, target_scaler)

    vx, vy, vz = y_test_tensor_cpu_inv[:, 0], y_test_tensor_cpu_inv[:, 1], y_test_tensor_cpu_inv[:, 2]

    # Example data (replace these with your actual data)
    vx = np.array(vx)  # Replace with your actual data
    vy = np.array(vy)
    vz = np.array(vz)
    x = np.array(x_tensor)
    y = np.array(y_tensor)
    z = np.array(z_tensor)

    unique_x = np.unique(x)
    unique_y = np.unique(y)
    unique_z = np.unique(z)

    grid_dimensions = (len(unique_x), len(unique_y), len(unique_z))
    print("Grid dimensions:", grid_dimensions)


    vx = y_test_tensor_cpu_inv[:, 0].reshape(grid_dimensions[0], grid_dimensions[1], grid_dimensions[2])
    vy = y_test_tensor_cpu_inv[:, 1].reshape(grid_dimensions[0], grid_dimensions[1], grid_dimensions[2])
    vz = y_test_tensor_cpu_inv[:, 2].reshape(grid_dimensions[0], grid_dimensions[1], grid_dimensions[2])

    # Assuming uniform grid spacing
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]

    # Compute partial derivatives using central differences
    dvx_dx = (np.roll(vx, -1, axis=0) - np.roll(vx, 1, axis=0)) / (2 * dx)
    dvy_dy = (np.roll(vy, -1, axis=1) - np.roll(vy, 1, axis=1)) / (2 * dy)
    dvz_dz = (np.roll(vz, -1, axis=2) - np.roll(vz, 1, axis=2)) / (2 * dz)

    # Calculate divergence
    divergence = dvx_dx + dvy_dy + dvz_dz

    print (divergence)

    # # Handling the boundaries (optional, based on how you want to treat the edges)
    # divergence[0, :, :], divergence[-1, :, :] = 0, 0  # For x boundaries
    # divergence[:, 0, :], divergence[:, -1, :] = 0, 0  # For y boundaries
    # divergence[:, :, 0], divergence[:, :, -1] = 0, 0  # For z boundaries

    # # Convert x_tensor, y_tensor, z_tensor to PyTorch tensors and enable gradient tracking
    # x_tensor = torch.tensor(x_tensor, dtype=torch.float32, requires_grad=True)
    # y_tensor = torch.tensor(y_tensor, dtype=torch.float32, requires_grad=True)
    # z_tensor = torch.tensor(z_tensor, dtype=torch.float32, requires_grad=True)



    # vx, vy, vz = y_test_tensor_cpu_inv[:, 0], y_test_tensor_cpu_inv[:, 1], y_test_tensor_cpu_inv[:, 2]

    # # Ensure vx, vy, vz are also PyTorch tensors with requires_grad set to True
    # vx = torch.tensor(vx, dtype=torch.float32, requires_grad=True)
    # vy = torch.tensor(vy, dtype=torch.float32, requires_grad=True)
    # vz = torch.tensor(vz, dtype=torch.float32, requires_grad=True)

    # # Compute gradients for each velocity component
    # grad_vx_x = torch.autograd.grad(outputs=vx, inputs=x_tensor, grad_outputs=torch.ones_like(vx), only_inputs=True, retain_graph=True, allow_unused=True)[0]
    # grad_vy_y = torch.autograd.grad(outputs=vy, inputs=y_tensor, grad_outputs=torch.ones_like(vy), only_inputs=True, retain_graph=True, allow_unused=True)[0]
    # grad_vz_z = torch.autograd.grad(outputs=vz, inputs=z_tensor, grad_outputs=torch.ones_like(vz), only_inputs=True, retain_graph=True, allow_unused=True)[0]

    # # Compute the divergence as the sum of the partial derivatives
    # divergence_actual = grad_vx_x + grad_vy_y + grad_vz_z

    # vx_pred, vy_pred, vz_pred = predictions_tensor[:, 0], predictions_tensor[:, 1], predictions_tensor[:, 2]


    
    # X_test_column_names = config["training"]["input_params_modf"]
    # X_test_dataframe = pd.DataFrame(X_test_tensor_cpu, columns=X_test_column_names)

    # y_test_tensor_cpu = y_test_tensor.cpu()
    # y_test_tensor_cpu = inverse_transform_targets(y_test_tensor_cpu, target_scaler)
    # output_column_names = config["training"]["output_params_modf"]
    # y_test_column_names = [item + "_Actual" for item in output_column_names]
    # y_test_dataframe = pd.DataFrame(y_test_tensor_cpu, columns=y_test_column_names)
    # y_test_dataframe['Velocity_Magnitude_Actual'] = np.sqrt(y_test_dataframe['Velocity_X_Actual']**2 + 
    #                                         y_test_dataframe['Velocity_Y_Actual']**2 + 
    #                                         y_test_dataframe['Velocity_Z_Actual']**2)


    # predictions_tensor_cpu = predictions_tensor.cpu()
    # predictions_tensor_cpu = inverse_transform_targets(predictions_tensor_cpu, target_scaler)
    # predictions_column_names = [item + "_Predicted" for item in output_column_names]
    # predictions_dataframe = pd.DataFrame(predictions_tensor_cpu, columns=predictions_column_names)
    # predictions_dataframe['Velocity_Magnitude_Predicted'] = np.sqrt(predictions_dataframe['Velocity_X_Predicted']**2 + 
    #                                         predictions_dataframe['Velocity_Y_Predicted']**2 + 
    #                                         predictions_dataframe['Velocity_Z_Predicted']**2)

    #     rows_list = []
    #     for i, var in enumerate(y_test_column_names):
    #         var_cleaned = var.replace('_Actual', '')
    #         actuals = y_test_dataframe.iloc[:, i]
    #         preds = predictions_dataframe.iloc[:, i]

    #         mse = sklearn.metrics.mean_squared_error(actuals, preds)
    #         rmse = np.sqrt(mse)
    #         mae = sklearn.metrics.mean_absolute_error(actuals, preds)
    #         r2 = sklearn.metrics.r2_score(actuals, preds)
            
    #         # Append the new row as a dictionary to the list
    #         rows_list.append({
    #             'Variable': var_cleaned, 
    #             'MSE': mse,
    #             'RMSE': rmse,
    #             'MAE': mae,
    #             'R2': r2
    #         })
        
    #     data_folder = os.path.join(output_folder, f'data_output_for_wind_angle_all')
    #     os.makedirs(data_folder, exist_ok=True)

    #     combined_df = pd.concat([X_test_dataframe, y_test_dataframe, predictions_dataframe], axis=1)
    #     test_predictions.append([combined_df])
    #     combined_file_path = os.path.join(data_folder, f'combined_actuals_and_predictions_for_wind_angle_all.csv')
    #     combined_df.to_csv(combined_file_path, index=False)

    #     metrics_df = pd.DataFrame(rows_list)   
    #     metrics_file_path = os.path.join(data_folder, f'metrics_for_wind_angle_all.csv')
    #     metrics_df.to_csv(metrics_file_path, index=False)

    #     for wind_angle in wind_angles:
    #         lower_bound = wind_angle - 2
    #         upper_bound = wind_angle + 2

    #         X_test_dataframe['WindAngle_rad'] = np.arctan2(X_test_dataframe['sin(WindAngle)'], X_test_dataframe['cos(WindAngle)'])
    #         X_test_dataframe['WindAngle'] = X_test_dataframe['WindAngle_rad'].apply(lambda x: int(np.ceil(np.degrees(x))))
    #         mask = X_test_dataframe['WindAngle'].between(lower_bound, upper_bound)
    #         filtered_X_test_dataframe = X_test_dataframe.loc[mask]

    #         filtered_y_test = y_test_dataframe.loc[mask]
    #         filtered_predictions = predictions_dataframe.loc[mask]

    #         if len(filtered_predictions)!= 0 and len(filtered_y_test)!=0:
    #             rows_list = []
    #             for i, var in enumerate(y_test_column_names):
    #                 var_cleaned = var.replace('_Actual', '')
    #                 actuals = filtered_y_test.iloc[:, i]
    #                 preds = filtered_predictions.iloc[:, i]

    #                 mse = sklearn.metrics.mean_squared_error(actuals, preds)
    #                 rmse = np.sqrt(mse)
    #                 mae = sklearn.metrics.mean_absolute_error(actuals, preds)
    #                 r2 = sklearn.metrics.r2_score(actuals, preds)
                    
    #                 # Append the new row as a dictionary to the list
    #                 rows_list.append({
    #                     'Variable': var_cleaned, 
    #                     'MSE': mse,
    #                     'RMSE': rmse,
    #                     'MAE': mae,
    #                     'R2': r2
    #                 })
                
    #             data_folder = os.path.join(output_folder, f'data_output_for_wind_angle_{wind_angle}')
    #             os.makedirs(data_folder, exist_ok=True)

    #             combined_df = pd.concat([filtered_X_test_dataframe, filtered_y_test, filtered_predictions], axis=1)
    #             test_predictions_wind_angle.append([wind_angle, combined_df])
    #             combined_file_path = os.path.join(data_folder, f'combined_actuals_and_predictions_for_wind_angle_{wind_angle}.csv')
    #             combined_df.to_csv(combined_file_path, index=False)

    #             metrics_df = pd.DataFrame(rows_list)   
    #             metrics_file_path = os.path.join(data_folder, f'metrics_for_wind_angle_{wind_angle}.csv')
    #             metrics_df.to_csv(metrics_file_path, index=False)

    # return test_predictions, test_predictions_wind_angle