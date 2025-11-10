from definitions import * 

# def invert_data_PCA(config, features, targets, predictions, pca_features=None, pca_targets=None, pls=None, orig=False):

#     output_params = config["training"]["output_params"]
#     input_params = config["training"]["input_params"]

#     if config["training"]["PCA"]:

#         features_reconstruct = compute_inverse_PCA_from_PCA(features, pca_features, config["training"]["PCA_Reduce"])
#         targets_reconstruct = compute_inverse_PCA_from_PCA(targets, pca_targets, config["training"]["PCA_Reduce"])
#         predictions_reconstruct = compute_inverse_PCA_from_PCA(predictions, pca_targets, config["training"]["PCA_Reduce"])

#         features_reconstruct_reshape = features_reconstruct.reshape(int(features_reconstruct.shape[0]*(features_reconstruct.shape[1]/len(input_params))), len(input_params))
#         targets_reconstruct_reshape = targets_reconstruct.reshape(int(targets_reconstruct.shape[0]*(targets_reconstruct.shape[1]/len(output_params))), len(output_params))
#         predictions_reconstruct_reshape = predictions_reconstruct.reshape(int(predictions_reconstruct.shape[0]*(predictions_reconstruct.shape[1]/len(output_params))), len(output_params))

#         if orig:
#             return features_reconstruct, targets_reconstruct, predictions_reconstruct
#         else:
#             return features_reconstruct_reshape, targets_reconstruct_reshape, predictions_reconstruct_reshape

#     elif config["training"]["PLS"]:
#         features_reconstruct, targets_reconstruct = compute_inverse_PLS_from_PLS(features, targets, pls, config["training"]["PLS_Reduce"])
#         _, predictions_reconstruct = compute_inverse_PLS_from_PLS(features, predictions, pls, config["training"]["PLS_Reduce"])

#         return features_reconstruct, targets_reconstruct, predictions_reconstruct

# # def get_features(config, filenames, datafolder_path, training_wind_angles, angle_to_label, features_scaler=None):
# #     df_points = pd.DataFrame()
    
# #     for filename in sorted(filenames):
# #         df_points_temp = pd.read_csv(os.path.join(datafolder_path, filename))
# #         input_params_points = ['Points:0', 'Points:1', 'Points:2']
# #         df_points = df_points_temp[input_params_points]
# #         break

# #     if features_scaler is None:
# #         features_scaler = config["training"]["feature_scaler"]
# #         features = features_scaler.fit_transform(df_points)
# #     else:
# #         features = features_scaler.transform(df_points)

# #     modified_features = []

# #     for wind_angle in training_wind_angles:
# #         angle_cos = np.cos(np.deg2rad(wind_angle))
# #         angle_sin = np.sin(np.deg2rad(wind_angle))
# #         for feature in features.T:
# #             modified_feature = feature * angle_cos + feature * angle_sin
# #             modified_features.append(modified_feature)

# #     modified_features_array = np.array(modified_features).T

# #     features_tiled = np.tile(modified_features_array, (1, config["training"]["features_factor"]))

# #     return features_tiled, features_scaler

# def get_features(config, filenames, datafolder_path, training_wind_angles, angle_to_label, features_scaler=None):
#     df = pd.DataFrame()
#     df_points = pd.DataFrame()
    
#     for filename in sorted(filenames):
#         df_points_temp = pd.read_csv(os.path.join(datafolder_path, filename))
#         input_params_points = ['Points:0', 'Points:1', 'Points:2']
#         df_points = df_points_temp[input_params_points]
#         break

#     for wind_angle in training_wind_angles:
#         for filename in filenames:
#             wind_angle_ = int(filename.split('_')[-1].split('.')[0])
#             if wind_angle_ == wind_angle:
#                 temp_df = pd.read_csv(os.path.join(datafolder_path, filename))
#                 temp_df['cos(WindAngle)'] = (np.cos(np.deg2rad(wind_angle)))
#                 temp_df['sin(WindAngle)'] = (np.sin(np.deg2rad(wind_angle)))
#                 temp_df = temp_df[['cos(WindAngle)','sin(WindAngle)']]
#                 df = pd.concat([df,df_points,temp_df], axis=1, ignore_index=True)

#     if features_scaler is None:
#         features_scaler = config["training"]["feature_scaler"]
#         features = features_scaler.fit_transform(df)
#     else:
#         features = features_scaler.transform(df)

#     return features, features_scaler

# # def get_features(config, filenames, datafolder_path, training_wind_angles, angle_to_label, features_scaler=None):
# #     df = pd.DataFrame()
# #     df_points = pd.DataFrame()
    
# #     for filename in sorted(filenames):
# #         df_points_temp = pd.read_csv(os.path.join(datafolder_path, filename))
# #         input_params_points = ['Points:0', 'Points:1', 'Points:2']
# #         df_points = df_points_temp[input_params_points]
# #         break

# #     for wind_angle in training_wind_angles:
# #         for filename in filenames:
# #             wind_angle_ = int(filename.split('_')[-1].split('.')[0])
# #             if wind_angle_ == wind_angle:
# #                 temp_df = pd.read_csv(os.path.join(datafolder_path, filename))
# #                 temp_df['cos(WindAngle)'] = (np.cos(np.deg2rad(wind_angle)))
# #                 temp_df['sin(WindAngle)'] = (np.sin(np.deg2rad(wind_angle)))
# #                 temp_df = temp_df[['cos(WindAngle)','sin(WindAngle)']]
# #                 df = pd.concat([df,df_points,temp_df], axis=1, ignore_index=True)

# #     if features_scaler is None:
# #         features_scaler = config["training"]["feature_scaler"]
# #         features = features_scaler.fit_transform(df)
# #     else:
# #         features = features_scaler.transform(df)

#     # features_tiled = np.tile(features, (1, config["training"]["PCA_factor"]))

#     # return features_tiled, features_scaler
    
# # def get_targets(config, filenames, datafolder_path, training_wind_angles, targets_scaler=None):

# #     df = pd.DataFrame()
# #     for wind_angle in training_wind_angles:
# #         for filename in filenames:
# #             wind_angle_ = int(filename.split('_')[-1].split('.')[0])
# #             if wind_angle_ == wind_angle:
# #                 temp_df = pd.read_csv(os.path.join(datafolder_path, filename))
# #                 temp_df = temp_df[config["training"]["output_params"]]
# #                 df = pd.concat([df,temp_df], axis=1, ignore_index=True)

# #     if targets_scaler is None:
# #         targets_scaler = config["training"]["target_scaler"]
# #         targets = targets_scaler.fit_transform(df)
# #     else:
# #         targets = targets_scaler.transform(df)

# #     return targets, targets_scaler

# def get_targets(config, filenames, datafolder_path, training_wind_angles, targets_scaler=None):

#     df = pd.DataFrame()
#     for wind_angle in training_wind_angles:
#         for filename in filenames:
#             wind_angle_ = int(filename.split('_')[-1].split('.')[0])
#             if wind_angle_ == wind_angle:
#                 temp_df = pd.read_csv(os.path.join(datafolder_path, filename))
#                 temp_df = temp_df[config["training"]["output_params"]]
#                 df = pd.concat([df,temp_df], axis=1, ignore_index=True)

#     if targets_scaler is None:
#         targets_scaler = config["training"]["target_scaler"]
#         targets = targets_scaler.fit_transform(df)
#     else:
#         targets = targets_scaler.transform(df)

#     targets_tiled = np.tile(targets, (1, config["training"]["targets_factor"]))

#     return targets_tiled, targets_scaler

# def compute_PLS(config, features, targets, pls):
#     if pls is None:
#         pls = PLSRegression(n_components=features.shape[1], scale=True, max_iter=10000, tol=1e-06, copy=True)
#         if config["training"]["PLS_Reduce"] == 'rows':
#             features_reduced_transposed, targets_reduced_transposed = pls.fit_transform(features.T, targets.T)
#             features_reduced, targets_reduced = features_reduced_transposed.T, targets_reduced_transposed.T
#         elif config["training"]["PLS_Reduce"] == 'columns':
#             features_reduced, targets_reduced = pls.fit_transform(features, targets)
#     else:
#         X_test_reduced, y_test_reduced = pls.transform(features, targets)
#         if config["training"]["PLS_Reduce"] == 'rows':
#             features_reduced_transposed, targets_reduced_transposed = pls.transform(features.T, targets.T)
#             features_reduced, targets_reduced = features_reduced_transposed.T, targets_reduced_transposed.T
#         elif config["training"]["PLS_Reduce"] == 'columns':
#             features_reduced, targets_reduced = pls.transform(features, targets)
#     Z_reduced = [features_reduced, targets_reduced]

#     features_reconstruct, targets_reconstruct = pls.inverse_transform(features_reduced, targets_reduced)

#     compute_mse(features, features_reconstruct)
#     compute_mse(targets, targets_reconstruct)

#     return Z_reduced, pls

# def compute_PCA(config, Z, pca_reduce, pca=None, n_components=None):
#     if pca is None:
#         print ('fitting')
#         if config["training"]["Incremental_PCA"]:
#             if pca_reduce == 'rows':
#                 batch_size = 1
#                 pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
#                 for i in range(0, (Z.T).shape[0], batch_size):
#                     pca.partial_fit(Z.T[i:i+batch_size])
#                 Z_reduced = (pca.transform(Z.T)).T
#             elif pca_reduce == 'columns':
#                 batch_size = int(Z.shape[0]*0.1)
#                 pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
#                 for i in range(0, Z.shape[0], batch_size):
#                     pca.partial_fit(Z[i:i+batch_size])
#                 Z_reduced = (pca.transform(Z))
#         elif config["training"]["Kernel_PCA"]:
#             pca = KernelPCA(n_components=Z.shape[1], kernel='poly', eigen_solver='dense', degree=1, fit_inverse_transform=True)
#             # pca = KernelPCA(n_components=Z.shape[1], kernel='linear', fit_inverse_transform=True)
#             # pca = KernelPCA(n_components=Z.shape[1], kernel='sigmoid', fit_inverse_transform=True)
#             # pca = KernelPCA(n_components=Z.shape[1], kernel=config["training"]["Kernel_PCA_kernel"], fit_inverse_transform=True)
#         elif config["training"]["Sparse_PCA"]:
#             pca = MiniBatchSparsePCA(n_components=Z.shape[1], alpha=0.01, ridge_alpha=0.01, max_iter=1000, callback=None, batch_size=3, verbose=True, shuffle=False, n_jobs=-1, method='cd', random_state=None, tol=0.0000000001, max_no_improvement=100000000)
#         elif config["training"]["PCA"]:
#             pca = PCA(n_components, svd_solver='full')
#         elif config["training"]["lazyPCA"]:
#             Z = da.from_array(Z, chunks=Z.shape)
#             pca = lazyPCA(n_components, svd_solver='full')
#         if pca_reduce == 'rows':
#             Z_reduced = (pca.fit_transform(Z.T)).T
#         elif pca_reduce == 'columns':
#             Z_reduced = (pca.fit_transform(Z))
#         if not (config["training"]["Kernel_PCA"] or config["training"]["Sparse_PCA"]):
#             eigenvalues_ratio = pca.explained_variance_ratio_
#             print(f"Explained variance ratio of the first {len(eigenvalues_ratio)} principal components: {eigenvalues_ratio} w sum = {np.sum(eigenvalues_ratio)} w reduced matrix {Z_reduced.shape}")
#         else:
#             feature_names = pca.get_feature_names_out()
#             print(f"All feature names: {feature_names} w reduced matrix {Z_reduced.shape}")
#     else:
#         print ('im not fitting')
#         if pca_reduce == 'rows':
#             Z_reduced = (pca.transform(Z.T)).T
#         elif pca_reduce == 'columns':
#             Z_reduced = (pca.transform(Z))
#     Z_r = compute_inverse_PCA_from_PCA(Z_reduced, pca, pca_reduce)
#     compute_mse(Z, Z_r, 'inverse of computed PCA')
#     return Z_reduced, pca

# def compute_inverse_PCA_from_PCA(Z_reduced, pca, type_):
#     if type_ == 'rows':
#         Z_r = (pca.inverse_transform((Z_reduced).T)).T
#     if type_ == 'columns':
#         Z_r = (pca.inverse_transform((Z_reduced)))
#     return Z_r

# def compute_inverse_PLS_from_PLS(features_reduced, targets_reduced, pls, type_):
#     if type_ == 'rows':
#         features_reconstruct, targets_reconstruct = pls.inverse_transform(features_reduced.T, targets_reduced.T)
#     if type_ == 'columns':
#         features_reconstruct, targets_reconstruct = pls.inverse_transform(features_reduced, targets_reduced)
#     return features_reconstruct, targets_reconstruct

# def compute_mse(Z, Z_r, description=None):
#     mse = mean_squared_error(np.sqrt(Z**2), np.abs(Z_r))
#     print (f'mse: {mse} - {description}')
#     return mse

# def get_tensor(config, filenames, datafolder_path, training_wind_angles, angle_to_label, device, features_scaler=None, targets_scaler=None, pca_features=None, pca_targets=None, pls=None):
#     features, features_scaler = get_features(config, filenames, datafolder_path, training_wind_angles, angle_to_label, features_scaler)
#     targets, targets_scaler = get_targets(config, filenames, datafolder_path, training_wind_angles, targets_scaler)
#     if (config["training"]["PCA"] or config["training"]["lazyPCA"]):
#         pca_reduce = config["training"]["PCA_Reduce"]
#         reduced_features, pca_features = compute_PCA(config, features, pca_reduce=pca_reduce, pca=pca_features)
#         reduced_targets, pca_targets = compute_PCA(config, targets, pca_reduce=pca_reduce, pca=pca_targets)
#         # reduced_features = reduced_features.compute()
#         # reduced_targets = reduced_targets.compute()
#         X_train_tensor = (torch.tensor(np.array(reduced_features), dtype=torch.float32)).to(device)
#         y_train_tensor = (torch.tensor(np.array(reduced_targets), dtype=torch.float32)).to(device)
#         x = [X_train_tensor, y_train_tensor, reduced_features, reduced_targets, features_scaler, targets_scaler, pca_features, pca_targets]

#         print ('I NEED TO CHECK NOW')
#         targets_reconstructed = compute_inverse_PCA_from_PCA(reduced_targets, pca_targets, 'rows')

#         mse = ((targets - targets_reconstructed) ** 2).mean(axis=0)

#         # Print the MSE for each column
#         for i, mse_value in enumerate(mse, 1):
#             print(f"Column {i} MSE: {mse_value}")

#         print ('I HAVE CHECK NOW')

#     if config["training"]["PLS"]:
#         Z_reduced, pls = compute_PLS(config, features, targets, pls)
#         X_train_tensor = (torch.tensor(np.array(Z_reduced[0]), dtype=torch.float32)).to(device)
#         y_train_tensor = (torch.tensor(np.array(Z_reduced[1]), dtype=torch.float32)).to(device)
#         x = [X_train_tensor, y_train_tensor, Z_reduced[0], Z_reduced[1], features_scaler, targets_scaler, pls]    
#     return x

# # def get_skipped_angles(skipped_wind_angles, training_wind_angles):
# #     new_skipped_angles = []
# #     for i in skipped_wind_angles:
# #         if i < 180:
# #             new_skipped_angles.append(i)
# #     if len(training_wind_angles) > len(skipped_wind_angles):
# #         for i in training_wind_angles:
# #             if i < 180:
# #                 new_skipped_angles.append(i)

# #     skipped_wind_angles = sorted(new_skipped_angles)
# #     skipped_angle_to_label = {angle: idx for idx, angle in enumerate(sorted(skipped_wind_angles))}

# #     return skipped_wind_angles, skipped_angle_to_label


# def get_skipped_angles(skipped_wind_angles, training_wind_angles):
#     def find_closest(target, values):
#         return min(values, key=lambda x: abs(x - target))

#     # values = []
#     # for i in training_wind_angles:
#     #     for j in skipped_wind_angles:
#     #         print (i,j,abs(i-j))
#     #         values.append(abs(i-j))

#     # print (sorted(values))

#     # first_list = training_wind_angles


#     # first_list = training_wind_angles
#     # second_list = skipped_wind_angles

#     # for i, value in enumerate(first_list):
#     #     closest = find_closest(value, second_list)
#     #     first_list[i] = closest


#     skipped_wind_angles = [0,15,30,45,60,75,90,105,120,135,165,180]

#     skipped_angle_to_label = {angle: idx for idx, angle in enumerate(sorted(skipped_wind_angles))}

#     return skipped_wind_angles, skipped_angle_to_label

# def load_data_PCA(config, device):
#     chosen_machine_key = config["chosen_machine"]
#     datafolder_path = os.path.join(config["machine"][chosen_machine_key], config["data"]["data_folder_name"])
#     filenames = get_filenames_from_folder(datafolder_path, config["data"]["extension"], config["data"]["startname_data"])
#     training_wind_angles = config["training"]["angles_to_train"]
#     skipped_wind_angles = config["training"]["angles_to_leave_out"]
#     output_params = config["training"]["output_params"]
#     input_params = config["training"]["input_params"]
#     angle_to_label = {angle: idx for idx, angle in enumerate(sorted(training_wind_angles))}

#     x_training = get_tensor(config, filenames, datafolder_path, training_wind_angles, angle_to_label, device)
#     skipped_wind_angles, skipped_angle_to_label = get_skipped_angles(skipped_wind_angles, training_wind_angles)

#     print (skipped_wind_angles)

#     print ('starting skipped')

#     if (config["training"]["PCA"] or config["training"]["lazyPCA"]):
#         X_train_tensor, y_train_tensor, reduced_features, reduced_targets, feature_scaler, target_scaler, pca_features, pca_targets = x_training
#         x_skipped = get_tensor(config, filenames, datafolder_path, skipped_wind_angles, skipped_angle_to_label, device, feature_scaler, target_scaler, pca_features, pca_targets)
#         X_test_tensor_skipped, y_test_tensor_skipped, reduced_features_skipped, reduced_targets_skipped, _, _, _, _ = x_skipped
#         data_dict = {
#         "X_train_tensor": X_train_tensor,
#         "y_train_tensor": y_train_tensor,
#         "X_test_tensor_skipped": X_test_tensor_skipped,
#         "y_test_tensor_skipped": y_test_tensor_skipped,
#         "feature_scaler": feature_scaler,
#         "target_scaler": target_scaler,
#         "pca_features": pca_features,
#         "pca_targets": pca_targets
#     }
#     if config["training"]["PLS"]:
#         X_train_tensor, y_train_tensor, reduced_features, reduced_targets, feature_scaler, target_scaler, pls = x_training
#         x_skipped = get_tensor(config, filenames, datafolder_path, skipped_wind_angles, skipped_angle_to_label, device, feature_scaler, target_scaler, pls=pls)
#         X_test_tensor_skipped, y_test_tensor_skipped, reduced_features_skipped, reduced_targets_skipped, _, _, _ = x_skipped
#         data_dict = {
#         "X_train_tensor": X_train_tensor,
#         "y_train_tensor": y_train_tensor,
#         "X_test_tensor_skipped": X_test_tensor_skipped,
#         "y_test_tensor_skipped": y_test_tensor_skipped,
#         "feature_scaler": feature_scaler,
#         "target_scaler": target_scaler,
#         "pls": pls,
#     }

#     for i in [reduced_features, reduced_targets, reduced_features_skipped, reduced_targets_skipped]:
#         print (i.shape)

    

#     return data_dict


























































from config import config

config["chosen_machine"] = "CREATE"
config["train_test"]["test_size"] = 0.1
config["train_test"]["train"] = True
config["plotting"]["make_logging_plots"] = True
# config["train_test"]["test"] = True
# config["train_test"]["evaluate"] = True
# # config["plotting"]["make_div_plots"] = True
# # config["plotting"]["make_RANS_plots"] = True
# config["plotting"]["save_csv_predictions"] = True
# config["plotting"]["save_vtk"] = True
# config["plotting"]["make_plots"] = True

# config["training"]["use_reduction"] = True
# config["training"]["PCA"] = True
# config["training"]["features_factor"] = 5
# config["training"]["targets_factor"] = 3
# config["training"]["PLS"] = True
# config["training"]["Kernel_PCA"] = True
# config["training"]["Kernel_PCA_kernel"] = 'precomputed'
# config["training"]["Sparse_PCA"] = True
# config["training"]["lazyPCA"] = True

# config["training"]["PCA_Reduce"] = 'columns'
# config["training"]["PLS_Reduce"] = 'columns'

config["training"]["use_fft"] = True

config["training"]["output_params"] = ['Pressure', 'Velocity:0', 'Velocity:1', 'Velocity:2', 'TurbVisc']
config["training"]["output_params_modf"] = ['Pressure', 'Velocity_X', 'Velocity_Y', 'Velocity_Z', 'TurbVisc']
config["training"]["input_params_points"] = ['Points:0', 'Points:1', 'Points:2']
config["training"]["input_params_points_modf"] = ["X", "Y", "Z"]



# config["training"]["neuron_number"] = 128
# config["training"]["angles_to_train"] = [0]
config["training"]["use_epochs"] = True
config["training"]["num_epochs"] = 2000
config["training"]["print_epochs"] = 1000
config["training"]["save_metrics"] = 1000

config["training"]["number_of_hidden_layers"] = 100


config["loss_components"]["data_loss"] = True






def load_data_PCA(config, device):
    chosen_machine_key = config["chosen_machine"]
    datafolder_path = os.path.join(config["machine"][chosen_machine_key], config["data"]["data_folder_name"])
    filenames = get_filenames_from_folder(datafolder_path, config["data"]["extension"], config["data"]["startname_data"])
    training_wind_angles = config["training"]["angles_to_train"]
    skipped_wind_angles = config["training"]["angles_to_leave_out"]
    output_params = config["training"]["output_params"]
    input_params = config["training"]["input_params"]
    angle_to_label = {angle: idx for idx, angle in enumerate(sorted(training_wind_angles))}

    def get_xyz(config, filenames, datafolder_path):
        df_points = pd.DataFrame()
    
        for filename in sorted(filenames):
            df_points_temp = pd.read_csv(os.path.join(datafolder_path, filename))
            input_params_points = ['Points:0', 'Points:1', 'Points:2']
            df_points = df_points_temp[input_params_points]
            break

        return df_points

    def scale_xyz(df, feature_scaler):
        if feature_scaler is None:
            feature_scaler = config["training"]["feature_scaler"]
            features = feature_scaler.fit_transform(df)
        else:
            features = feature_scaler.transform(df)
        return features, feature_scaler

    def get_features(config, filenames, datafolder_path, wind_angles, angle_to_label, feature_scaler):
        df_points = get_xyz(config, filenames, datafolder_path)
        
        df_points, feature_scaler = scale_xyz(df_points, feature_scaler)

        features_ = []

        for wind_angle in wind_angles:
            cos_val = np.cos(np.deg2rad(wind_angle))
            sin_val = np.sin(np.deg2rad(wind_angle))
            cos_sin_repeated = np.tile([cos_val, sin_val], (df_points.shape[0], 1))
            temp = np.hstack((df_points, cos_sin_repeated))
            features_.append(temp)

        features = np.hstack(features_)

        return features, feature_scaler
        
    def get_targets(config, filenames, datafolder_path, wind_angles, targets_scaler):

        df = pd.DataFrame()
        for wind_angle in wind_angles:
            for filename in filenames:
                wind_angle_ = int(filename.split('_')[-1].split('.')[0])
                if wind_angle_ == wind_angle:
                    temp_df = pd.read_csv(os.path.join(datafolder_path, filename))
                    temp_df = temp_df[config["training"]["output_params"]]
                    df = pd.concat([df,temp_df], axis=1, ignore_index=True)

        if targets_scaler is None:
            targets_scaler = config["training"]["target_scaler"]
            targets_scaler.fit(df)

        targets = targets_scaler.transform(df)
        return targets, targets_scaler

    def get_tensor(config, filenames, datafolder_path, wind_angles, angle_to_label, device, pca_reduce, features_scaler=None, targets_scaler=None, pca_features=None, pca_targets=None):
        features, features_scaler = get_features(config, filenames, datafolder_path, wind_angles, angle_to_label, features_scaler)
        targets, targets_scaler = get_targets(config, filenames, datafolder_path, wind_angles, targets_scaler)
        reduced_features, pca_features = compute_PCA(features, pca_reduce, pca=pca_features)
        reduced_targets, pca_targets = compute_PCA(targets, pca_reduce, pca=pca_targets)

        x = [reduced_features, pca_features, reduced_targets, pca_targets, features, targets, features_scaler, targets_scaler]

        print ('I NEED TO CHECK NOW')
        targets_reconstructed = compute_inverse_PCA_from_PCA(reduced_targets, pca_targets, 'rows')
        mse = ((targets - targets_reconstructed) ** 2).mean(axis=0)
        for i, mse_value in enumerate(mse, 1):
            print(f"Column {i} MSE: {mse_value}")
        print ('I HAVE CHECK NOW')

        return x 

    def compute_PCA(Z, pca_reduce, pca=None, n_components=None):
        if pca is None:
                print ('fitting')
                pca = PCA(n_components, svd_solver='full')
                if pca_reduce == 'rows':
                    Z_reduced = (pca.fit_transform(Z.T)).T
                elif pca_reduce == 'columns':
                    Z_reduced = (pca.fit_transform(Z))
                eigenvalues_ratio = pca.explained_variance_ratio_
                print(f"Explained variance ratio of the first {len(eigenvalues_ratio)} principal components: {eigenvalues_ratio} w sum = {np.sum(eigenvalues_ratio)} w reduced matrix {Z_reduced.shape}")
        else:
                print ('im not fitting')
                if pca_reduce == 'rows':
                    Z_reduced = (pca.transform(Z.T)).T
                elif pca_reduce == 'columns':
                    Z_reduced = (pca.transform(Z))
        Z_r = compute_inverse_PCA_from_PCA(Z_reduced, pca, pca_reduce)
        compute_mse(Z, Z_r, 'inverse of computed PCA')
        return Z_reduced, pca

    def compute_inverse_PCA_from_PCA(Z_reduced, pca, pca_reduce):
            if pca_reduce == 'rows':
                Z_r = (pca.inverse_transform((Z_reduced).T)).T
            if pca_reduce == 'columns':
                Z_r = (pca.inverse_transform((Z_reduced)))
            return Z_r

    def get_skipped_angles(skipped_wind_angles, training_wind_angles):

        for val in skipped_wind_angles:
            index_of_closest = np.argmin(np.abs(np.array(training_wind_angles) - val))
            training_wind_angles[index_of_closest] = val

        skipped_angle_to_label = {angle: idx for idx, angle in enumerate(sorted(training_wind_angles))}

        return training_wind_angles, skipped_angle_to_label

    def compute_mse(Z, Z_r, description=None):
        mse = mean_squared_error(np.abs(Z), np.abs(Z_r))
        print (f'mse: {mse} - {description}')
        return mse

    pca_reduce = 'rows'
    x_training = get_tensor(config, filenames, datafolder_path, training_wind_angles, angle_to_label, device, pca_reduce)
    
    skipped_wind_angles, skipped_angle_to_label = get_skipped_angles(skipped_wind_angles, training_wind_angles)

    print ('starting skipped', skipped_wind_angles)

    x_skipped = get_tensor(config, filenames, datafolder_path, skipped_wind_angles, skipped_angle_to_label, device, pca_reduce, x_training[6], x_training[7], x_training[1], x_training[3])

    X_train_tensor = (torch.tensor(np.array(x_training[0]), dtype=torch.float32)).to(device)
    y_train_tensor = (torch.tensor(np.array(x_training[2]), dtype=torch.float32)).to(device)

    X_test_tensor_skipped = (torch.tensor(np.array(x_skipped[0]), dtype=torch.float32)).to(device)
    y_test_tensor_skipped = (torch.tensor(np.array(x_skipped[2]), dtype=torch.float32)).to(device)

    data_dict = {
        "X_train_tensor": X_train_tensor,
        "y_train_tensor": y_train_tensor,
        "X_test_tensor_skipped": X_test_tensor_skipped,
        "y_test_tensor_skipped": y_test_tensor_skipped,
        "relevant_data_training": x_training,
        "relevant_data_skipped": x_skipped
        }
    
    return data_dict


load_data_PCA(config, device='cuda:0')



# def load_data_fft(config, device):
#     chosen_machine_key = config["chosen_machine"]
#     datafolder_path = os.path.join(config["machine"][chosen_machine_key], config["data"]["data_folder_name"])
#     filenames = get_filenames_from_folder(datafolder_path, config["data"]["extension"], config["data"]["startname_data"])
#     training_wind_angles = config["training"]["angles_to_train"]
#     skipped_wind_angles = config["training"]["angles_to_leave_out"]
#     angle_to_label = {angle: idx for idx, angle in enumerate(sorted(training_wind_angles))}
        
#     data, labels = concatenate_data_files(filenames, datafolder_path, training_wind_angles, angle_to_label)
#     features = data[config["training"]["input_params"]]
#     targets = data[config["training"]["output_params"]]
#     feature_scaler, target_scaler = initialize_and_fit_scalers(features, targets, config)
#     normalized_features, normalized_targets = transform_data_with_scalers(features, targets, feature_scaler, target_scaler)
#     # # X_train, X_test, y_train, y_test, labels_train, labels_test = train_test_split(normalized_features, normalized_targets, labels, test_size=config["train_test"]["test_size"], random_state=config["train_test"]["random_state"])
#     # X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = convert_to_tensor(X_train, X_test, y_train, y_test, device=device)
#     # labels_train_tensor = torch.tensor(np.array(labels_train), dtype=torch.long)
    
#     data_skipped = concatenate_data_files(filenames, datafolder_path, skipped_wind_angles)
#     features_skipped = data_skipped[config["training"]["input_params"]]
#     targets_skipped = data_skipped[config["training"]["output_params"]]
#     # normalized_features_skipped, normalized_targets_skipped = transform_data_with_scalers(features_skipped, targets_skipped, feature_scaler, target_scaler)
#     # X_train_skipped, X_test_skipped, y_train_skipped, y_test_skipped = train_test_split(normalized_features_skipped, normalized_targets_skipped,test_size=len(data_skipped)-1, random_state=config["train_test"]["random_state"])    
#     # X_train_tensor_skipped, y_train_tensor_skipped, X_test_tensor_skipped, y_test_tensor_skipped = convert_to_tensor(X_train_skipped, X_test_skipped, y_train_skipped, y_test_skipped, device=device)
    

    

#     normalized_features, normalized_targets = normalized_features.T, normalized_targets.T

#     features_fft = np.fft.rfftn(normalized_features) #["X", "Y", "Z", "cos(WindAngle)", "sin(WindAngle)"]
#     targets_fft = np.fft.rfftn(normalized_targets) #['Pressure', 'Velocity_X', 'Velocity_Y', 'Velocity_Z', 'TurbVisc']

#     # features_fft = np.fft.fftn(normalized_features)
#     # targets_fft = np.fft.fftn(normalized_targets)

#     print (f' before FFT - {normalized_features.shape} ; after FFT - {features_fft.shape}')
#     print (f' before FFT - {normalized_targets.shape} ; after FFT - {targets_fft.shape}')

#     magnitude_features_fft = np.abs(features_fft)
#     magnitude_targets_fft = np.abs(targets_fft)


#     # Step 2: Define the threshold as the 95th percentile of the magnitude values
#     threshold = np.percentile(magnitude_targets_fft, 95)

#     # Step 3: Find indices of significant modes that exceed the threshold
#     # np.where returns a tuple of arrays, each representing the indices along a dimension
#     significant_modes_indices = np.where(magnitude_targets_fft > threshold)

#     # Optional: Extract and sort these significant modes by their magnitude
#     # This creates a list of (index, magnitude) tuples
#     significant_modes_with_magnitudes = [(index, magnitude_targets_fft[index]) for index in zip(*significant_modes_indices)]

#     # Sort the list by magnitude in descending order
#     significant_modes_with_magnitudes.sort(key=lambda x: x[1], reverse=True)

#     # Optional: If you only need the indices, sorted by magnitude
#     sorted_indices_by_magnitude = [index for index, _ in significant_modes_with_magnitudes]

#     # print("Significant modes indices:", significant_modes_indices)
#     # print("Sorted significant modes with magnitudes:", significant_modes_with_magnitudes)



#     # Step 1: Create a mask of the same shape as features_fft, initialized to False
#     mask = np.zeros_like(targets_fft, dtype=bool)

#     # Step 2: Set True in the mask for significant indices
#     # This example assumes significant_modes_indices can be directly applied
#     # If the shapes are different, you'll need to adjust the indices accordingly
#     mask[significant_modes_indices] = True

#     # Step 3: Apply the mask to features_fft to retain only significant modes
#     targets_fft_filtered = np.zeros_like(targets_fft)
#     targets_fft_filtered[mask] = targets_fft[mask]

#     reduced_representation = (significant_modes_indices, targets_fft[mask])
#     indices_flat = np.stack(significant_modes_indices, axis=-1)
#     significant_values = targets_fft[mask]
#     reduced_representation = np.concatenate([indices_flat, significant_values[:, None]], axis=1)

#     # print("Reduced Representation:\n", reduced_representation)

#     print (targets_fft_filtered.shape, significant_values.shape)

#     # Optional: If you want to zero out non-significant modes instead of creating a new array
#     # features_fft[~mask] = 0  # This line zeroes out all non-significant modes in-place

#     # Proceed with inverse FFT or further analysis on features_fft_filtered as needed

#     # normalized_magnitude_features_fft = magnitude_features_fft / np.sum(magnitude_features_fft)
#     # normalized_magnitude_targets_fft = magnitude_targets_fft / np.sum(magnitude_targets_fft)

#     # # print (normalized_magnitude_features_fft)
#     # # print (normalized_magnitude_targets_fft)

#     # for i in normalized_magnitude_features_fft:
#     #     filtered_values_greater = [value for value in sorted(i, reverse=True) if value > 1e-5]
#     #     filtered_values_smaller = [value for value in sorted(i, reverse=True) if value <= 1e-5]
#     #     print (sum(filtered_values_greater), sum(filtered_values_smaller))
#         # print(filtered_values)  # This will now correctly print the list of filtered values


#     # print (magnitude_features_fft, magnitude_targets_fft)

#     # # Example: Retain frequencies with magnitudes greater than 10% of the maximum magnitude
#     # threshold_features = 0.1 * np.max(magnitude_features_fft)
#     # threshold_targets = 0.1 * np.max(magnitude_targets_fft)


#     # threshold_features = 0.000001 * np.max(magnitude_features_fft)
#     # threshold_targets = 0.000001 * np.max(magnitude_targets_fft)

#     # # Create masks for significant frequencies
#     # significant_features_mask = magnitude_features_fft > threshold_features
#     # significant_targets_mask = magnitude_targets_fft > threshold_targets

#     # Apply the masks to retain only significant frequencies
#     # features_fft_filtered = np.zeros_like(features_fft)
#     # targets_fft_filtered = np.zeros_like(targets_fft)

#     # features_fft_filtered[significant_features_mask] = features_fft[significant_features_mask]
#     # targets_fft_filtered[significant_targets_mask] = targets_fft[significant_targets_mask]

#     # Perform inverse FFT if needed
#     # from numpy.fft import irfftn

#     features_fft_filtered = features_fft
#     # targets_fft_filtered = targets_fft

#     features_reconstructed = np.fft.irfftn(features_fft_filtered)
#     targets_reconstructed = np.fft.irfftn(targets_fft_filtered)

#     # features_reconstructed = np.fft.ifftn(features_fft_filtered)
#     # targets_reconstructed = np.fft.ifftn(targets_fft_filtered)


#     print (f' before FFT - {normalized_features.shape} ; after inv FFT reconstructed - {features_reconstructed.shape}')
#     print (f' before FFT - {normalized_targets.shape} ; after inv FFT reconstructed - {targets_reconstructed.shape}')


#     compute_mse(normalized_features, features_reconstructed, description='MSE inv FFT features')
#     compute_mse(normalized_targets, targets_reconstructed, description='MSE inv FFT targets')

#     data_dict = {}

#     return data_dict


# def load_data_fft(config, device):
#     chosen_machine_key = config["chosen_machine"]
#     datafolder_path = os.path.join(config["machine"][chosen_machine_key], config["data"]["data_folder_name"])
#     filenames = get_filenames_from_folder(datafolder_path, config["data"]["extension"], config["data"]["startname_data"])
#     training_wind_angles = config["training"]["angles_to_train"]
#     skipped_wind_angles = config["training"]["angles_to_leave_out"]
#     angle_to_label = {angle: idx for idx, angle in enumerate(sorted(training_wind_angles))}
        
#     data, labels = concatenate_data_files(filenames, datafolder_path, training_wind_angles, angle_to_label)
#     features = data[config["training"]["input_params"]]
#     targets = data[config["training"]["output_params"]]
    
#     data_ = pd.concat([features, targets], axis=1, ignore_index=True)

#     scaler = config["training"]["target_scaler"]
#     data_normalized = scaler.fit_transform(data_)

#     data_normalized = data_normalized.T

#     data_fft = np.fft.rfftn(data_normalized)

#     print (f' before FFT - {data_normalized.shape} ; after FFT - {data_fft.shape}')


#     data_reconstructed = np.fft.irfftn(data_fft)

#     print (f' before FFT - {data_normalized.shape} ; after inv FFT reconstructed - {data_reconstructed.shape}')



#     compute_mse(data_normalized, data_reconstructed, description='MSE inv FFT')

#     data_dict = {}

#     return data_dict


# def load_data_fft(config, device):

#     chosen_machine_key = config["chosen_machine"]
#     datafolder_path = os.path.join(config["machine"][chosen_machine_key], config["data"]["data_folder_name"])
#     filenames = get_filenames_from_folder(datafolder_path, config["data"]["extension"], config["data"]["startname_data"])
#     training_wind_angles = config["training"]["angles_to_train"]
#     skipped_wind_angles = config["training"]["angles_to_leave_out"]
#     angle_to_label = {angle: idx for idx, angle in enumerate(sorted(training_wind_angles))}
        
#     data, labels = concatenate_data_files(filenames, datafolder_path, training_wind_angles, angle_to_label)
#     features = data[config["training"]["input_params"]]
#     targets = data[config["training"]["output_params"]]

#     def get_xyz(config, filenames, datafolder_path):
#         df_points = pd.DataFrame()
    
#         for filename in sorted(filenames):
#             df_points_temp = pd.read_csv(os.path.join(datafolder_path, filename))
#             input_params_points = ['Points:0', 'Points:1', 'Points:2']
#             df_points = df_points_temp[input_params_points]
#             break

#         return df_points

#     def scale_xyz(df, feature_scaler):
#         if feature_scaler is None:
#             feature_scaler = config["training"]["feature_scaler"]
#             features = feature_scaler.fit_transform(df)
#         else:
#             features = feature_scaler.transform(df)
#         return features, feature_scaler

#     def get_features(config, filenames, datafolder_path, wind_angles, angle_to_label, feature_scaler):
#         df_points = get_xyz(config, filenames, datafolder_path)
        
#         df_points, feature_scaler = scale_xyz(df_points, feature_scaler)

#         features_ = []

#         for wind_angle in wind_angles:
#             cos_val = np.cos(np.deg2rad(wind_angle))
#             sin_val = np.sin(np.deg2rad(wind_angle))
#             cos_sin_repeated = np.tile([cos_val, sin_val], (df_points.shape[0], 1))
#             temp = np.hstack((df_points, cos_sin_repeated))
#             features_.append(temp)

#         features = np.hstack(features_)

#         return features, feature_scaler
        
#     def get_targets(config, filenames, datafolder_path, wind_angles, targets_scaler):

#         df = pd.DataFrame()
#         for wind_angle in wind_angles:
#             for filename in filenames:
#                 wind_angle_ = int(filename.split('_')[-1].split('.')[0])
#                 if wind_angle_ == wind_angle:
#                     temp_df = pd.read_csv(os.path.join(datafolder_path, filename))
#                     temp_df = temp_df[config["training"]["output_params"]]
#                     df = pd.concat([df,temp_df], axis=1, ignore_index=True)

#         if targets_scaler is None:
#             targets_scaler = config["training"]["target_scaler"]
#             targets = targets_scaler.fit_transform(df)
#             return targets, targets_scaler
#         else:
#             targets = targets_scaler.transform(df)
#             return targets

#     def compute_mse(Z, Z_r, description=None):
#         mse = mean_squared_error(np.abs(Z), np.abs(Z_r))
#         print (f'mse: {mse} - {description}')
#         return mse

#     def compute_PCA(A):
#         data_dask = da.from_array(A, chunks=A.shape)
#         u, s, v = svd(data_dask)
#         U, s, Vh = da.compute(u, s, v)

#         V = Vh.conjugate().T
#         variance_explained = s**2 / np.sum(s**2)
#         projected_data = np.dot(A, V)

#         #A = 10^7, 60, 
#         #U = 10^7, 10^7 ; V.T = 60, 60

#         # u, s, v = svd(data_dask)
#         #u = 60, 60 ; v.T = 10^7, 60

#         # A.T = 60,10^7

#         print(f"SVD: U - {U.shape}, s - {s.shape}, V_dagger - {Vh.shape}")
#         print("Variance explained by each principal component:", variance_explained)
#         print("Projected data:\n", projected_data.shape)

#         return projected_data, Vh

#         small_matrix, hermit_eigenvectors = compute_PCA(big_matrix.T)

#     def apply_PCA(A, Vh):
#         V = Vh.conjugate().T
#         projected_data = np.dot(A, V)
#         return projected_data, Vh

#     def compute_inverse_PCA(A, Vh):
#         reconstructed_data = np.dot(A, Vh)
#         return reconstructed_data

#     def compute_fft_pca(config, filenames, datafolder_path, wind_angles, angle_to_label, feature_scaler=None, target_scaler=None, features_fft_eigenvectors=None, targets_fft_eigenvectors=None):

#         features, feature_scaler = get_features(config, filenames, datafolder_path, wind_angles, angle_to_label, feature_scaler)
#         targets, targets_scaler = get_targets(config, filenames, datafolder_path, wind_angles, target_scaler)

#         features, targets = features.T, targets.T

#         features_fft = np.fft.rfftn(features) #["X", "Y", "Z", "cos(WindAngle)", "sin(WindAngle)"]
#         targets_fft = np.fft.rfftn(targets) #['Pressure', 'Velocity_X', 'Velocity_Y', 'Velocity_Z', 'TurbVisc']

#         print (f'FEATURES before FFT - {features.shape} ; after FFT - {features_fft.shape}')
#         print (f'TARGETS before FFT - {targets.shape} ; after FFT - {targets_fft.shape}')

#         # features_fft_dagger = features_fft.conjugate().T
#         # targets_fft_dagger = targets_fft.conjugate().T

#         # features_fft_norm = features_fft_dagger.dot(features_fft)
#         # targets_fft_norm = targets_fft_dagger.dot(targets_fft)

#         # features_fft_norm = np.abs(features_fft)
#         # targets_fft_norm = np.abs(targets_fft)

#         # print (f'FEATURES before FFT - {features.shape} ; after norm - {features_fft_norm.shape}')
#         # print (f'TARGETS before FFT - {targets.shape} ; after norm - {targets_fft_norm.shape}')

#         # pca_features = PCA(n_components=None, svd_solver='full')
#         # pca_targets = PCA(n_components=None, svd_solver='full')

#         # pca_features.fit(features_fft_norm)
#         # pca_targets.fit(targets_fft_norm)

#         # features_fft_pca = pca_features.transform(features_fft)
#         # targets_fft_pca = pca_targets.transform(targets_fft)

#         if features_fft_eigenvectors is None:
#             features_fft_pca, features_fft_eigenvectors = compute_PCA(features_fft)
#             targets_fft_pca, targets_fft_eigenvectors = compute_PCA(targets_fft)
#         else:
#             features_fft_pca, features_fft_eigenvectors = apply_PCA(features_fft, features_fft_eigenvectors)
#             targets_fft_pca, targets_fft_eigenvectors = apply_PCA(targets_fft, targets_fft_eigenvectors)

#         print (f'FEATURES before FFT - {features.shape} ; after FFT, PCA - {features_fft_pca.shape}')
#         print (f'TARGETS before FFT - {targets.shape} ; after FFT, PCA - {targets_fft_pca.shape}')

#         x = [features_fft, features_fft_pca, features_fft_eigenvectors, targets_fft, targets_fft_pca, targets_fft_eigenvectors, features, targets, feature_scaler, target_scaler]

#         return x

#     def compute_inv_fft_inv_pca(x):

#         [features_fft, features_fft_pca, features_fft_eigenvectors, targets_fft, targets_fft_pca, targets_fft_eigenvectors, features, targets, feature_scaler, target_scaler] = x

#         features_fft_reconstructed = compute_inverse_PCA(features_fft_pca, features_fft_eigenvectors)
#         targets_fft_reconstructed = compute_inverse_PCA(targets_fft_pca, targets_fft_eigenvectors)

#         print (f'FEATURES before FFT - {features.shape} ; after FFT, inverse PCA - {features_fft_reconstructed.shape}')
#         print (f'TARGETS before FFT - {targets.shape} ; after FFT, inverse PCA - {targets_fft_reconstructed.shape}')

#         compute_mse(features_fft, features_fft_reconstructed, description='FEATURES MSE FFT inv PCA')
#         compute_mse(targets_fft, targets_fft_reconstructed, description='TARGETS MSE FFT inv PCA')

#         features_reconstructed = np.fft.irfftn(features_fft, s=features.shape)
#         targets_reconstructed = np.fft.irfftn(targets_fft, s=targets.shape)

#         print (f'FEATURES before FFT - {features.shape} ; after inv FFT reconstructed - {features_reconstructed.shape}')
#         print (f'TARGETS before FFT - {targets.shape} ; after inv FFT reconstructed - {targets_reconstructed.shape}')

#         compute_mse(features, features_reconstructed, description='FEATURES MSE inv FFT')
#         compute_mse(targets, targets_reconstructed, description='TARGETS MSE inv FFT')


#         features_reconstructed = np.fft.irfftn(features_fft_reconstructed, s=features.shape)
#         targets_reconstructed = np.fft.irfftn(targets_fft_reconstructed, s=targets.shape)

#         print (f'FEATURES before FFT - {features.shape} ; after inv FFT reconstructed - {features_reconstructed.shape}')
#         print (f'TARGETS before FFT - {targets.shape} ; after inv FFT reconstructed - {targets_reconstructed.shape}')

#         x_reconstructed = [features_reconstructed, targets_reconstructed]

#         return x_reconstructed

#     def get_skipped_angles(skipped_wind_angles, training_wind_angles):

#         for val in skipped_wind_angles:
#             index_of_closest = np.argmin(np.abs(np.array(training_wind_angles) - val))
#             training_wind_angles[index_of_closest] = val

#         skipped_angle_to_label = {angle: idx for idx, angle in enumerate(sorted(training_wind_angles))}

#         return training_wind_angles, skipped_angle_to_label

#     x_training = compute_fft_pca(config, filenames, datafolder_path, training_wind_angles, angle_to_label)
#     x_training_reconstructed = compute_inv_fft_inv_pca(x_training)
#     compute_mse(x_training[6], x_training_reconstructed[0], description='FEATURES TRAINING MSE inv FFT inv PCA')
#     compute_mse(x_training[7], x_training_reconstructed[1], description='TARGETS TRAINING MSE inv FFT inv PCA')

#     skipped_wind_angles, skipped_angle_to_label = get_skipped_angles(skipped_wind_angles, training_wind_angles)
#     skipped_wind_angles_ = [0,15,30,45,60,75,90,105,120,135,165,180]
#     print ('skipped', skipped_wind_angles)

#     x_skipped = compute_fft_pca(config, filenames, datafolder_path, skipped_wind_angles, skipped_angle_to_label, x_training[8], x_training[9], x_training[2], x_training[5])
#     x_skipped_reconstructed = compute_inv_fft_inv_pca(x_skipped)
#     compute_mse(x_skipped[6], x_skipped_reconstructed[0], description='FEATURES SKIPPED MSE inv FFT inv PCA')
#     compute_mse(x_skipped[7], x_skipped_reconstructed[1], description='TARGETS SKIPPED MSE inv FFT inv PCA')

#     def to_real_imag(tensor_complex):
#         real_part = tensor_complex.real
#         imag_part = tensor_complex.imag
#         return np.stack((real_part, imag_part), axis=-1)

#     X_train_real_imag = to_real_imag(np.array(x_training[1]))
#     y_train_real_imag = to_real_imag(np.array(x_training[4]))

#     print (x_training[1].shape, X_train_real_imag.shape)

#     X_train_tensor = torch.tensor(X_train_real_imag, dtype=torch.float32).to(device)
#     y_train_tensor = torch.tensor(y_train_real_imag, dtype=torch.float32).to(device)

#     X_test_real_imag_skipped = to_real_imag(np.array(x_skipped[1]))
#     y_test_real_imag_skipped = to_real_imag(np.array(x_skipped[4]))

#     X_test_tensor_skipped = torch.tensor(X_test_real_imag_skipped, dtype=torch.float32).to(device)
#     y_test_tensor_skipped = torch.tensor(y_test_real_imag_skipped, dtype=torch.float32).to(device)


#     all_tensors = [X_train_tensor, y_train_tensor, X_test_tensor_skipped, y_test_tensor_skipped]
#     for tensor in all_tensors:
#         print (tensor.shape)
#         tensor = tensor.view(tensor.size(0), -1)
#         print (tensor.shape)

#     data_dict = {
#         "X_train_tensor": X_train_tensor,
#         "y_train_tensor": y_train_tensor,
#         "X_test_tensor_skipped": X_test_tensor_skipped,
#         "y_test_tensor_skipped": y_test_tensor_skipped,
#         "relevant_data_training": x_training,
#         "relevant_data_skipped": x_skipped
#         }

#     return data_dict


# load_data_fft(config, device='cuda:0')



# def load_data_fft(config, device):

#     chosen_machine_key = config["chosen_machine"]
#     datafolder_path = os.path.join(config["machine"][chosen_machine_key], config["data"]["data_folder_name"])
#     filenames = get_filenames_from_folder(datafolder_path, config["data"]["extension"], config["data"]["startname_data"])
#     training_wind_angles = config["training"]["angles_to_train"]
#     skipped_wind_angles = config["training"]["angles_to_leave_out"]
#     angle_to_label = {angle: idx for idx, angle in enumerate(sorted(training_wind_angles))}
        
#     data, labels = concatenate_data_files(filenames, datafolder_path, training_wind_angles, angle_to_label)
#     features = data[config["training"]["input_params"]]
#     targets = data[config["training"]["output_params"]]

#     def get_xyz(config, filenames, datafolder_path):
#         df_points = pd.DataFrame()
    
#         for filename in sorted(filenames):
#             df_points_temp = pd.read_csv(os.path.join(datafolder_path, filename))
#             input_params_points = ['Points:0', 'Points:1', 'Points:2']
#             df_points = df_points_temp[input_params_points]
#             break

#         return df_points

#     def scale_xyz(df, feature_scaler):
#         if feature_scaler is None:
#             feature_scaler = config["training"]["feature_scaler"]
#             features = feature_scaler.fit_transform(df)
#         else:
#             features = feature_scaler.transform(df)
#         return features, feature_scaler

#     def get_features(config, filenames, datafolder_path, wind_angles, angle_to_label, feature_scaler):
#         df_points = get_xyz(config, filenames, datafolder_path)
        
#         df_points, feature_scaler = scale_xyz(df_points, feature_scaler)

#         features_ = []

#         for wind_angle in wind_angles:
#             cos_val = np.cos(np.deg2rad(wind_angle))
#             sin_val = np.sin(np.deg2rad(wind_angle))
#             cos_sin_repeated = np.tile([cos_val, sin_val], (df_points.shape[0], 1))
#             temp = np.hstack((df_points, cos_sin_repeated))
#             features_.append(temp)

#         features = np.hstack(features_)

#         return features, feature_scaler
        
#     def get_targets(config, filenames, datafolder_path, wind_angles, targets_scaler):

#         df = pd.DataFrame()
#         for wind_angle in wind_angles:
#             for filename in filenames:
#                 wind_angle_ = int(filename.split('_')[-1].split('.')[0])
#                 if wind_angle_ == wind_angle:
#                     temp_df = pd.read_csv(os.path.join(datafolder_path, filename))
#                     temp_df = temp_df[config["training"]["output_params"]]
#                     df = pd.concat([df,temp_df], axis=1, ignore_index=True)

#         if targets_scaler is None:
#             targets_scaler = config["training"]["target_scaler"]
#             targets = targets_scaler.fit_transform(df)
#             return targets, targets_scaler
#         else:
#             targets = targets_scaler.transform(df)
#             return targets

#     def compute_mse(Z, Z_r, description=None):
#         mse = mean_squared_error(np.abs(Z), np.abs(Z_r))
#         print (f'mse: {mse} - {description}')
#         return mse

#     def compute_PCA(A):
#         data_dask = da.from_array(A, chunks=A.shape)
#         u, s, v = svd(data_dask)
#         U, s, Vh = da.compute(u, s, v)

#         V = Vh.conjugate().T
#         variance_explained = s**2 / np.sum(s**2)
#         projected_data = np.dot(A, V)

#         #A = 10^7, 60, 
#         #U = 10^7, 10^7 ; V.T = 60, 60

#         # u, s, v = svd(data_dask)
#         #u = 60, 60 ; v.T = 10^7, 60

#         # A.T = 60,10^7

#         print(f"SVD: A - {A.shape}, U - {U.shape}, s - {s.shape}, V_dagger - {Vh.shape}")
#         print(f"SVD: U - {U.shape}, s - {s.shape}, V_dagger - {Vh.shape}")
#         print("Variance explained by each principal component:", variance_explained)
#         print("Projected data:\n", projected_data.shape)

#         return projected_data, Vh

#     def apply_PCA(A, Vh):
#         V = Vh.conjugate().T
#         projected_data = np.dot(A, V)
#         return projected_data, Vh

#     def compute_inverse_PCA(A, Vh):
#         reconstructed_data = np.dot(A, Vh)
#         return reconstructed_data

#     def compute_fft_pca(config, filenames, datafolder_path, wind_angles, angle_to_label, feature_scaler=None, target_scaler=None, features_fft_eigenvectors=None, targets_fft_eigenvectors=None):

#         features, feature_scaler = get_features(config, filenames, datafolder_path, wind_angles, angle_to_label, feature_scaler)
#         targets, targets_scaler = get_targets(config, filenames, datafolder_path, wind_angles, target_scaler)

#         features, targets = features.T, targets.T

#         x = compute_PCA(targets)

#         # features_fft = np.fft.rfftn(features) #["X", "Y", "Z", "cos(WindAngle)", "sin(WindAngle)"]
#         # targets_fft = np.fft.rfftn(targets) #['Pressure', 'Velocity_X', 'Velocity_Y', 'Velocity_Z', 'TurbVisc']

#         # print (f'FEATURES before FFT - {features.shape} ; after FFT - {features_fft.shape}')
#         # print (f'TARGETS before FFT - {targets.shape} ; after FFT - {targets_fft.shape}')

#         # features_fft_dagger = features_fft.conjugate().T
#         # targets_fft_dagger = targets_fft.conjugate().T

#         # features_fft_norm = features_fft_dagger.dot(features_fft)
#         # targets_fft_norm = targets_fft_dagger.dot(targets_fft)

#         # features_fft_norm = np.abs(features_fft)
#         # targets_fft_norm = np.abs(targets_fft)

#         # print (f'FEATURES before FFT - {features.shape} ; after norm - {features_fft_norm.shape}')
#         # print (f'TARGETS before FFT - {targets.shape} ; after norm - {targets_fft_norm.shape}')

#         # pca_features = PCA(n_components=None, svd_solver='full')
#         # pca_targets = PCA(n_components=None, svd_solver='full')

#         # pca_features.fit(features_fft_norm)
#         # pca_targets.fit(targets_fft_norm)

#         # features_fft_pca = pca_features.transform(features_fft)
#         # targets_fft_pca = pca_targets.transform(targets_fft)

#         # if features_fft_eigenvectors is None:
#         #     features_fft_pca, features_fft_eigenvectors = compute_PCA(features_fft)
#         #     targets_fft_pca, targets_fft_eigenvectors = compute_PCA(targets_fft)
#         # else:
#         #     features_fft_pca, features_fft_eigenvectors = apply_PCA(features_fft, features_fft_eigenvectors)
#         #     targets_fft_pca, targets_fft_eigenvectors = apply_PCA(targets_fft, targets_fft_eigenvectors)

#         # print (f'FEATURES before FFT - {features.shape} ; after FFT, PCA - {features_fft_pca.shape}')
#         # print (f'TARGETS before FFT - {targets.shape} ; after FFT, PCA - {targets_fft_pca.shape}')

#         # x = [features_fft, features_fft_pca, features_fft_eigenvectors, targets_fft, targets_fft_pca, targets_fft_eigenvectors, features, targets, feature_scaler, target_scaler]

#         # return x

#     def compute_inv_fft_inv_pca(x):

#         [features_fft, features_fft_pca, features_fft_eigenvectors, targets_fft, targets_fft_pca, targets_fft_eigenvectors, features, targets, feature_scaler, target_scaler] = x

#         features_fft_reconstructed = compute_inverse_PCA(features_fft_pca, features_fft_eigenvectors)
#         targets_fft_reconstructed = compute_inverse_PCA(targets_fft_pca, targets_fft_eigenvectors)

#         print (f'FEATURES before FFT - {features.shape} ; after FFT, inverse PCA - {features_fft_reconstructed.shape}')
#         print (f'TARGETS before FFT - {targets.shape} ; after FFT, inverse PCA - {targets_fft_reconstructed.shape}')

#         compute_mse(features_fft, features_fft_reconstructed, description='FEATURES MSE FFT inv PCA')
#         compute_mse(targets_fft, targets_fft_reconstructed, description='TARGETS MSE FFT inv PCA')

#         features_reconstructed = np.fft.irfftn(features_fft, s=features.shape)
#         targets_reconstructed = np.fft.irfftn(targets_fft, s=targets.shape)

#         print (f'FEATURES before FFT - {features.shape} ; after inv FFT reconstructed - {features_reconstructed.shape}')
#         print (f'TARGETS before FFT - {targets.shape} ; after inv FFT reconstructed - {targets_reconstructed.shape}')

#         compute_mse(features, features_reconstructed, description='FEATURES MSE inv FFT')
#         compute_mse(targets, targets_reconstructed, description='TARGETS MSE inv FFT')


#         features_reconstructed = np.fft.irfftn(features_fft_reconstructed, s=features.shape)
#         targets_reconstructed = np.fft.irfftn(targets_fft_reconstructed, s=targets.shape)

#         print (f'FEATURES before FFT - {features.shape} ; after inv FFT reconstructed - {features_reconstructed.shape}')
#         print (f'TARGETS before FFT - {targets.shape} ; after inv FFT reconstructed - {targets_reconstructed.shape}')

#         x_reconstructed = [features_reconstructed, targets_reconstructed]

#         return x_reconstructed

#     def get_skipped_angles(skipped_wind_angles, training_wind_angles):

#         for val in skipped_wind_angles:
#             index_of_closest = np.argmin(np.abs(np.array(training_wind_angles) - val))
#             training_wind_angles[index_of_closest] = val

#         skipped_angle_to_label = {angle: idx for idx, angle in enumerate(sorted(training_wind_angles))}

#         return training_wind_angles, skipped_angle_to_label

#     compute_fft_pca(config, filenames, datafolder_path, training_wind_angles, angle_to_label)
#     # x_training_reconstructed = compute_inv_fft_inv_pca(x_training)
#     # compute_mse(x_training[6], x_training_reconstructed[0], description='FEATURES TRAINING MSE inv FFT inv PCA')
#     # compute_mse(x_training[7], x_training_reconstructed[1], description='TARGETS TRAINING MSE inv FFT inv PCA')

#     # skipped_wind_angles, skipped_angle_to_label = get_skipped_angles(skipped_wind_angles, training_wind_angles)
#     # skipped_wind_angles_ = [0,15,30,45,60,75,90,105,120,135,165,180]
#     # print ('skipped', skipped_wind_angles)

#     # x_skipped = compute_fft_pca(config, filenames, datafolder_path, skipped_wind_angles, skipped_angle_to_label, x_training[8], x_training[9], x_training[2], x_training[5])
#     # x_skipped_reconstructed = compute_inv_fft_inv_pca(x_skipped)
#     # compute_mse(x_skipped[6], x_skipped_reconstructed[0], description='FEATURES SKIPPED MSE inv FFT inv PCA')
#     # compute_mse(x_skipped[7], x_skipped_reconstructed[1], description='TARGETS SKIPPED MSE inv FFT inv PCA')

#     # def to_real_imag(tensor_complex):
#     #     real_part = tensor_complex.real
#     #     imag_part = tensor_complex.imag
#     #     return np.stack((real_part, imag_part), axis=-1)

#     # X_train_real_imag = to_real_imag(np.array(x_training[1]))
#     # y_train_real_imag = to_real_imag(np.array(x_training[4]))

#     # print (x_training[1].shape, X_train_real_imag.shape)

#     # X_train_tensor = torch.tensor(X_train_real_imag, dtype=torch.float32).to(device)
#     # y_train_tensor = torch.tensor(y_train_real_imag, dtype=torch.float32).to(device)

#     # X_test_real_imag_skipped = to_real_imag(np.array(x_skipped[1]))
#     # y_test_real_imag_skipped = to_real_imag(np.array(x_skipped[4]))

#     # X_test_tensor_skipped = torch.tensor(X_test_real_imag_skipped, dtype=torch.float32).to(device)
#     # y_test_tensor_skipped = torch.tensor(y_test_real_imag_skipped, dtype=torch.float32).to(device)


#     # all_tensors = [X_train_tensor, y_train_tensor, X_test_tensor_skipped, y_test_tensor_skipped]
#     # for tensor in all_tensors:
#     #     print (tensor.shape)
#     #     tensor = tensor.view(tensor.size(0), -1)
#     #     print (tensor.shape)

#     # data_dict = {
#     #     "X_train_tensor": X_train_tensor,
#     #     "y_train_tensor": y_train_tensor,
#     #     "X_test_tensor_skipped": X_test_tensor_skipped,
#     #     "y_test_tensor_skipped": y_test_tensor_skipped,
#     #     "relevant_data_training": x_training,
#     #     "relevant_data_skipped": x_skipped
#     #     }

#     return data_dict


# load_data_fft(config, device='cuda:0')