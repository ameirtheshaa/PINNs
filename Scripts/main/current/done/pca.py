from definitions import *
from config import config
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

device = print_and_set_available_gpus()

# config["training"]["output_params"] = ['Pressure', 'Velocity:0', 'Velocity:1', 'Velocity:2', 'TurbVisc']
# config["training"]["input_params"] = ['Points:0', 'Points:1', 'Points:2']
config["chosen_machine"] = "CREATE"

def compute_PCA(datafolder_path, wind_angles, filenames, output_params, input_params, n_components, angle_labels=None):
    concatenated_velocities = pd.DataFrame()
    position_data = pd.DataFrame()
    input_params = ['Points:0', 'Points:1', 'Points:2']

    for angle in wind_angles:
        for filename in filenames:
            wind_angle = int(filename.split('_')[-1].split('.')[0])
            if angle == wind_angle:
                data = pd.read_csv(os.path.join(datafolder_path,filename))
                velocity_data = data[output_params]
                pos_data = data[input_params]
                iteration_velocities = velocity_data.rename(columns=lambda x: f"{x}_t{angle}")
                concatenated_velocities = pd.concat([concatenated_velocities, iteration_velocities], axis=1)
                if len(position_data) == 0:
                    position_data = pd.concat([position_data, pos_data], axis=1)

    concatenated_velocities = concatenated_velocities.T

    feature_scaler_, target_scaler_ = initialize_and_fit_scalers(position_data, concatenated_velocities, config)
    normalized_features, normalized_targets = transform_data_with_scalers(position_data, concatenated_velocities, feature_scaler_, target_scaler_)
    ##split into xtrain ytrain before pca###
    
    pca = PCA(n_components=n_components)
    
    targets_pca = pca.fit_transform(normalized_targets)

    # Displaying the variance ratio to understand how much variance is captured by the components
    explained_variance_ratio = pca.explained_variance_ratio_
    print(f"Explained variance ratio of the first {len(targets_pca.T)} principal components:", explained_variance_ratio)
    print(f"Explained variance sum of the first {len(targets_pca.T)} principal components:", np.sum(explained_variance_ratio))

    targets_pca_df = pd.DataFrame(targets_pca.T, columns=concatenated_velocities.T.columns.tolist())

    dfs = []
    labels = []

    for angle in wind_angles:
        relevant_columns = [col for col in targets_pca_df.columns if col.endswith(f"_t{angle}")]
        temp_df = targets_pca_df[relevant_columns].copy()
        temp_df.columns = [col.split('_t')[0] for col in temp_df.columns]
        temp_df[f'sin(WindAngle)'] = np.sin(np.deg2rad(angle))
        temp_df[f'cos(WindAngle)'] = np.cos(np.deg2rad(angle))
        temp_df = pd.concat([position_data[:len(temp_df)], temp_df], axis=1)

        if angle_labels is not None:
            label = angle_labels[angle]
            labels.extend([label] * len(temp_df))
            temp_df['WindAngle'] = (angle)

        dfs.append(temp_df)

    data = pd.concat(dfs, ignore_index=True)

    if angle_labels is not None:
        labels = np.array(labels)
        return data, labels
    else:
        return data


def load_data_PCA(config, device):
    chosen_machine_key = config["chosen_machine"]
    datafolder_path = os.path.join(config["machine"][chosen_machine_key], config["data"]["data_folder_name"])
    filenames = get_filenames_from_folder(datafolder_path, config["data"]["extension"], config["data"]["startname_data"])
    training_wind_angles = config["training"]["angles_to_train"]
    skipped_wind_angles = config["training"]["angles_to_leave_out"]
    output_params = config["training"]["output_params"]
    input_params = config["training"]["input_params"]
    angle_to_label = {angle: idx for idx, angle in enumerate(sorted(training_wind_angles))}

    n_components=0.999

    data, labels = compute_PCA(datafolder_path, training_wind_angles, filenames, output_params, input_params, n_components, angle_to_label)

    features = data[config["training"]["input_params"]]
    targets = data[config["training"]["output_params"]]

    feature_scaler, target_scaler = initialize_and_fit_scalers(features, targets, config)

    normalized_features, normalized_targets = transform_data_with_scalers(features, targets, feature_scaler, target_scaler)
    X_train, X_test, y_train, y_test, labels_train, labels_test = train_test_split(normalized_features, normalized_targets, labels, test_size=config["train_test"]["test_size"], random_state=config["train_test"]["random_state"])
    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = convert_to_tensor(X_train, X_test, y_train, y_test, device=device)
    labels_train_tensor = torch.tensor(np.array(labels_train), dtype=torch.long)

    data_skipped = compute_PCA(datafolder_path, skipped_wind_angles, filenames, output_params, input_params, n_components=0.999)
    features_skipped = data_skipped[config["training"]["input_params"]]
    targets_skipped = data_skipped[config["training"]["output_params"]]
    print (features)
    print (features_skipped)
    print (targets)
    print (targets_skipped)

    are_columns_identical = (features.columns == features_skipped.columns).all()
    print(f"Column names and order are identical: {are_columns_identical}")


    normalized_features_skipped, normalized_targets_skipped = transform_data_with_scalers(features_skipped, targets_skipped, feature_scaler, target_scaler)
    X_train_skipped, X_test_skipped, y_train_skipped, y_test_skipped = train_test_split(normalized_features_skipped, normalized_targets_skipped,test_size=len(data_skipped)-1, random_state=config["train_test"]["random_state"])    
    X_train_tensor_skipped, y_train_tensor_skipped, X_test_tensor_skipped, y_test_tensor_skipped = convert_to_tensor(X_train_skipped, X_test_skipped, y_train_skipped, y_test_skipped, device=device)

    data_dict = {
        "X_train_tensor": X_train_tensor,
        "y_train_tensor": y_train_tensor,
        "X_test_tensor": X_test_tensor,
        "y_test_tensor": y_test_tensor,
        "labels_train_tensor": labels_train_tensor,
        "X_train_tensor_skipped": X_train_tensor_skipped,
        "y_train_tensor_skipped": y_train_tensor_skipped,
        "X_test_tensor_skipped": X_test_tensor_skipped,
        "y_test_tensor_skipped": y_test_tensor_skipped,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler
    }

    return data_dict



print (load_data_PCA(config, device))


    
    
    






# Projecting the original standardized velocity data onto the 2 principal components
# Note: velocity_pca already contains the projection, so we'll use it directly

# # Transforming the projected data back to the original space for comparison
# # This involves reversing the standardization and PCA transformation

# # Reverse PCA transformation
# velocity_pca_inverse = pca.inverse_transform(velocity_pca)

# # Reverse Standardization
# velocity_original_space = scaler.inverse_transform(velocity_pca_inverse)

# # Creating a DataFrame for the projected data in the original space
# velocity_original_space_df = pd.DataFrame(velocity_original_space, columns=concatenated_velocities.columns)

# original_column_names = [item + "_Original" for item in concatenated_velocities.columns]
# reconstructed_column_names = [item + "_Reconstructed" for item in concatenated_velocities.columns]

# # Displaying the first few rows of the original and reconstructed velocity data for comparison
# comparison_df = pd.concat([concatenated_velocities, velocity_original_space_df], axis=1)
# comparison_df.columns = [*original_column_names, *reconstructed_column_names]


# comparison_df.to_csv('comparison.csv')














# # Load the dataset
# file_path = 'CFD_cell_data_simulation_0.csv'
# data = pd.read_csv(file_path)

# # Display the first few rows of the dataframe to understand its structure
# print (data.head())



# # Extracting velocity components
# velocity_data = data[['Pressure', 'Velocity:0', 'Velocity:1', 'Velocity:2', 'TurbVisc']]

# # Initialize an empty DataFrame to hold the concatenated data
# concatenated_velocities = pd.DataFrame()

# # Loop over the file 12 times to concatenate velocities column-wise
# for i in range(12):  # 12 iterations for 12 timeframes
#     # Rename columns for each iteration to avoid column name duplication
#     iteration_velocities = velocity_data.rename(columns=lambda x: f"{x}_t{i}")
#     concatenated_velocities = pd.concat([concatenated_velocities, iteration_velocities], axis=1)

# # # Add a label column with enumerated class values
# # concatenated_velocities['Label'] = np.random.randint(0, 4, concatenated_velocities.shape[0])  # Example: 4 classes

# # Display the shape of the concatenated DataFrame to verify the structure
# print(concatenated_velocities.shape)
# # Display the first few rows to inspect the result
# print (concatenated_velocities.head())

# # Standardizing the data
# scaler = StandardScaler()
# velocity_standardized = scaler.fit_transform(concatenated_velocities)

# print (pd.DataFrame(velocity_standardized))

# # Applying PCA
# pca = PCA(n_components=6)  # For demonstration, reduce to 2 principal components
# velocity_pca = pca.fit_transform(velocity_standardized)

# # Displaying the variance ratio to understand how much variance is captured by the components
# explained_variance_ratio = pca.explained_variance_ratio_
# print("Explained variance ratio of the first 2 principal components:", explained_variance_ratio)

# velocity_pca_df = pd.DataFrame(velocity_pca)

# velocity_pca_df.to_csv('pca.csv')

# # Projecting the original standardized velocity data onto the 2 principal components
# # Note: velocity_pca already contains the projection, so we'll use it directly

# # Transforming the projected data back to the original space for comparison
# # This involves reversing the standardization and PCA transformation

# # Reverse PCA transformation
# velocity_pca_inverse = pca.inverse_transform(velocity_pca)

# # Reverse Standardization
# velocity_original_space = scaler.inverse_transform(velocity_pca_inverse)

# # Creating a DataFrame for the projected data in the original space
# velocity_original_space_df = pd.DataFrame(velocity_original_space, columns=['Pressure', 'Velocity:0', 'Velocity:1', 'Velocity:2', 'TurbVisc'])

# # Displaying the first few rows of the original and reconstructed velocity data for comparison
# comparison_df = pd.concat([velocity_data.head(), velocity_original_space_df.head()], axis=1)
# comparison_df.columns = ['Original Pressure', 'Original Velocity:0', 'Original Velocity:1', 'Original Velocity:2', 'Original TurbVisc',
#                          'Reconstructed Pressure', 'Reconstructed Velocity:0', 'Reconstructed Velocity:1', 'Reconstructed Velocity:2', 'Reconstructed TurbVisc']


# comparison_df.to_csv('comparison.csv')
