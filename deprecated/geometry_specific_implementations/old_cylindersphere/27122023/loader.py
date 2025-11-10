from definitions import *
from config import config

def load_data(filenames, datafolder_path, config):
    rho = config["data"]["density"]
    nu = config["data"]["kinematic_viscosity"]
    wind_angles = config["training"]["angles_to_train"]
    angle_to_label = {angle: idx for idx, angle in enumerate(sorted(wind_angles))}
    data, labels = concatenate_data_files(filenames, datafolder_path, wind_angles, angle_to_label)
    features = data[['Points:0', 'Points:1', 'Points:2', 'cos(WindAngle)', 'sin(WindAngle)', 'WindAngle']]
    targets = data[config["training"]["output_params"]]
    feature_scaler, target_scaler = initialize_and_fit_scalers(features, targets, config)
    normalized_features, normalized_targets = features, targets
    X_train, X_test, y_train, y_test, labels_train, labels_test = train_test_split(normalized_features, normalized_targets, labels, test_size=config["train_test"]["test_size"], random_state=config["train_test"]["random_state"])
    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = convert_to_tensor(X_train, X_test, y_train, y_test, device=None)
    labels_train_tensor = torch.tensor(np.array(labels_train), dtype=torch.long)
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, labels_train_tensor, feature_scaler, target_scaler


chosen_machine_key = config["chosen_machine"]
wind_angles = config["training"]["angles_to_train"]
datafolder_path = os.path.join(config["machine"][chosen_machine_key], config["data"]["data_folder_name"])
filenames = get_filenames_from_folder(datafolder_path, config["data"]["extension"], config["data"]["startname_data"])
meteo_filenames = get_filenames_from_folder(datafolder_path, config["data"]["extension"], config["data"]["startname_meteo"])
geometry_filename = os.path.join(datafolder_path,'scaled_cylinder_sphere.stl')
X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, labels_train_tensor, feature_scaler, target_scaler = load_data(filenames, datafolder_path, config)

def get_train_loader(X_train_tensor, y_train_tensor, labels_train_tensor, batch_size, wind_angles):
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    dataset = WindAngleDataset(train_dataset, labels_train_tensor)
    sampler = BalancedWindAngleSampler(dataset, wind_angles=np.arange(len(wind_angles)))
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return train_loader

batch_size = 1
train_loader = get_train_loader(X_train_tensor, y_train_tensor, labels_train_tensor, batch_size, wind_angles)

for i in train_loader:
    print (i, len(i))
    break

for i,j in train_loader:
    print (i,j)
    print (i[0],i[1])

    break
