from definitions import *
from config import *

config["machine"] = {
        "mac": os.path.join(os.path.expanduser("~"), "Dropbox", "School", "Graduate", "CERN", "Temp_Files", "nonlineardynamics", "ameir_PINNs", "ladefense"),
        "CREATE": os.path.join('Z:\\', "ladefense"),
        "google": f"/content/drive/Othercomputers/MacMini/ladefense",
    }
config["data"]["geometry"] = "ladefense.stl"
config["data"]["startname_data"] = "wtf"
config["training"]["output_params"] = ['Pressure', 'Velocity:0', 'Velocity:1', 'Velocity:2', 'TurbVisc']

config["chosen_machine"] = "CREATE"
chosen_machine_key = config["chosen_machine"]
datafolder_path = os.path.join(config["machine"][chosen_machine_key], config["data"]["data_folder_name"])
filenames = get_filenames_from_folder(datafolder_path, config["data"]["extension"], config["data"]["startname_data"])

all_wind_angles = config["training"]["all_angles"]

def append_nn2CFD(CFD_data, nn_data, name):

    nn_data = nn_data.to_numpy()
    csv_ind = nn_data[:, 0].astype(int)
    # Scalar data
    if nn_data.shape[1] == 2:
        csv_val = nn_data[:, 1]
    # 3D vector data
    elif nn_data.shape[1] == 4:
        csv_val = nn_data[:, 1:4]
    else:
        raise ValueError('User-defined data must either be a scalar or a 3D vector!')

    assert CFD_data.GetNumberOfCells() == len(csv_ind)

    nn_dict = {}
    for i in range(len(csv_ind)):
        cell_id  = csv_ind[i]
        cell_val = csv_val[i]
        nn_dict[cell_id] = cell_val

    nn_dict = {k: nn_dict[k] for k in sorted(nn_dict)}
    nn_array = np.array(list(nn_dict.values()))

    vtk_user_data = numpy_to_vtk(nn_array, deep=True)
    vtk_user_data.SetName(name) 
    CFD_data.GetCellData().AddArray(vtk_user_data)

    return CFD_data

def output_nn_to_vtk(config, angle, filename, df, column_names, output_folder):
    chosen_machine_key = config["chosen_machine"]
    datafolder_path = os.path.join(config["machine"][chosen_machine_key], config["data"]["data_folder_name"])
    core_data = os.path.join(datafolder_path, "core_data", "core_data", f"deg_{angle}")
    case_file = os.path.join(core_data, "RESULTS_FLUID_DOMAIN.case")

    ensight_data = load_ensight_data(case_file)
    ensight_data = ensight_data.GetBlock(0)
    cell_id_to_position = {}
    cell_centers_filter = vtk.vtkCellCenters()
    cell_centers_filter.SetInputData(ensight_data)
    cell_centers_filter.VertexCellsOn()
    cell_centers_filter.Update()
    centers_polydata = cell_centers_filter.GetOutput()
    points = centers_polydata.GetPoints()
    for i in range(points.GetNumberOfPoints()):
        position = points.GetPoint(i)
        cell_id_to_position[i] = position
    output_data = add_cell_ids_to_df(df, cell_id_to_position)
    indices_to_keep = output_data['cell_id']

    ids = vtk.vtkIdTypeArray()
    ids.SetNumberOfComponents(1)
    for cell_id in indices_to_keep:
        ids.InsertNextValue(cell_id)

    selectionNode = vtk.vtkSelectionNode()
    selectionNode.SetFieldType(vtk.vtkSelectionNode.CELL)
    selectionNode.SetContentType(vtk.vtkSelectionNode.INDICES)
    selectionNode.SetSelectionList(ids)
    selection = vtk.vtkSelection()
    selection.AddNode(selectionNode)
    extractSelection = vtk.vtkExtractSelection()
    extractSelection.SetInputData(0, ensight_data)
    extractSelection.SetInputData(1, selection)
    extractSelection.Update()
    trimmed_data = extractSelection.GetOutput()

    column_names.append('cell_id')
    velocities, extras = slice_lists(column_names)
    velocities = ['cell_id', *velocities]

    output_data_trimmed = output_data[column_names]
    output_data_velocity = output_data_trimmed[velocities]

    trimmed_data = append_nn2CFD(trimmed_data, output_data_velocity, 'Velocity_Predicted')

    if len(extras)!=0:
        for i in extras:
            if i != 'cell_id':
                column_name_ = ['cell_id', i]
                output_data_i = output_data_trimmed[column_name_]
                trimmed_data = append_nn2CFD(ensight_data, output_data_i, i)
    
    save_vtu_data(os.path.join(output_folder, f'{filename}.vtu'), trimmed_data)
    save_vtu_data(os.path.join(output_folder, f'Original_{filename}.vtu'), ensight_data)

for angle in all_wind_angles:
	for filename in filenames:
		wind_angle = int(filename.split('_')[-1].split('.')[0])
		if angle == wind_angle:
			print (angle)
			data = pd.read_csv(os.path.join(datafolder_path,filename))
			output_column_names = config["training"]["output_params_modf"]
			data.rename(columns={'Points:0': 'X', 'Points:1': 'Y', 'Points:2': 'Z', 'Velocity:0': 'Velocity_X_Predicted', 'Velocity:1': 'Velocity_Y_Predicted', 'Velocity:2': 'Velocity_Z_Predicted', 'Pressure': 'Pressure_Predicted', 'TurbVisc': 'TurbVisc_Predicted'}, inplace=True)
			predictions_column_names = [item + "_Predicted" for item in output_column_names]
			# df = pd.DataFrame(data, columns=predictions_column_names)
			vtk_output = os.path.join(datafolder_path, "wtf")
			os.makedirs(vtk_output, exist_ok=True)
			output_nn_to_vtk(config, angle, f'Sampled_70_70_10_CFD_{wind_angle}', data, predictions_column_names, vtk_output)