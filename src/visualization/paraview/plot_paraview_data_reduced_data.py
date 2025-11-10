import paraview.simple as pvs
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

cylindercell_epoch = 28280
ladefense_sampled_epoch = 1303
ladefense_full_epoch = 202

# ###################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
# # base_folders = [os.path.join(os.path.expanduser("~"), "Dropbox", "School", "Graduate", "CERN", "Temp_Files", "nonlineardynamics", "ameir_PINNs", "cylinder_cell", "06012024_adam_datalossonly_infinite", "vtk_output_2740"), os.path.join(os.path.expanduser("~"), "Dropbox", "School", "Graduate", "CERN", "Temp_Files", "nonlineardynamics", "ameir_PINNs", "cylinder_cell", "16012024_adam_datalossonly_infinite_continued_test", "vtk_output_28280")]
# # base_folders = [os.path.join('Z:\\', "cylinder_cell", "06012024_adam_datalossonly_infinite", "vtk_output_2740"), os.path.join('Z:\\', "cylinder_cell", "16012024_adam_datalossonly_infinite_continued_test", "vtk_output_28280")]
# base_folders = [os.path.join('Z:\\', "cylinder_cell", "16012024_adam_datalossonly_infinite_continued_test", f"vtk_output_{cylindercell_epoch}")]
# slice_params = [['xy', np.array([0, 0, 1]), np.array([0, 0, -1]), (500, 535, 50), (0, 1, 0), 250, 23], ['yz', np.array([1, 0, 0]), np.array([0, 0, -1]), (500, 570, 50), (0, 0, 1), 250, 23]]
# ###################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################

##################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
base_folders = [os.path.join('Z:\\', "ladefense", "data", f"wtf")]
# base_folders = [os.path.join(os.path.expanduser("~"), "Dropbox", "School", "Graduate", "CERN", "Temp_Files", "nonlineardynamics", "ameir_PINNs", "ladefense", "19022024_adam_datalossonly_sampled_70_70_10_infinite_ladefense", "vtk_output_694")]
# base_folders = [os.path.join(os.path.expanduser("~"), "Dropbox", "School", "Graduate", "CERN", "Temp_Files", "nonlineardynamics", "ameir_PINNs", "ladefense", "16012024_adam_datalossonly_infinite", "vtk_output_176")]
# base_folders = [os.path.join(os.path.expanduser("~"), "Dropbox", "School", "Graduate", "CERN", "Temp_Files", "nonlineardynamics", "ameir_PINNs", "ladefense", "19022024_adam_datalossonly_sampled_70_70_10_infinite_ladefense", "vtk_output_694"), os.path.join(os.path.expanduser("~"), "Dropbox", "School", "Graduate", "CERN", "Temp_Files", "nonlineardynamics", "ameir_PINNs", "ladefense", "16012024_adam_datalossonly_infinite", "vtk_output_176")]
# slice_params = [['xy', np.array([0, 0, 1]), np.array([0, 0, -1]), (0, 0, 5), (0, 1, 0), 2000], ['xy', np.array([0, 0, 1]), np.array([0, 0, -1]), (0, 0, 10), (0, 1, 0), 2000], ['xy', np.array([0, 0, 1]), np.array([0, 0, -1]), (0, 0, 15), (0, 1, 0), 2000], ['xy', np.array([0, 0, 1]), np.array([0, 0, -1]), (0, 0, 20), (0, 1, 0), 2000], ['xy', np.array([0, 0, 1]), np.array([0, 0, -1]), (0, 0, 25), (0, 1, 0), 2000], ['xy', np.array([0, 0, 1]), np.array([0, 0, -1]), (0, 0, 30), (0, 1, 0), 2000], ['xy', np.array([0, 0, 1]), np.array([0, 0, -1]), (0, 0, 35), (0, 1, 0), 2000], ['xy', np.array([0, 0, 1]), np.array([0, 0, -1]), (0, 0, 40), (0, 1, 0), 2000], ['xy', np.array([0, 0, 1]), np.array([0, 0, -1]), (0, 0, 45), (0, 1, 0), 2000], ['xy', np.array([0, 0, 1]), np.array([0, 0, -1]), (0, 0, 50), (0, 1, 0), 2000], ['yz', np.array([1, 0, 0]), np.array([0, 0, -1]), (0, 0, 5), (0, 0, 1), 2000]]
slice_params = [['xy', np.array([0, 0, 1]), np.array([0, 0, -1]), (0, 75, 50), (0, 1, 0), 2250, 100], ['yz', np.array([1, 0, 0]), np.array([0, 0, -1]), (0, 100, 270), (0, 0, 1), 1000, 50]]
##################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################

parameters_block = [['Velocity', 'Velocity_Predicted']]
all_physics = ['Sampled']

BIGGER_SIZE = 20
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def get_filenames_from_folder(path, extension, startname):
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith(extension) and f.startswith(startname)]

def join_plots(image1_path, image2_path, savename, wind_angle, figsize, dpi):
    image1 = mpimg.imread(image1_path)
    image2 = mpimg.imread(image2_path)

    fig, axs = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    axs[0].imshow(image1)
    axs[0].axis('off')
    axs[0].set_title('Actual CFD Data')
    axs[1].imshow(image2)
    axs[1].axis('off')  # Hide the axes
    axs[1].set_title('(PI)NN Prediction')  # Caption for the second image
    plt.suptitle(f'Actual CFD Data vs (PI)NN Prediction for Wind Angle = {wind_angle}')

    # fig, axs = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    # axs[0].imshow(image1)
    # axs[0].axis('off')
    # axs[0].set_title('Actual CFD Data')
    # axs[1].imshow(image2)
    # axs[1].axis('off')  # Hide the axes
    # axs[1].set_title('(PI)NN Validation CFD Data')  # Caption for the second image
    # plt.suptitle(f'Actual CFD Data vs (PI)NN Validation CFD Data for Wind Angle = {wind_angle}')

    plt.tight_layout()
    plt.savefig(savename)
    plt.close()

def rotate_vector(normal_vector, axis, theta_deg):
    """
    Rotate a vector around an arbitrary axis by a given angle.
    
    Parameters:
    - normal_vector: The initial normal vector as a numpy array.
    - axis: The axis of rotation as a unit vector (numpy array).
    - theta_deg: The rotation angle in degrees.
    
    Returns:
    - Rotated vector as a numpy array.
    """
    # Convert angle from degrees to radians
    theta_rad = np.radians(theta_deg)
    
    # Normalize the rotation axis
    axis = axis / np.linalg.norm(axis)
    
    # Compute components of the rotation axis
    ux, uy, uz = axis
    
    # Skew-symmetric matrix for the rotation axis
    u_cross = np.array([[0, -uz, uy],
                        [uz, 0, -ux],
                        [-uy, ux, 0]])
    
    # Outer product of axis
    uuT = np.outer(axis, axis)
    
    # Identity matrix
    I = np.eye(3)
    
    # Rodrigues' rotation formula
    R = np.cos(theta_rad) * I + np.sin(theta_rad) * u_cross + (1 - np.cos(theta_rad)) * uuT
    
    # Rotate the normal vector
    rotated_vector = np.dot(R, normal_vector)
    
    return rotated_vector

def base_plotting(base_folder, filename, param, plane, plane_point, image_res, rotation_vector, cam_pos, cam_foc, cam_up, mask_points):
    pvs.ResetSession()
    data_source = pvs.OpenDataFile(os.path.join(base_folder,filename))
    renderView1 = pvs.GetActiveViewOrCreate('RenderView')
    renderView1.ViewSize = image_res
    cell_to_point = pvs.CellDatatoPointData(Input=data_source)
    sliceFilter = pvs.Slice(Input=cell_to_point)
    sliceFilter.SliceType = 'Plane'
    sliceFilter.SliceOffsetValues = [0.0]
    sliceFilter.SliceType.Origin = plane_point #Slice in the middle of the sphere
    normal_vector_slice = list(rotation_vector)
    sliceFilter.SliceType.Normal = normal_vector_slice
    surfaceVectors = pvs.SurfaceVectors(Input=sliceFilter)
    surfaceVectors.SelectInputVectors = ['POINTS', param]
    maskPoints = pvs.MaskPoints(Input=surfaceVectors)
    maskPoints.OnRatio = mask_points
    streamTracer = pvs.StreamTracerWithCustomSource(Input=surfaceVectors, SeedSource=maskPoints)
    streamTracer.Vectors = ['POINTS', param]  # Adjust 'Velocity' based on your vector field name
    streamTracer.MaximumStreamlineLength = 1000.0  # Adjust based on your dataset
    surfaceVectorsDisplay = pvs.Show(surfaceVectors, renderView1)
    pvs.ColorBy(surfaceVectorsDisplay, ('POINTS', param), True)
    surfaceVectorsDisplay.SetScalarBarVisibility(renderView1, True)
    streamTracerDisplay = pvs.Show(streamTracer, renderView1)
    pvs.ColorBy(streamTracerDisplay, ('POINTS', param), True)
    streamlineLUT = pvs.GetColorTransferFunction(f'{param}')
    streamlineLUT.ApplyPreset('X Ray', True)  # Choose a preset or customize your color map
    streamTracerDisplay.LookupTable = streamlineLUT
    streamTracerDisplay.SetScalarBarVisibility(renderView1, False)
    pvs.Render(renderView1)
    camera = renderView1.GetActiveCamera()
    camera.SetPosition(cam_pos)  # Position the camera wayyyy above the slice.
    camera.SetFocalPoint(cam_foc)  # Focus on the center of the slice.
    camera.SetViewUp(cam_up)  # Adjust view-up vector. view up is y since we are in the x-y plane.
    pvs.Render()
    screenshot_path = os.path.join(base_folder, f"slice_streamTracer_visualization_{param}_{plane}_{plane_point}_{filename[:-4]}.png")
    pvs.SaveScreenshot(screenshot_path, renderView1)
    state_path = os.path.join(base_folder, f"visualization_state_slice_streamTracer_{param}_{plane}_{plane_point}_{filename[:-4]}.pvsm")
    pvs.SaveState(state_path)

    return screenshot_path

def plot_paraview_predictions_streamtracer(base_folder, physics, parameters, slice_param):
    filenames = get_filenames_from_folder(base_folder, '.vtu', physics)
    for filename in filenames:

        wind_angle = int(filename.split('_')[-1].split('.')[0])

        plane, normal_vector, axis, plane_point, cam_up, cam_dist, mask_points = slice_param[0], slice_param[1], slice_param[2], slice_param[3], slice_param[4], slice_param[5], slice_param[6]
        rotation_vector = rotate_vector(normal_vector, axis, wind_angle)

        camera_distance = cam_dist  # Distance from the plane to the camera
        cam_pos = plane_point + (rotation_vector * camera_distance)
        cam_foc = plane_point
        image_res = [3840, 2160]

        image_actual = base_plotting(base_folder, f"Original_{filename}", parameters[0], plane, plane_point, image_res, rotation_vector, cam_pos, cam_foc, cam_up, mask_points)
        image_nn = base_plotting(base_folder, filename, parameters[1], plane, plane_point, image_res, rotation_vector, cam_pos, cam_foc, cam_up, mask_points)

        merged_image = os.path.join(base_folder, f"MERGED_slice_streamTracer_visualization_{plane}_{parameters[0]}_{parameters[1]}_{plane_point}_{filename[:-4]}.png")
        total_pixel_width = image_res[0]*2 # for two images side by side
        pixel_height = image_res[1]
        dpi = 300
        error_factor = 1
        figure_width_in_inches = total_pixel_width / dpi + error_factor
        figure_height_in_inches = pixel_height / dpi + error_factor
        figsize = (figure_width_in_inches, figure_height_in_inches)
        join_plots(image_actual, image_nn, merged_image, wind_angle, figsize, dpi)

for base_folder in base_folders:
    for physics in all_physics:
        for parameters in parameters_block:
            for slice_param in slice_params:
                plot_paraview_predictions_streamtracer(base_folder, physics, parameters, slice_param)