from definitions import *
from config import config

BIGGER_SIZE = 20
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def normalizing_details():
    x_length = config["training"]["boundary"][0][1] - config["training"]["boundary"][0][0]
    y_length = config["training"]["boundary"][0][3] - config["training"]["boundary"][0][2]
    z_length = config["training"]["boundary"][0][5] - config["training"]["boundary"][0][4]
    return x_length, y_length, z_length

def plotting_details(z_actual):
    mean_z = np.mean(z_actual)
    std_z = np.std(z_actual)
    vmin = mean_z - 2 * std_z
    vmax = mean_z + 2 * std_z
    vmin = float(vmin)
    vmax = float(vmax)
    # cmap = plt.cm.RdBu_r
    cmap = 'jet'
    scatter_size = 5
    # ticks = np.arange(vmin,vmax,abs(vmax-vmin)/10)
    ticks = None
    levels = 128
    return vmin, vmax, levels, cmap, scatter_size, ticks

def normalize_grids(plane, points1, points2):
    x_length, y_length, z_length = normalizing_details()
    points1 = np.array(points1)
    points2 = np.array(points2)
    if config["data"]["geometry"] == 'ladefense.stl':
        x_length = x_length/2
        y_length = y_length/2
        if plane == 'X-Z':
            points1 = (points1)/x_length
            points2 = (points2)/z_length
        if plane == 'Y-Z':
            points1 = (points1)/y_length
            points2 = (points2)/z_length
        if plane == 'X-Y':
            points1 = (points1)/x_length
            points2 = (points2)/y_length
    else:
        x_length = x_length/2
        y_length = y_length/2
        z_length = z_length*0.3
        if plane == 'X-Z':
            points1 = (points1 - x_length)/z_length
            points2 = (points2)/z_length
        if plane == 'Y-Z':
            points1 = (points1 - y_length)/z_length
            points2 = (points2)/z_length
        if plane == 'X-Y':
            points1 = (points1 - x_length)/z_length
            points2 = (points2 - y_length)/z_length
    return points1, points2

def normalize_cut_tolerance(plane, cut, tolerance):
    x_length, y_length, z_length = normalizing_details()
    if plane == 'X-Z':
        cut = (cut - y_length)/z_length
        tolerance = (tolerance - y_length)/z_length
    if plane == 'Y-Z':
        cut = (cut - x_length)/z_length
        tolerance = (tolerance - x_length)/z_length
    if plane == 'X-Y':
        cut = (cut)/z_length
        tolerance = (tolerance)/z_length
    return cut, tolerance

def get_plane_config(plane):
    coordinate3 = {'X-Z': 'Y', 'Y-Z': 'X', 'X-Y': 'Z'}[plane]
    coordinate1 = plane.split('-')[0]
    coordinate2 = plane.split('-')[1]
    if plane == 'X-Z':
        lim_min1, lim_max1 = config["plotting"]["lim_min_max"][0]
        lim_min2, lim_max2 = config["plotting"]["lim_min_max"][2]
    elif plane == 'Y-Z':
        lim_min1, lim_max1 = config["plotting"]["lim_min_max"][1]
        lim_min2, lim_max2 = config["plotting"]["lim_min_max"][2] 
    elif plane == 'X-Y':
        lim_min1, lim_max1 = config["plotting"]["lim_min_max"][0]
        lim_min2, lim_max2 = config["plotting"]["lim_min_max"][1] 
    return coordinate1, coordinate2, coordinate3, lim_min1, lim_max1, lim_min2, lim_max2  

def filter_dataframe(config, df, wind_angle, noplotdata, cut, tolerance):

    def extract_dataset_by_wind_angle(df, dataset_size, lower_bound, upper_bound):
        for start_idx in range(0, len(df), dataset_size):
            if df.iloc[start_idx]['WindAngle'] >= lower_bound and df.iloc[start_idx]['WindAngle'] <= upper_bound:
                return df.iloc[start_idx:start_idx+dataset_size]
        return pd.DataFrame()

    def get_dataset_size(config):
        chosen_machine_key = config["chosen_machine"]
        datafolder_path = os.path.join(config["machine"][chosen_machine_key], config["data"]["data_folder_name"])
        filenames = get_filenames_from_folder(datafolder_path, config["data"]["extension"], config["data"]["startname_data"])
        for filename in sorted(filenames):
            df = pd.read_csv(os.path.join(datafolder_path, filename))
            return len(df)
            break

    lower_bound = wind_angle - 5
    upper_bound = wind_angle + 5
    
    df_filtered = extract_dataset_by_wind_angle(df, get_dataset_size(config), lower_bound, upper_bound)
    df_filtered = df_filtered[(df_filtered[noplotdata] >= cut - tolerance) & (df_filtered[noplotdata] <= cut + tolerance)]
    return df_filtered

def define_scatter_grid(filtered_df, points1, points2):
    grid1, grid2 = filtered_df[points1], filtered_df[points2]
    return grid1, grid2

def define_grid(filtered_df, points1, points2):
    grid1, grid2 = np.mgrid[filtered_df[points1].min():filtered_df[points1].max():1000j, filtered_df[points2].min():filtered_df[points2].max():1000j]
    return grid1, grid2

def interpolate_to_grid(grid1, grid2, xy, z):
    return griddata(xy, z, (grid1, grid2), method='linear', fill_value=0)

def get_all_grids(grid1, grid2, points1, points2, filtered_df,actual=None):
    if actual is None:
        grid_vx = interpolate_to_grid(grid1, grid2, filtered_df[[points1, points2]].values, filtered_df['Velocity_X'].values)
        grid_vy = interpolate_to_grid(grid1, grid2, filtered_df[[points1, points2]].values, filtered_df['Velocity_Y'].values)
        grid_vz = interpolate_to_grid(grid1, grid2, filtered_df[[points1, points2]].values, filtered_df['Velocity_Z'].values)
        grid_magnitude = np.sqrt(grid_vx**2 + grid_vy**2 + grid_vz**2)
        grid_pressure = interpolate_to_grid(grid1, grid2, filtered_df[[points1, points2]].values, filtered_df['Pressure'].values)
        grid_turbvisc = interpolate_to_grid(grid1, grid2, filtered_df[[points1, points2]].values, filtered_df['TurbVisc'].values)
        return grid_vx, grid_vy, grid_vz, grid_magnitude, grid_pressure, grid_turbvisc
    else:
        grid_vx = interpolate_to_grid(grid1, grid2, filtered_df[[points1, points2]].values, filtered_df[f'Velocity_X_{actual}'].values)
        grid_vy = interpolate_to_grid(grid1, grid2, filtered_df[[points1, points2]].values, filtered_df[f'Velocity_Y_{actual}'].values)
        grid_vz = interpolate_to_grid(grid1, grid2, filtered_df[[points1, points2]].values, filtered_df[f'Velocity_Z_{actual}'].values)
        grid_magnitude = np.sqrt(grid_vx**2 + grid_vy**2 + grid_vz**2)
        grid_pressure = interpolate_to_grid(grid1, grid2, filtered_df[[points1, points2]].values, filtered_df[f'Pressure_{actual}'].values)
        grid_turbvisc = interpolate_to_grid(grid1, grid2, filtered_df[[points1, points2]].values, filtered_df[f'TurbVisc_{actual}'].values)
        return grid_vx, grid_vy, grid_vz, grid_magnitude, grid_pressure, grid_turbvisc

def get_velocity_grids(grid1, grid2, points1, points2, filtered_df,actual=None):
    if actual is None:
        grid_vx = interpolate_to_grid(grid1, grid2, filtered_df[[points1, points2]].values, filtered_df['Velocity_X'].values)
        grid_vy = interpolate_to_grid(grid1, grid2, filtered_df[[points1, points2]].values, filtered_df['Velocity_Y'].values)
        grid_vz = interpolate_to_grid(grid1, grid2, filtered_df[[points1, points2]].values, filtered_df['Velocity_Z'].values)
        grid_magnitude = np.sqrt(grid_vx**2 + grid_vy**2 + grid_vz**2)
        return grid_vx, grid_vy, grid_vz, grid_magnitude
    else:
        grid_vx = interpolate_to_grid(grid1, grid2, filtered_df[[points1, points2]].values, filtered_df[f'Velocity_X_{actual}'].values)
        grid_vy = interpolate_to_grid(grid1, grid2, filtered_df[[points1, points2]].values, filtered_df[f'Velocity_Y_{actual}'].values)
        grid_vz = interpolate_to_grid(grid1, grid2, filtered_df[[points1, points2]].values, filtered_df[f'Velocity_Z_{actual}'].values)
        grid_magnitude = np.sqrt(grid_vx**2 + grid_vy**2 + grid_vz**2)
        return grid_vx, grid_vy, grid_vz, grid_magnitude

def make_quiver_grid(plane, grid1, grid2, grid_vx, grid_vy, grid_vz, stride):
    quiver_points1 = grid1[::stride, ::stride]
    quiver_points2 = grid2[::stride, ::stride]
    quiver_vx = grid_vx[::stride, ::stride]
    quiver_vy = grid_vy[::stride, ::stride]
    quiver_vz = grid_vz[::stride, ::stride]
    if plane == 'X-Z':
        quiver_v1 = quiver_vx
        quiver_v2 = quiver_vz
    elif plane == 'Y-Z':
        quiver_v1 = quiver_vy
        quiver_v2 = quiver_vz
    elif plane == 'X-Y':
        quiver_v1 = quiver_vx
        quiver_v2 = quiver_vy
    return quiver_points1, quiver_points2, quiver_v1, quiver_v2

def make_arrow(plane,x_coord,y_coord,wind_angle):
    arrow_x1 = x_coord
    arrow_y1 = y_coord
    length = 10000000
    wind_angle_modf = 270 - wind_angle
    wind_angle_rad_modf = np.deg2rad(wind_angle_modf)
    arrow_dx1 = length*(np.cos(wind_angle_rad_modf))
    arrow_dy1 = length*(np.sin(wind_angle_rad_modf))
    arrow_x1, arrow_y1 = normalize_grids(plane, arrow_x1, arrow_y1)
    arrow_dx1, arrow_dy1 = normalize_grids(plane, arrow_dx1, arrow_dy1)
    arrow = patches.Arrow(arrow_x1, arrow_y1, arrow_dx1, arrow_dy1, width=0.05, color="white")
    return arrow

def scale_geometry(filename, scaled_filename):
    your_mesh = mesh.Mesh.from_file(filename)
    your_mesh.vectors *= 1/1000
    your_mesh.save(scaled_filename)

def get_geometry(plane, filename):
    your_mesh = mesh.Mesh.from_file(filename)
    vectors = your_mesh.vectors
    return vectors

def plot_geometry(plane, geometry, scatter_size, ax):
    if config["data"]["geometry"] == 'scaled_cylinder_sphere.stl':
        flattened = geometry.reshape(-1, 3)
        x, y, z = flattened[:, 0], flattened[:, 1], flattened[:, 2]
        if plane == 'X-Z':
            g1, g2 = normalize_grids(plane, x, z)
        elif plane == 'Y-Z':
            g1, g2 = normalize_grids(plane, y, z)
        elif plane == 'X-Y':
            g1, g2 = normalize_grids(plane, x, y)
        ax.scatter(g1, g2, c='black', s=scatter_size, label='Geometry')
    elif config["data"]["geometry"] == 'ladefense.stl':
        for triangle in geometry:
            x = [point[0] for point in triangle] + [triangle[0][0]] # Closing the loop
            y = [point[1] for point in triangle] + [triangle[0][1]] # Closing the loop
            z = [point[2] for point in triangle] + [triangle[0][2]] # Closing the loop
            if plane == 'X-Y':
                g1, g2 = normalize_grids(plane, x, y)
            if plane == 'X-Z':
                g1, g2 = normalize_grids(plane, x, z)
            if plane == 'Y-Z':
                g1, g2 = normalize_grids(plane, y, z)
            ax.plot(g1, g2, color='black')