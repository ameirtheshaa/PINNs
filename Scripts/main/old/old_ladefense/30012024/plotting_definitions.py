from definitions import *

def normalizing_details():
    x_length = 2520
    y_length = 2520
    z_length = 1000
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
    # ticks = int(vmax-vmin)+1
    ticks = 5
    levels = 128
    return vmin, vmax, levels, cmap, scatter_size, ticks

def normalize_grids(plane, points1, points2):
    x_length, y_length, z_length = normalizing_details()
    points1 = np.array(points1)
    points2 = np.array(points2)
    if plane == 'X-Z':
        points1 = (points1)/x_length
        points2 = (points2)/z_length
    if plane == 'Y-Z':
        points1 = (points1)/y_length
        points2 = (points2)/z_length
    if plane == 'X-Y':
        points1 = (points1)/x_length
        points2 = (points2)/y_length
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
    if plane in ['X-Z', 'Y-Z']:   
        lim_min1, lim_max1 = (-1, 1)
        lim_min2, lim_max2 = (0, 1)
    elif plane == 'X-Y':
        lim_min1, lim_max1 = (-0.3,0.3)
        lim_min2, lim_max2 = (-0.3,0.3) 
    return coordinate1, coordinate2, coordinate3, lim_min1, lim_max1, lim_min2, lim_max2  

def filter_dataframe(df, wind_angle, noplotdata, cut, tolerance):
    lower_bound = wind_angle - 2
    upper_bound = wind_angle + 2
    mask = df['WindAngle'].between(lower_bound, upper_bound)
    df_filtered = df.loc[mask]
    df_filtered = df_filtered[(df_filtered[noplotdata] >= cut - tolerance) & (df_filtered[noplotdata] <= cut + tolerance)]
    return df_filtered

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
    vertices = []
    for triangle in your_mesh.vectors:
        vertices.append(triangle)
    return vertices