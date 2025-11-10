from definitions import *
from plotting import *

def plot_div_new_angles(df,angle,datafolder_path,savename_total_div, savename_total_velocity,plane,cut,tolerance):
    wind_angle = angle
    if plane == 'X-Z':
        points = ['X', 'Z']
        noplotdata = 'Y'
        coordinate3 = 'Y'
        lim_min1, lim_max1 = (-1,1)
        lim_min2, lim_max2 = (0,1)
    if plane == 'Y-Z':
        points = ['Y', 'Z']
        noplotdata = 'X'
        coordinate3 = 'X'
        lim_min1, lim_max1 = (-1,1)
        lim_min2, lim_max2 = (0,1)
    if plane == 'X-Y':
        points = ['X', 'Y']
        noplotdata = 'Z'
        coordinate3 = 'Z'
        lim_min1, lim_max1 = (-1,1)
        lim_min2, lim_max2 = (-1,1)

    points1 = points[0]
    points2 = points[1]

    coordinate1 = plane.split('-')[0]
    coordinate2 = plane.split('-')[1]

    # Filter the data to focus on the y-z plane/...
    filtered_df = df[(df[noplotdata] >= cut - tolerance) & (df[noplotdata] <= cut + tolerance)]

    # Define a regular grid covering the range of y and z coordinates/...
    grid1, grid2 = np.mgrid[filtered_df[points1].min():filtered_df[points1].max():1000j, filtered_df[points2].min():filtered_df[points2].max():1000j]

    grid_vx = griddata(filtered_df[[points1, points2]].values, filtered_df['Velocity_X'].values, (grid1, grid2), method='linear', fill_value=0)
    grid_vy = griddata(filtered_df[[points1, points2]].values, filtered_df['Velocity_Y'].values, (grid1, grid2), method='linear', fill_value=0)
    grid_vz = griddata(filtered_df[[points1, points2]].values, filtered_df['Velocity_Z'].values, (grid1, grid2), method='linear', fill_value=0)

    # Calculate the total magnitude of the velocity on the grid
    grid_magnitude = np.sqrt(grid_vx**2 + grid_vy**2 + grid_vz**2)

    spherefill1, spherefill2, cylinderfill1, cylinderfill2 = geometry_coordinates(plane, datafolder_path)

    grid1, grid2 = normalize_grids(plane, grid1, grid2)
    spherefill1, spherefill2 = normalize_grids(plane, spherefill1, spherefill2)
    cylinderfill1, cylinderfill2 = normalize_grids(plane, cylinderfill1, cylinderfill2)
    
    cut, tolerance = normalize_cut_tolerance(plane, cut, tolerance)

    def calculate_derivatives(grid_v, d, axis):
        return (np.roll(grid_v, -1, axis=axis) - np.roll(grid_v, 1, axis=axis)) / (2 * d)

    dx = grid1[1, 0] - grid1[0, 0]
    dy = grid2[0, 1] - grid2[0, 0]

    if plane == 'X-Z':
        dvx_dx = calculate_derivatives(grid_vx, dx, axis=0)  # Derivative with respect to X
        dvz_dz = calculate_derivatives(grid_vz, dy, axis=1)  # Derivative with respect to Z
        divergence = dvx_dx + dvz_dz
    elif plane == 'Y-Z':
        dvy_dy = calculate_derivatives(grid_vy, dx, axis=0)  # Derivative with respect to Y
        dvz_dz = calculate_derivatives(grid_vz, dy, axis=1)  # Derivative with respect to Z
        divergence = dvy_dy + dvz_dz
    elif plane == 'X-Y':
        dvx_dx = calculate_derivatives(grid_vx, dx, axis=0)  # Derivative with respect to X
        dvy_dy = calculate_derivatives(grid_vy, dy, axis=1)  # Derivative with respect to Y
        divergence = dvx_dx + dvy_dy

    divergence_flat = divergence.flatten()

    # For Div Velocity
    fig, ax = plt.subplots(figsize=(16, 8), sharey=True)
    vmin, vmax, cmap, scatter_size, ticks = plotting_details(divergence_flat)
    contour = ax.contourf(grid1, grid2, divergence, levels=4, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
    ax.tricontourf(tri.Triangulation(cylinderfill1, cylinderfill2), np.zeros(len(cylinderfill1)), colors='black', alpha=1)  # Fill color set to black
    cylinder_patch = patches.Patch(color='black', label='Cylinder Fill')
    cbar = fig.colorbar(contour, ax=ax, ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
    cbar.set_label('Div Velocity', rotation=270, labelpad=15)
    ax.set_xlabel(f'{coordinate1} Coordinate')
    ax.set_ylabel(f'{coordinate2} Coordinate')
    ax.set_xlim(lim_min1, lim_max1) 
    ax.set_ylim(lim_min2, lim_max2)
    ax.set_title(f'Div V in the {plane} Plane for Wind Angle = {angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')
    plt.tight_layout()
    plt.savefig(savename_total_div)
    plt.close()

    # For Total Velocity
    fig, ax = plt.subplots(figsize=(16, 8), sharey=True)
    vmin, vmax, cmap, scatter_size, ticks = plotting_details(filtered_df['Velocity_Magnitude'].values)
    contour = ax.contourf(grid1, grid2, grid_magnitude, levels=128, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
    ax.tricontourf(tri.Triangulation(cylinderfill1, cylinderfill2), np.zeros(len(cylinderfill1)), colors='black', alpha=1)  # Fill color set to black
    cylinder_patch = patches.Patch(color='black', label='Cylinder Fill')
    cbar = fig.colorbar(contour, ax=ax, ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
    cbar.set_label('Velocity Magnitude', rotation=270, labelpad=15)
    ax.set_xlabel(f'{coordinate1} Coordinate')
    ax.set_ylabel(f'{coordinate2} Coordinate')
    ax.set_xlim(lim_min1, lim_max1) 
    ax.set_ylim(lim_min2, lim_max2)
    ax.set_title(f'Total Velocity in the {plane} Plane for Wind Angle = {angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')
    plt.tight_layout()
    plt.savefig(savename_total_velocity)
    plt.close()

def plot_div_angles_2d(df,angle,datafolder_path):
    params = [['X-Z',570,5],['Y-Z',500,5],['X-Y',50,5]]
    plot_folder = os.path.join('D:\\', "Dropbox", "datadivplots")
    for j in params:
        plane = j[0]
        cut = j[1]
        tolerance = j[2]
        savename_total_velocity = os.path.join(plot_folder,f'{plane}_totalvelocity_{angle}.png')
        savename_total_div = os.path.join(plot_folder,f'{plane}_divvelocity_{angle}.png')
        plot_div_new_angles(df,angle,datafolder_path,savename_total_div,savename_total_velocity,plane,cut,tolerance)


datafolder_path = os.path.join('D:\\', "Dropbox", "data")
filenames = get_filenames_from_folder(datafolder_path, '.csv', 'CFD')

print (filenames)

for filename in sorted(filenames):
    df = pd.read_csv(os.path.join(datafolder_path, filename))
    print (filename)
    wind_angle = int(filename.split('_')[4].split('.')[0])  # Extract the index part of the filename

    print (wind_angle)
    
    # Add new columns with unique values for each file
    df['cos(WindAngle)'] = (np.cos(np.deg2rad(wind_angle)))
    df['sin(WindAngle)'] = (np.sin(np.deg2rad(wind_angle)))

    df.rename(columns={'Points:0': 'X', 'Points:1': 'Y', 'Points:2': 'Z', 'Velocity:0': 'Velocity_X', 'Velocity:1': 'Velocity_Y', 'Velocity:2': 'Velocity_Z'}, inplace=True)
    
    df['Velocity_Magnitude'] = np.sqrt(df['Velocity_X']**2 + 
                                                df['Velocity_Y']**2 + 
                                                df['Velocity_Z']**2)

    plot_div_angles_2d(df,wind_angle,datafolder_path)