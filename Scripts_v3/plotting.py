from definitions import *
from training import *
from PINN import *

def plotting_details(z_actual):
    mean_z = np.mean(z_actual)
    std_z = np.std(z_actual)
    vmin = mean_z - 2 * std_z
    vmax = mean_z + 2 * std_z

    vmin = int(vmin)
    vmax = int(vmax)

    cmap = plt.cm.RdBu_r
    scatter_size = 5
    ticks = 5

    return vmin, vmax, cmap, scatter_size, ticks

def normalize_grids(plane, points1, points2):
    if plane == 'X-Z':
        points1 = (points1 - 500)/300
        points2 = (points2)/300
    if plane == 'Y-Z':
        points1 = (points1 - 500)/300
        points2 = (points2)/300
    if plane == 'X-Y':
        points1 = (points1 - 500)/300
        points2 = (points2 - 500)/300

    return points1, points2

def normalize_cut_tolerance(plane, cut, tolerance):
    if plane == 'X-Z':
        cut = (cut - 500)/300
        tolerance = (tolerance - 500)/300
    if plane == 'Y-Z':
        cut = (cut - 500)/300
        tolerance = (tolerance - 500)/300
    if plane == 'X-Y':
        cut = (cut)/300
        tolerance = (tolerance)/300

    return cut, tolerance

def geometry_coordinates(plane):
    # Generate scatter points within the sphere's volume
    phi = np.linspace(0, 2*np.pi, 100)
    theta = np.linspace(0, np.pi, 100)
    theta, phi = np.meshgrid(theta, phi)
    sphere_center_x = 500
    sphere_radius_x = 5
    sphere_center_y = 500
    sphere_radius_y = 5
    sphere_center_z = 50
    sphere_radius_z = 5
    theta_min = 0
    theta_max = np.pi
    phi_min = 0
    phi_max = 2 * np.pi
    r_min = 0  # Minimum radius
    r_max = 5  # Maximum radius (sphere's radius)
    num_points = 1000  # Number of scatter points
    theta_values = np.random.uniform(theta_min, theta_max, num_points)
    phi_values = np.random.uniform(phi_min, phi_max, num_points)
    r_values = np.random.uniform(r_min, r_max, num_points)
    x_sphere_fill = sphere_center_x + r_values * np.sin(theta_values) * np.cos(phi_values)
    y_sphere_fill = sphere_center_y + r_values * np.sin(theta_values) * np.sin(phi_values)
    z_sphere_fill = sphere_center_z + r_values * np.cos(theta_values)

    # Generate scatter points for the cylinder body
    cylinder_center_x = 500
    cylinder_radius_x = 7.5
    cylinder_center_y = 570
    cylinder_radius_y = 7.5
    cylinder_height = 65

    # Generate scatter points for the cylinder's rounded cap
    z_cap_corrected = np.linspace(cylinder_height - 1, cylinder_height, 100)
    
    if plane == 'Y-Z':
        y_cylinder_body_fill, z_cylinder_body_fill = np.meshgrid(np.linspace(cylinder_center_y - cylinder_radius_y, cylinder_center_y + cylinder_radius_y, 100), np.linspace(0, cylinder_height, 100))
        y_cylinder_cap, z_cylinder_cap_corrected = np.meshgrid(np.linspace(cylinder_center_y - cylinder_radius_y, cylinder_center_y + cylinder_radius_y, 100),  z_cap_corrected)
        return y_sphere_fill, z_sphere_fill, y_cylinder_body_fill, z_cylinder_body_fill, y_cylinder_cap, z_cylinder_cap_corrected
    if plane == 'X-Z':
        x_cylinder_body_fill, z_cylinder_body_fill = np.meshgrid(np.linspace(cylinder_center_x - cylinder_radius_x, cylinder_center_x + cylinder_radius_x, 100), np.linspace(0, cylinder_height, 100))
        x_cylinder_cap, z_cylinder_cap_corrected = np.meshgrid(np.linspace(cylinder_center_x - cylinder_radius_x, cylinder_center_x + cylinder_radius_x, 100),  z_cap_corrected)
        return x_sphere_fill, z_sphere_fill, x_cylinder_body_fill, z_cylinder_body_fill, x_cylinder_cap, z_cylinder_cap_corrected
    if plane == 'X-Y':
        x_cylinder_body_fill, y_cylinder_body_fill = np.meshgrid(np.linspace(cylinder_center_x - cylinder_radius_x, cylinder_center_x + cylinder_radius_x, 100), np.linspace(cylinder_center_y - cylinder_radius_y, cylinder_center_y + cylinder_radius_y, 100))
        x_cylinder_cap, y_cylinder_cap = np.meshgrid(np.linspace(cylinder_center_x - cylinder_radius_x, cylinder_center_x + cylinder_radius_x, 100),  np.linspace(cylinder_center_y - cylinder_radius_y, cylinder_center_y + cylinder_radius_y, 100))
        return x_sphere_fill, y_sphere_fill, x_cylinder_body_fill, y_cylinder_body_fill, x_cylinder_cap, y_cylinder_cap

def plot_data(filename,angle,savename,plane,cut,tolerance,cmap):
    
    df = pd.read_csv(filename)

    if plane == 'X-Z':
        points = ['Points:0', 'Points:2']
        noplotdata = 'Points:1'
        coordinate3 = 'Y'
        lim_min1, lim_max1 = (-1,1)
        lim_min2, lim_max2 = (0,1)
    if plane == 'Y-Z':
        points = ['Points:1', 'Points:2']
        noplotdata = 'Points:0'
        coordinate3 = 'X'
        lim_min1, lim_max1 = (-1,1)
        lim_min2, lim_max2 = (0,1)
    if plane == 'X-Y':
        points = ['Points:0', 'Points:1']
        noplotdata = 'Points:2'
        coordinate3 = 'Z'
        lim_min1, lim_max1 = (-1,1)
        lim_min2, lim_max2 = (-1,1)

    points1 = points[0]
    points2 = points[1]

    coordinate1 = plane.split('-')[0]
    coordinate2 = plane.split('-')[1]

    # Filter the data to focus on the y-z plane
    filtered_df = df[(df[noplotdata] >= cut - tolerance) & (df[noplotdata] <= cut + tolerance)]

    # Define a regular grid covering the range of y and z coordinates
    grid1, grid2 = np.mgrid[filtered_df[points1].min():filtered_df[points1].max():1000j, filtered_df[points2].min():filtered_df[points2].max():1000j]

    # Interpolate all velocity components onto the grid
    grid_vx = griddata(filtered_df[[points1, points2]].values, filtered_df['Velocity:0'].values, (grid1, grid2), method='linear', fill_value=0)
    grid_vy = griddata(filtered_df[[points1, points2]].values, filtered_df['Velocity:1'].values, (grid1, grid2), method='linear', fill_value=0)
    grid_vz = griddata(filtered_df[[points1, points2]].values, filtered_df['Velocity:2'].values, (grid1, grid2), method='linear', fill_value=0)

    # Calculate the total magnitude of the velocity on the grid
    grid_magnitude = np.sqrt(grid_vx**2 + grid_vy**2 + grid_vz**2)

    spherefill1, spherefill2, cylinderfill1, cylinderfill2, cylindercapfill1, cylindercapfill2 = geometry_coordinates(plane)


    grid1, grid2 = normalize_grids(plane, grid1, grid2)
    spherefill1, spherefill2 = normalize_grids(plane, spherefill1, spherefill2)
    cylinderfill1, cylinderfill2 = normalize_grids(plane, cylinderfill1, cylinderfill2)
    cylindercapfill1, cylindercapfill2 = normalize_grids(plane, cylindercapfill1, cylindercapfill2)
    cut, tolerance = normalize_cut_tolerance(plane, cut, tolerance)

    # Visualize using matplotlib
    fig, ax = plt.subplots(figsize=(12, 8))
    contour = ax.contourf(grid1, grid2, grid_magnitude, levels=128, cmap=cmap)
    ax.scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
    ax.scatter(cylinderfill1.ravel(),  cylinderfill2.ravel(), c='black', s=1, label='Cylinder Body Fill')
    ax.scatter(cylindercapfill1.ravel(), cylindercapfill2, c='black', s=1, label='Cylinder Cap Fill')
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label('Velocity Magnitude', rotation=270, labelpad=15)
    ax.set_xlabel(f'{coordinate1} Coordinate')
    ax.set_ylabel(f'{coordinate2} Coordinate')
    ax.set_xlim(lim_min1, lim_max1) 
    ax.set_ylim(lim_min2, lim_max2)
    ax.set_title(f'Total Velocity in the {plane} Plane for Wind Angle = {angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')
    plt.tight_layout()
    plt.savefig(savename)
    plt.close()

def plot_data_predictions(df,angle,savename_all,savename_total_velocity,savename_vx,savename_vy,savename_vz,savename_pressure,plane,cut,tolerance,cmap):
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

    lower_bound = wind_angle - 2
    upper_bound = wind_angle + 2
    mask = df['WindAngle'].between(lower_bound, upper_bound)
    df = df.loc[mask]

    # Filter the data to focus on the y-z plane/...
    filtered_df = df[(df[noplotdata] >= cut - tolerance) & (df[noplotdata] <= cut + tolerance)]

    # Define a regular grid covering the range of y and z coordinates/...
    grid1, grid2 = np.mgrid[filtered_df[points1].min():filtered_df[points1].max():1000j, filtered_df[points2].min():filtered_df[points2].max():1000j]

    # Interpolate all velocity components onto the grid
    try:
        grid_vx_actual = griddata(filtered_df[[points1, points2]].values, filtered_df['Velocity_X_Actual'].values, (grid1, grid2), method='linear', fill_value=0)
        grid_vy_actual = griddata(filtered_df[[points1, points2]].values, filtered_df['Velocity_Y_Actual'].values, (grid1, grid2), method='linear', fill_value=0)
        grid_vz_actual = griddata(filtered_df[[points1, points2]].values, filtered_df['Velocity_Z_Actual'].values, (grid1, grid2), method='linear', fill_value=0)
        grid_pressure_actual = griddata(filtered_df[[points1, points2]].values, filtered_df['Pressure_Actual'].values, (grid1, grid2), method='linear', fill_value=0)
        grid_vx_pred = griddata(filtered_df[[points1, points2]].values, filtered_df['Velocity_X_Predicted'].values, (grid1, grid2), method='linear', fill_value=0)
        grid_vy_pred = griddata(filtered_df[[points1, points2]].values, filtered_df['Velocity_Y_Predicted'].values, (grid1, grid2), method='linear', fill_value=0)
        grid_vz_pred = griddata(filtered_df[[points1, points2]].values, filtered_df['Velocity_Z_Predicted'].values, (grid1, grid2), method='linear', fill_value=0)
        grid_pressure_pred = griddata(filtered_df[[points1, points2]].values, filtered_df['Pressure_Predicted'].values, (grid1, grid2), method='linear', fill_value=0)
    except:
        print(f"not enough points")
        return

    # Calculate the total magnitude of the velocity on the grid
    grid_magnitude_actual = np.sqrt(grid_vx_actual**2 + grid_vy_actual**2 + grid_vz_actual**2)
    grid_magnitude_pred = np.sqrt(grid_vx_pred**2 + grid_vy_pred**2 + grid_vz_pred**2)

    spherefill1, spherefill2, cylinderfill1, cylinderfill2, cylindercapfill1, cylindercapfill2 = geometry_coordinates(plane)

    grid1, grid2 = normalize_grids(plane, grid1, grid2)
    spherefill1, spherefill2 = normalize_grids(plane, spherefill1, spherefill2)
    cylinderfill1, cylinderfill2 = normalize_grids(plane, cylinderfill1, cylinderfill2)
    cylindercapfill1, cylindercapfill2 = normalize_grids(plane, cylindercapfill1, cylindercapfill2)
    cut, tolerance = normalize_cut_tolerance(plane, cut, tolerance)

    fig, axs = plt.subplots(5, 2, figsize=(16, 40), sharey=True)
    fig.suptitle(f'Comparison of Actual vs. Predicted values with Wind Angle = {wind_angle} in the {plane} Plane with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')

    # For Total Velocity
    contour_actual = axs[0,0].contourf(grid1, grid2, grid_magnitude_actual, levels=128, cmap=cmap)
    axs[0,0].scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
    axs[0,0].scatter(cylinderfill1.ravel(),  cylinderfill2.ravel(), c='black', s=1, label='Cylinder Body Fill')
    axs[0,0].scatter(cylindercapfill1.ravel(), cylindercapfill2, c='black', s=1, label='Cylinder Cap Fill')
    vmin, vmax, cmap_, scatter_size, ticks = plotting_details(filtered_df['Velocity_Magnitude_Actual'].values)
    cbar = fig.colorbar(contour_actual, ax=axs[0,0], ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
    cbar.set_label('Velocity Magnitude - Actual', rotation=270, labelpad=15)
    axs[0,0].set_xlabel(f'{coordinate1} Coordinate')
    axs[0,0].set_ylabel(f'{coordinate2} Coordinate')
    axs[0,0].set_xlim(lim_min1, lim_max1) 
    axs[0,0].set_ylim(lim_min2, lim_max2)
    axs[0,0].set_title(f'Total Actual Velocity in the {plane} Plane for Wind Angle = {angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')
    
    contour_pred = axs[0,1].contourf(grid1, grid2, grid_magnitude_pred, levels=128, cmap=cmap)
    axs[0,1].scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
    axs[0,1].scatter(cylinderfill1.ravel(),  cylinderfill2.ravel(), c='black', s=1, label='Cylinder Body Fill')
    axs[0,1].scatter(cylindercapfill1.ravel(), cylindercapfill2, c='black', s=1, label='Cylinder Cap Fill')
    cbar = fig.colorbar(contour_pred, ax=axs[0,1], ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
    cbar.set_label('Velocity Magnitude - Predicted', rotation=270, labelpad=15)
    axs[0,1].set_xlabel(f'{coordinate1} Coordinate')
    axs[0,1].set_ylabel(f'{coordinate2} Coordinate')
    axs[0,1].set_xlim(lim_min1, lim_max1) 
    axs[0,1].set_ylim(lim_min2, lim_max2)
    axs[0,1].set_title(f'Total Predicted Velocity in the {plane} Plane for Wind Angle = {angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')
    
    # For Velocity X
    contour_vx_actual = axs[1,0].contourf(grid1, grid2, grid_vx_actual, levels=128, cmap=cmap)
    axs[1,0].scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
    axs[1,0].scatter(cylinderfill1.ravel(),  cylinderfill2.ravel(), c='black', s=1, label='Cylinder Body Fill')
    axs[1,0].scatter(cylindercapfill1.ravel(), cylindercapfill2, c='black', s=1, label='Cylinder Cap Fill')
    vmin, vmax, cmap_, scatter_size, ticks = plotting_details(filtered_df['Velocity_X_Actual'].values)
    cbar = fig.colorbar(contour_vx_actual, ax=axs[1, 0], ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
    cbar.set_label('Velocity X - Actual', rotation=270, labelpad=15)
    axs[1,0].set_xlabel(f'{coordinate1} Coordinate')
    axs[1,0].set_ylabel(f'{coordinate2} Coordinate')
    axs[1,0].set_xlim(lim_min1, lim_max1) 
    axs[1,0].set_ylim(lim_min2, lim_max2)
    axs[1,0].set_title(f'Actual Velocity X in the {plane} Plane for Wind Angle = {angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')

    contour_vx_pred = axs[1,1].contourf(grid1, grid2, grid_vx_pred, levels=128, cmap=cmap)
    axs[1,1].scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
    axs[1,1].scatter(cylinderfill1.ravel(),  cylinderfill2.ravel(), c='black', s=1, label='Cylinder Body Fill')
    axs[1,1].scatter(cylindercapfill1.ravel(), cylindercapfill2, c='black', s=1, label='Cylinder Cap Fill')
    cbar = fig.colorbar(contour_vx_pred, ax=axs[1, 1], ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
    cbar.set_label('Velocity X - Predicted', rotation=270, labelpad=15)
    axs[1,1].set_xlabel(f'{coordinate1} Coordinate')
    axs[1,1].set_ylabel(f'{coordinate2} Coordinate')
    axs[1,1].set_xlim(lim_min1, lim_max1) 
    axs[1,1].set_ylim(lim_min2, lim_max2)
    axs[1,1].set_title(f'Predicted Velocity X in the {plane} Plane for Wind Angle = {angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')
    
    # For Velocity Y
    contour_vy_actual = axs[2,0].contourf(grid1, grid2, grid_vy_actual, levels=128, cmap=cmap)
    axs[2,0].scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
    axs[2,0].scatter(cylinderfill1.ravel(),  cylinderfill2.ravel(), c='black', s=1, label='Cylinder Body Fill')
    axs[2,0].scatter(cylindercapfill1.ravel(), cylindercapfill2, c='black', s=1, label='Cylinder Cap Fill')
    vmin, vmax, cmap_, scatter_size, ticks = plotting_details(filtered_df['Velocity_Y_Actual'].values)
    cbar = fig.colorbar(contour_vy_actual, ax=axs[2, 0], ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
    cbar.set_label('Velocity X - Actual', rotation=270, labelpad=15)
    axs[2,0].set_xlabel(f'{coordinate1} Coordinate')
    axs[2,0].set_ylabel(f'{coordinate2} Coordinate')
    axs[2,0].set_xlim(lim_min1, lim_max1) 
    axs[2,0].set_ylim(lim_min2, lim_max2)
    axs[2,0].set_title(f'Actual Velocity Y in the {plane} Plane for Wind Angle = {angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')

    contour_vy_pred = axs[2,1].contourf(grid1, grid2, grid_vy_pred, levels=128, cmap=cmap)
    axs[2,1].scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
    axs[2,1].scatter(cylinderfill1.ravel(),  cylinderfill2.ravel(), c='black', s=1, label='Cylinder Body Fill')
    axs[2,1].scatter(cylindercapfill1.ravel(), cylindercapfill2, c='black', s=1, label='Cylinder Cap Fill')
    cbar = fig.colorbar(contour_vy_pred, ax=axs[2, 1], ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
    cbar.set_label('Velocity X - Predicted', rotation=270, labelpad=15)
    axs[2,1].set_xlabel(f'{coordinate1} Coordinate')
    axs[2,1].set_ylabel(f'{coordinate2} Coordinate')
    axs[2,1].set_xlim(lim_min1, lim_max1) 
    axs[2,1].set_ylim(lim_min2, lim_max2)
    axs[2,1].set_title(f'Predicted Velocity Y in the {plane} Plane for Wind Angle = {angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')

    # For Velocity Z
    contour_vz_actual = axs[3,0].contourf(grid1, grid2, grid_vz_actual, levels=128, cmap=cmap)
    axs[3,0].scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
    axs[3,0].scatter(cylinderfill1.ravel(),  cylinderfill2.ravel(), c='black', s=1, label='Cylinder Body Fill')
    axs[3,0].scatter(cylindercapfill1.ravel(), cylindercapfill2, c='black', s=1, label='Cylinder Cap Fill')
    vmin, vmax, cmap_, scatter_size, ticks = plotting_details(filtered_df['Velocity_Z_Actual'].values)
    cbar = fig.colorbar(contour_vz_actual, ax=axs[3, 0], ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
    cbar.set_label('Velocity Z - Actual', rotation=270, labelpad=15)
    axs[3,0].set_xlabel(f'{coordinate1} Coordinate')
    axs[3,0].set_ylabel(f'{coordinate2} Coordinate')
    axs[3,0].set_xlim(lim_min1, lim_max1) 
    axs[3,0].set_ylim(lim_min2, lim_max2)
    axs[3,0].set_title(f'Actual Velocity Z in the {plane} Plane for Wind Angle = {angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')

    contour_vz_pred = axs[3,1].contourf(grid1, grid2, grid_vz_pred, levels=128, cmap=cmap)
    axs[3,1].scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
    axs[3,1].scatter(cylinderfill1.ravel(),  cylinderfill2.ravel(), c='black', s=1, label='Cylinder Body Fill')
    axs[3,1].scatter(cylindercapfill1.ravel(), cylindercapfill2, c='black', s=1, label='Cylinder Cap Fill')
    cbar = fig.colorbar(contour_vz_pred, ax=axs[3, 1], ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
    cbar.set_label('Velocity Z - Predicted', rotation=270, labelpad=15)
    axs[3,1].set_xlabel(f'{coordinate1} Coordinate')
    axs[3,1].set_ylabel(f'{coordinate2} Coordinate')
    axs[3,1].set_xlim(lim_min1, lim_max1) 
    axs[3,1].set_ylim(lim_min2, lim_max2)
    axs[3,1].set_title(f'Predicted Velocity Z in the {plane} Plane for Wind Angle = {angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')
    
    # For Pressure
    contour_pressure_actual = axs[4,0].contourf(grid1, grid2, grid_pressure_actual, levels=128, cmap=cmap)
    axs[4,0].scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
    axs[4,0].scatter(cylinderfill1.ravel(),  cylinderfill2.ravel(), c='black', s=1, label='Cylinder Body Fill')
    axs[4,0].scatter(cylindercapfill1.ravel(), cylindercapfill2, c='black', s=1, label='Cylinder Cap Fill')
    vmin, vmax, cmap_, scatter_size, ticks = plotting_details(filtered_df['Pressure_Actual'].values)
    cbar = fig.colorbar(contour_pressure_actual, ax=axs[4, 0], ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
    cbar.set_label('Pressure - Actual', rotation=270, labelpad=15)
    axs[4,0].set_xlabel(f'{coordinate1} Coordinate')
    axs[4,0].set_ylabel(f'{coordinate2} Coordinate')
    axs[4,0].set_xlim(lim_min1, lim_max1) 
    axs[4,0].set_ylim(lim_min2, lim_max2)
    axs[4,0].set_title(f'Actual Pressure in the {plane} Plane for Wind Angle = {angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')

    contour_pressure_pred = axs[4,1].contourf(grid1, grid2, grid_pressure_pred, levels=128, cmap=cmap)
    axs[4,1].scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
    axs[4,1].scatter(cylinderfill1.ravel(),  cylinderfill2.ravel(), c='black', s=1, label='Cylinder Body Fill')
    axs[4,1].scatter(cylindercapfill1.ravel(), cylindercapfill2, c='black', s=1, label='Cylinder Cap Fill')
    cbar = fig.colorbar(contour_pressure_pred, ax=axs[4, 1], ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
    cbar.set_label('Pressure - Predicted', rotation=270, labelpad=15)
    axs[4,1].set_xlabel(f'{coordinate1} Coordinate')
    axs[4,1].set_ylabel(f'{coordinate2} Coordinate')
    axs[4,1].set_xlim(lim_min1, lim_max1) 
    axs[4,1].set_ylim(lim_min2, lim_max2)
    axs[4,1].set_title(f'Predicted Pressure in the {plane} Plane for Wind Angle = {angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')

    plt.tight_layout()
    plt.savefig(savename_all)
    plt.close()

    fig, axs = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    fig.suptitle(f'Comparison of Actual vs. Predicted values with Wind Angle = {wind_angle} in the {plane} Plane with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')
    
    # For Total Velocity
    contour_actual = axs[0].contourf(grid1, grid2, grid_magnitude_actual, levels=128, cmap=cmap)
    axs[0].scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
    axs[0].scatter(cylinderfill1.ravel(),  cylinderfill2.ravel(), c='black', s=1, label='Cylinder Body Fill')
    axs[0].scatter(cylindercapfill1.ravel(), cylindercapfill2, c='black', s=1, label='Cylinder Cap Fill')
    vmin, vmax, cmap_, scatter_size, ticks = plotting_details(filtered_df['Velocity_Magnitude_Actual'].values)
    cbar = fig.colorbar(contour_actual, ax=axs[0], ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
    cbar.set_label('Velocity Magnitude - Actual', rotation=270, labelpad=15)
    axs[0].set_xlabel(f'{coordinate1} Coordinate')
    axs[0].set_ylabel(f'{coordinate2} Coordinate')
    axs[0].set_xlim(lim_min1, lim_max1) 
    axs[0].set_ylim(lim_min2, lim_max2)
    axs[0].set_title(f'Total Actual Velocity in the {plane} Plane for Wind Angle = {angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')
    
    contour_pred = axs[1].contourf(grid1, grid2, grid_magnitude_pred, levels=128, cmap=cmap)
    axs[1].scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
    axs[1].scatter(cylinderfill1.ravel(),  cylinderfill2.ravel(), c='black', s=1, label='Cylinder Body Fill')
    axs[1].scatter(cylindercapfill1.ravel(), cylindercapfill2, c='black', s=1, label='Cylinder Cap Fill')
    cbar = fig.colorbar(contour_pred, ax=axs[1], ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
    cbar.set_label('Velocity Magnitude - Predicted', rotation=270, labelpad=15)
    axs[1].set_xlabel(f'{coordinate1} Coordinate')
    axs[1].set_ylabel(f'{coordinate2} Coordinate')
    axs[1].set_xlim(lim_min1, lim_max1) 
    axs[1].set_ylim(lim_min2, lim_max2)
    axs[1].set_title(f'Total Predicted Velocity in the {plane} Plane for Wind Angle = {angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')

    plt.tight_layout()
    plt.savefig(savename_total_velocity)
    plt.close()

    fig, axs = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    fig.suptitle(f'Comparison of Actual vs. Predicted values with Wind Angle = {wind_angle} in the {plane} Plane with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')

    # For Velocity X
    contour_vx_actual = axs[0].contourf(grid1, grid2, grid_vx_actual, levels=128, cmap=cmap)
    axs[0].scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
    axs[0].scatter(cylinderfill1.ravel(),  cylinderfill2.ravel(), c='black', s=1, label='Cylinder Body Fill')
    axs[0].scatter(cylindercapfill1.ravel(), cylindercapfill2, c='black', s=1, label='Cylinder Cap Fill')
    vmin, vmax, cmap_, scatter_size, ticks = plotting_details(filtered_df['Velocity_X_Actual'].values)
    cbar = fig.colorbar(contour_vx_actual, ax=axs[0], ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
    cbar.set_label('Velocity X - Actual', rotation=270, labelpad=15)
    axs[0].set_xlabel(f'{coordinate1} Coordinate')
    axs[0].set_ylabel(f'{coordinate2} Coordinate')
    axs[0].set_xlim(lim_min1, lim_max1) 
    axs[0].set_ylim(lim_min2, lim_max2)
    axs[0].set_title(f'Actual Velocity X in the {plane} Plane for Wind Angle = {angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')

    contour_vx_pred = axs[1].contourf(grid1, grid2, grid_vx_pred, levels=128, cmap=cmap)
    axs[1].scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
    axs[1].scatter(cylinderfill1.ravel(),  cylinderfill2.ravel(), c='black', s=1, label='Cylinder Body Fill')
    axs[1].scatter(cylindercapfill1.ravel(), cylindercapfill2, c='black', s=1, label='Cylinder Cap Fill')
    cbar = fig.colorbar(contour_vx_pred, ax=axs[1], ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
    cbar.set_label('Velocity X - Predicted', rotation=270, labelpad=15)
    axs[1].set_xlabel(f'{coordinate1} Coordinate')
    axs[1].set_ylabel(f'{coordinate2} Coordinate')
    axs[1].set_xlim(lim_min1, lim_max1) 
    axs[1].set_ylim(lim_min2, lim_max2)
    axs[1].set_title(f'Predicted Velocity X in the {plane} Plane for Wind Angle = {angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')
    
    plt.tight_layout()
    plt.savefig(savename_vx)
    plt.close()

    fig, axs = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    fig.suptitle(f'Comparison of Actual vs. Predicted values with Wind Angle = {wind_angle} in the {plane} Plane with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')

    # For Velocity Y
    contour_vy_actual = axs[0].contourf(grid1, grid2, grid_vy_actual, levels=128, cmap=cmap)
    axs[0].scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
    axs[0].scatter(cylinderfill1.ravel(),  cylinderfill2.ravel(), c='black', s=1, label='Cylinder Body Fill')
    axs[0].scatter(cylindercapfill1.ravel(), cylindercapfill2, c='black', s=1, label='Cylinder Cap Fill')
    vmin, vmax, cmap_, scatter_size, ticks = plotting_details(filtered_df['Velocity_Y_Actual'].values)
    cbar = fig.colorbar(contour_vy_actual, ax=axs[0], ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
    cbar.set_label('Velocity Y - Actual', rotation=270, labelpad=15)
    axs[0].set_xlabel(f'{coordinate1} Coordinate')
    axs[0].set_ylabel(f'{coordinate2} Coordinate')
    axs[0].set_xlim(lim_min1, lim_max1) 
    axs[0].set_ylim(lim_min2, lim_max2)
    axs[0].set_title(f'Actual Velocity Y in the {plane} Plane for Wind Angle = {angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')

    contour_vy_pred = axs[1].contourf(grid1, grid2, grid_vy_pred, levels=128, cmap=cmap)
    axs[1].scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
    axs[1].scatter(cylinderfill1.ravel(),  cylinderfill2.ravel(), c='black', s=1, label='Cylinder Body Fill')
    axs[1].scatter(cylindercapfill1.ravel(), cylindercapfill2, c='black', s=1, label='Cylinder Cap Fill')
    cbar = fig.colorbar(contour_vy_pred, ax=axs[1], ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
    cbar.set_label('Velocity Y - Predicted', rotation=270, labelpad=15)
    axs[1].set_xlabel(f'{coordinate1} Coordinate')
    axs[1].set_ylabel(f'{coordinate2} Coordinate')
    axs[1].set_xlim(lim_min1, lim_max1) 
    axs[1].set_ylim(lim_min2, lim_max2)
    axs[1].set_title(f'Predicted Velocity Y in the {plane} Plane for Wind Angle = {angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')
    
    plt.tight_layout()
    plt.savefig(savename_vy)
    plt.close()

    fig, axs = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    fig.suptitle(f'Comparison of Actual vs. Predicted values with Wind Angle = {wind_angle} in the {plane} Plane with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')

    # For Velocity Z
    contour_vz_actual = axs[0].contourf(grid1, grid2, grid_vz_actual, levels=128, cmap=cmap)
    axs[0].scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
    axs[0].scatter(cylinderfill1.ravel(),  cylinderfill2.ravel(), c='black', s=1, label='Cylinder Body Fill')
    axs[0].scatter(cylindercapfill1.ravel(), cylindercapfill2, c='black', s=1, label='Cylinder Cap Fill')
    vmin, vmax, cmap_, scatter_size, ticks = plotting_details(filtered_df['Velocity_Z_Actual'].values)
    cbar = fig.colorbar(contour_vz_actual, ax=axs[0], ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
    cbar.set_label('Velocity Z - Actual', rotation=270, labelpad=15)
    axs[0].set_xlabel(f'{coordinate1} Coordinate')
    axs[0].set_ylabel(f'{coordinate2} Coordinate')
    axs[0].set_xlim(lim_min1, lim_max1) 
    axs[0].set_ylim(lim_min2, lim_max2)
    axs[0].set_title(f'Actual Velocity Z in the {plane} Plane for Wind Angle = {angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')

    contour_vz_pred = axs[1].contourf(grid1, grid2, grid_vz_pred, levels=128, cmap=cmap)
    axs[1].scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
    axs[1].scatter(cylinderfill1.ravel(),  cylinderfill2.ravel(), c='black', s=1, label='Cylinder Body Fill')
    axs[1].scatter(cylindercapfill1.ravel(), cylindercapfill2, c='black', s=1, label='Cylinder Cap Fill')
    cbar = fig.colorbar(contour_vz_pred, ax=axs[1], ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
    cbar.set_label('Velocity Z - Predicted', rotation=270, labelpad=15)
    axs[1].set_xlabel(f'{coordinate1} Coordinate')
    axs[1].set_ylabel(f'{coordinate2} Coordinate')
    axs[1].set_xlim(lim_min1, lim_max1) 
    axs[1].set_ylim(lim_min2, lim_max2)
    axs[1].set_title(f'Predicted Velocity Z in the {plane} Plane for Wind Angle = {angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')
    
    plt.tight_layout()
    plt.savefig(savename_vz)
    plt.close()

    fig, axs = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    fig.suptitle(f'Comparison of Actual vs. Predicted values with Wind Angle = {wind_angle} in the {plane} Plane with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')

    # For Pressure
    contour_pressure_actual = axs[0].contourf(grid1, grid2, grid_pressure_actual, levels=128, cmap=cmap)
    axs[0].scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
    axs[0].scatter(cylinderfill1.ravel(),  cylinderfill2.ravel(), c='black', s=1, label='Cylinder Body Fill')
    axs[0].scatter(cylindercapfill1.ravel(), cylindercapfill2, c='black', s=1, label='Cylinder Cap Fill')
    vmin, vmax, cmap_, scatter_size, ticks = plotting_details(filtered_df['Pressure_Actual'].values)
    cbar = fig.colorbar(contour_pressure_actual, ax=axs[0], ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
    cbar.set_label('Pressure - Actual', rotation=270, labelpad=15)
    axs[0].set_xlabel(f'{coordinate1} Coordinate')
    axs[0].set_ylabel(f'{coordinate2} Coordinate')
    axs[0].set_xlim(lim_min1, lim_max1) 
    axs[0].set_ylim(lim_min2, lim_max2)
    axs[0].set_title(f'Actual Pressure in the {plane} Plane for Wind Angle = {angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')

    contour_pressure_pred = axs[1].contourf(grid1, grid2, grid_pressure_pred, levels=128, cmap=cmap)
    axs[1].scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
    axs[1].scatter(cylinderfill1.ravel(),  cylinderfill2.ravel(), c='black', s=1, label='Cylinder Body Fill')
    axs[1].scatter(cylindercapfill1.ravel(), cylindercapfill2, c='black', s=1, label='Cylinder Cap Fill')
    cbar = fig.colorbar(contour_pressure_pred, ax=axs[1], ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
    cbar.set_label('Pressure - Predicted', rotation=270, labelpad=15)
    axs[1].set_xlabel(f'{coordinate1} Coordinate')
    axs[1].set_ylabel(f'{coordinate2} Coordinate')
    axs[1].set_xlim(lim_min1, lim_max1) 
    axs[1].set_ylim(lim_min2, lim_max2)
    axs[1].set_title(f'Predicted Pressure in the {plane} Plane for Wind Angle = {angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')
    
    plt.tight_layout()
    plt.savefig(savename_pressure)
    plt.close()

def data_plot_scatter_2d(datafolder,plot_folder):    
    thetas = [[1,0], [2,30], [3,60], [4,90] , [5,120], [6,135], [7,150], [8,180]]
    params = [['X-Z',570,20],['Y-Z',500,20],['X-Y',50,20]]
    cmap = 'jet'
    for i in thetas:
        for j in params:
            angle = i[1]
            num = i[0]
            filename = os.path.join(datafolder,f'CFD_cell_data_simulation_{num}.csv')
            plane = j[0]
            cut = j[1]
            tolerance = j[2]
            savename = os.path.join(plot_folder,f'{plane}_total_velocity_{cmap}_{angle}')
            plot_data(filename,angle,savename,plane,cut,tolerance,cmap)

def plot_prediction_2d(df,angle, plot_folder):
    params = [['X-Z',570,5],['Y-Z',500,5],['X-Y',50,5]]
    cmap = 'jet'
    for j in params:
        plane = j[0]
        cut = j[1]
        tolerance = j[2]
        savename_all = os.path.join(plot_folder,f'{plane}_allplots_{cmap}_{angle}.png')
        savename_total_velocity = os.path.join(plot_folder,f'{plane}_totalvelocity_{cmap}_{angle}.png')
        savename_vx = os.path.join(plot_folder,f'{plane}_vx_{cmap}_{angle}.png')
        savename_vy = os.path.join(plot_folder,f'{plane}_vy_{cmap}_{angle}.png')
        savename_vz = os.path.join(plot_folder,f'{plane}_vz_{cmap}_{angle}.png')
        savename_pressure = os.path.join(plot_folder,f'{plane}_pressure_{cmap}_{angle}.png')
        plot_data_predictions(df,angle,savename_all,savename_total_velocity,savename_vx,savename_vy,savename_vz,savename_pressure,plane,cut,tolerance,cmap)