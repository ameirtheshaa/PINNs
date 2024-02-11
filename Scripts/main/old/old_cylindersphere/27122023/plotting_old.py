from definitions import *
from training import *
from PINN import *

def normalizing_details():
    x_length = 1000
    y_length = 1000
    z_length = 300

    return x_length/2, y_length/2, z_length

def plotting_details(z_actual):
    mean_z = np.mean(z_actual)
    std_z = np.std(z_actual)
    vmin = mean_z - 2 * std_z
    vmax = mean_z + 2 * std_z

    vmin = int(vmin)
    vmax = int(vmax)

    # cmap = plt.cm.RdBu_r
    cmap = 'jet'
    scatter_size = 5
    ticks = 5

    return vmin, vmax, cmap, scatter_size, ticks

def normalize_grids(plane, points1, points2):

    x_length, y_length, z_length = normalizing_details()

    points1 = np.array(points1)
    points2 = np.array(points2)
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

def geometry_coordinates(plane, datafolder_path):
    num_points = 0.5
    df_sphere = pd.read_csv(os.path.join(datafolder_path, 'sphere.csv'))
    df_cylinder = pd.read_csv(os.path.join(datafolder_path, 'cylinder.csv'))
    
    # Randomly sample a subset of faces
    sphere_faces = random.sample(df_sphere.values.tolist(), int(len(df_sphere) * num_points))
    cylinder_faces = random.sample(df_cylinder.values.tolist(), int(len(df_cylinder) * num_points))

    # Function to reshape face data into a list of tuples
    def reshape_face_data(data):
        reshaped_data = []
        for face in data:
            reshaped_face = [tuple(face[i:i+3]) for i in range(0, len(face), 3)]
            reshaped_data.append(reshaped_face)
        return reshaped_data

    # Extracting x, y, z coordinates
    sphere_x, sphere_y, sphere_z = zip(*[coord for face in sphere_faces for coord in reshape_face_data([face])[0]])
    cylinder_x, cylinder_y, cylinder_z = zip(*[coord for face in cylinder_faces for coord in reshape_face_data([face])[0]])

    if plane == 'Y-Z':
        return sphere_y, sphere_z, cylinder_y, cylinder_z
    elif plane == 'X-Z':
        return sphere_x, sphere_z, cylinder_x, cylinder_z
    elif plane == 'X-Y':
        return sphere_x, sphere_y, cylinder_x, cylinder_y

def plot_data(df,angle,config,datafolder_path,savename_total_velocity,savename_total_velocity_arrow,savename_vx,savename_vy,savename_vz,savename_pressure,savename_turbvisc,savename_turbvisc_arrow,plane,cut,tolerance):
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
        arrow_x1 = 500
        arrow_y1 = 500
        arrow_x2 = 500
        arrow_y2 = 570
        length = 10000000
        wind_angle_modf = 270 - wind_angle
        wind_angle_rad_modf = np.deg2rad(wind_angle_modf)
        arrow_dx1 = length*(np.cos(wind_angle_rad_modf))
        arrow_dy1 = length*(np.sin(wind_angle_rad_modf))
        arrow_dx2 = length*(np.cos(wind_angle_rad_modf))
        arrow_dy2 = length*(np.sin(wind_angle_rad_modf))
        arrow_x1, arrow_y1 = normalize_grids(plane, arrow_x1, arrow_y1)
        arrow_dx1, arrow_dy1 = normalize_grids(plane, arrow_dx1, arrow_dy1)
        arrow_x2, arrow_y2 = normalize_grids(plane, arrow_x2, arrow_y2)
        arrow_dx2, arrow_dy2 = normalize_grids(plane, arrow_dx2, arrow_dy2)
        arrow1 = patches.Arrow(arrow_x1, arrow_y1, arrow_dx1, arrow_dy1, width=0.05, color="white")
        arrow2 = patches.Arrow(arrow_x2, arrow_y2, arrow_dx2, arrow_dy2, width=0.05, color="white")
        arrow1_ = patches.Arrow(arrow_x1, arrow_y1, arrow_dx1, arrow_dy1, width=0.05, color="white")
        arrow2_ = patches.Arrow(arrow_x2, arrow_y2, arrow_dx2, arrow_dy2, width=0.05, color="white")

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
        grid_vx = griddata(filtered_df[[points1, points2]].values, filtered_df['Velocity_X'].values, (grid1, grid2), method='linear', fill_value=0)
        grid_vy = griddata(filtered_df[[points1, points2]].values, filtered_df['Velocity_Y'].values, (grid1, grid2), method='linear', fill_value=0)
        grid_vz = griddata(filtered_df[[points1, points2]].values, filtered_df['Velocity_Z'].values, (grid1, grid2), method='linear', fill_value=0)
        grid_pressure = griddata(filtered_df[[points1, points2]].values, filtered_df['Pressure'].values, (grid1, grid2), method='linear', fill_value=0)
        grid_turbvisc = griddata(filtered_df[[points1, points2]].values, filtered_df['TurbVisc'].values, (grid1, grid2), method='linear', fill_value=0)
    except:
        print(f"not enough points")
        return

    # Calculate the total magnitude of the velocity on the grid
    grid_magnitude = np.sqrt(grid_vx**2 + grid_vy**2 + grid_vz**2)

    spherefill1, spherefill2, cylinderfill1, cylinderfill2 = geometry_coordinates(plane, datafolder_path)

    grid1, grid2 = normalize_grids(plane, grid1, grid2)
    spherefill1, spherefill2 = normalize_grids(plane, spherefill1, spherefill2)
    cylinderfill1, cylinderfill2 = normalize_grids(plane, cylinderfill1, cylinderfill2)
    
    cut, tolerance = normalize_cut_tolerance(plane, cut, tolerance)

    if config["plotting"]["make_pure_data_plots_quiver"]:
        # Decide on quiver density (e.g., every 20th point)
        stride = 100
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

    if config["plotting"]["make_pure_data_plots_total_velocity"] and not Path(savename_total_velocity).exists():

        # For Total Velocity
        if plane == 'X-Y':
            if config["plotting"]["make_pure_data_plots_total_velocity_arrow"]:
                fig, ax = plt.subplots(figsize=(16, 8), sharey=True)
                vmin, vmax, cmap, scatter_size, ticks = plotting_details(filtered_df['Velocity_Magnitude'].values)
                contour = ax.contourf(grid1, grid2, grid_magnitude, levels=128, vmin=vmin, vmax=vmax, cmap=cmap)
                if config["plotting"]["make_pure_data_plots_quiver"]:
                    ax.quiver(quiver_points1, quiver_points2, quiver_v1, quiver_v2, scale=200, color='white')
                ax.scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere')
                
                ax.tricontourf(tri.Triangulation(cylinderfill1, cylinderfill2), np.zeros(len(cylinderfill1)), colors='black', alpha=1)
                cylinder_patch = patches.Patch(color='black', label='Cylinder Fill')
                # plt.legend(handles=[cylinder_patch])
                ax.add_patch(arrow1)
                ax.add_patch(arrow2)
                cbar = fig.colorbar(contour, ax=ax, ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
                cbar.set_label('Velocity Magnitude', rotation=270, labelpad=15)
                ax.set_xlabel(f'{coordinate1} Coordinate')
                ax.set_ylabel(f'{coordinate2} Coordinate')
                ax.set_xlim(lim_min1, lim_max1) 
                ax.set_ylim(lim_min2, lim_max2)
                ax.set_title(f'Total Velocity in the {plane} Plane for Wind Angle = {angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')
                plt.tight_layout()
                plt.savefig(savename_total_velocity_arrow)
                plt.close()

        # For Total Velocity
        fig, ax = plt.subplots(figsize=(16, 8), sharey=True)
        vmin, vmax, cmap, scatter_size, ticks = plotting_details(filtered_df['Velocity_Magnitude'].values)
        contour = ax.contourf(grid1, grid2, grid_magnitude, levels=128, vmin=vmin, vmax=vmax, cmap=cmap)
        if config["plotting"]["make_pure_data_plots_quiver"]:
            ax.quiver(quiver_points1, quiver_points2, quiver_v1, quiver_v2, scale=200, color='white')
        ax.scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
        ax.tricontourf(tri.Triangulation(cylinderfill1, cylinderfill2), np.zeros(len(cylinderfill1)), colors='black', alpha=1)  # Fill color set to black
        cylinder_patch = patches.Patch(color='black', label='Cylinder Fill')
        # plt.legend(handles=[cylinder_patch])
        
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

    if config["plotting"]["make_pure_data_plots_vx"] and not Path(savename_vx).exists():
        fig, ax = plt.subplots(figsize=(16, 8), sharey=True)

        # For Velocity_X
        vmin, vmax, cmap, scatter_size, ticks = plotting_details(filtered_df['Velocity_X'].values)
        contour = ax.contourf(grid1, grid2, grid_vx, levels=128, vmin=vmin, vmax=vmax, cmap=cmap)
        if config["plotting"]["make_pure_data_plots_quiver"]:
            ax.quiver(quiver_points1, quiver_points2, quiver_v1, quiver_v2, scale=200, color='white')
        ax.scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
        ax.tricontourf(tri.Triangulation(cylinderfill1, cylinderfill2), np.zeros(len(cylinderfill1)), colors='black', alpha=1)  # Fill color set to black
        cylinder_patch = patches.Patch(color='black', label='Cylinder Fill')
        # plt.legend(handles=[cylinder_patch])
        
        cbar = fig.colorbar(contour, ax=ax, ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
        cbar.set_label('Velocity X', rotation=270, labelpad=15)
        ax.set_xlabel(f'{coordinate1} Coordinate')
        ax.set_ylabel(f'{coordinate2} Coordinate')
        ax.set_xlim(lim_min1, lim_max1) 
        ax.set_ylim(lim_min2, lim_max2)
        ax.set_title(f'Velocity X in the {plane} Plane for Wind Angle = {angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')
        
        plt.tight_layout()
        plt.savefig(savename_vx)
        plt.close()

    if config["plotting"]["make_pure_data_plots_vy"] and not Path(savename_vy).exists():
        fig, ax = plt.subplots(figsize=(16, 8), sharey=True)

        # For Velocity_Y
        vmin, vmax, cmap, scatter_size, ticks = plotting_details(filtered_df['Velocity_Y'].values)
        contour = ax.contourf(grid1, grid2, grid_vy, levels=128, vmin=vmin, vmax=vmax, cmap=cmap)
        if config["plotting"]["make_pure_data_plots_quiver"]:
            ax.quiver(quiver_points1, quiver_points2, quiver_v1, quiver_v2, scale=200, color='white')
        ax.scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
        ax.tricontourf(tri.Triangulation(cylinderfill1, cylinderfill2), np.zeros(len(cylinderfill1)), colors='black', alpha=1)  # Fill color set to black
        cylinder_patch = patches.Patch(color='black', label='Cylinder Fill')
        # plt.legend(handles=[cylinder_patch])
        
        cbar = fig.colorbar(contour, ax=ax, ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
        cbar.set_label('Velocity Y', rotation=270, labelpad=15)
        ax.set_xlabel(f'{coordinate1} Coordinate')
        ax.set_ylabel(f'{coordinate2} Coordinate')
        ax.set_xlim(lim_min1, lim_max1) 
        ax.set_ylim(lim_min2, lim_max2)
        ax.set_title(f'Velocity Y in the {plane} Plane for Wind Angle = {angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')
        
        plt.tight_layout()
        plt.savefig(savename_vy)
        plt.close()

    if config["plotting"]["make_pure_data_plots_vz"] and not Path(savename_vz).exists():
        fig, ax = plt.subplots(figsize=(16, 8), sharey=True)

        # For Velocity_Z
        vmin, vmax, cmap, scatter_size, ticks = plotting_details(filtered_df['Velocity_Z'].values)
        contour = ax.contourf(grid1, grid2, grid_vz, levels=128, vmin=vmin, vmax=vmax, cmap=cmap)
        if config["plotting"]["make_pure_data_plots_quiver"]:
            ax.quiver(quiver_points1, quiver_points2, quiver_v1, quiver_v2, scale=200, color='white')
        ax.scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
        ax.tricontourf(tri.Triangulation(cylinderfill1, cylinderfill2), np.zeros(len(cylinderfill1)), colors='black', alpha=1)  # Fill color set to black
        cylinder_patch = patches.Patch(color='black', label='Cylinder Fill')
        # plt.legend(handles=[cylinder_patch])
        
        cbar = fig.colorbar(contour, ax=ax, ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
        cbar.set_label('Velocity Z', rotation=270, labelpad=15)
        ax.set_xlabel(f'{coordinate1} Coordinate')
        ax.set_ylabel(f'{coordinate2} Coordinate')
        ax.set_xlim(lim_min1, lim_max1) 
        ax.set_ylim(lim_min2, lim_max2)
        ax.set_title(f'Velocity Z in the {plane} Plane for Wind Angle = {angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')
        
        plt.tight_layout()
        plt.savefig(savename_vz)
        plt.close()

    if config["plotting"]["make_pure_data_plots_pressure"] and not Path(savename_pressure).exists():
        fig, ax = plt.subplots(figsize=(16, 8), sharey=True)

        # For Pressure
        vmin, vmax, cmap, scatter_size, ticks = plotting_details(filtered_df['Pressure'].values)
        contour = ax.contourf(grid1, grid2, grid_pressure, levels=128, vmin=vmin, vmax=vmax, cmap=cmap)
        if config["plotting"]["make_pure_data_plots_quiver"]:
            ax.quiver(quiver_points1, quiver_points2, quiver_v1, quiver_v2, scale=200, color='white')
        ax.scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
        ax.tricontourf(tri.Triangulation(cylinderfill1, cylinderfill2), np.zeros(len(cylinderfill1)), colors='black', alpha=1)  # Fill color set to black
        cylinder_patch = patches.Patch(color='black', label='Cylinder Fill')
        # plt.legend(handles=[cylinder_patch])
        
        cbar = fig.colorbar(contour, ax=ax, ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
        cbar.set_label('Pressure', rotation=270, labelpad=15)
        ax.set_xlabel(f'{coordinate1} Coordinate')
        ax.set_ylabel(f'{coordinate2} Coordinate')
        ax.set_xlim(lim_min1, lim_max1) 
        ax.set_ylim(lim_min2, lim_max2)
        ax.set_title(f'Pressure in the {plane} Plane for Wind Angle = {angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')
        
        plt.tight_layout()
        plt.savefig(savename_pressure)
        plt.close()

    if config["plotting"]["make_pure_data_plots_turbvisc"] and not Path(savename_turbvisc).exists():
        # For TurbVisc

        if plane == 'X-Y':
            if config["plotting"]["make_pure_data_plots_turbvisc_arrow"]:
                fig, ax = plt.subplots(figsize=(16, 8), sharey=True)
                vmin, vmax, cmap, scatter_size, ticks = plotting_details(filtered_df['TurbVisc'].values)
                contour = ax.contourf(grid1, grid2, grid_turbvisc, levels=128, vmin=vmin, vmax=vmax, cmap=cmap)
                if config["plotting"]["make_pure_data_plots_quiver"]:
                    ax.quiver(quiver_points1, quiver_points2, quiver_v1, quiver_v2, scale=200, color='white')
                ax.scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
                ax.tricontourf(tri.Triangulation(cylinderfill1, cylinderfill2), np.zeros(len(cylinderfill1)), colors='black', alpha=1)  # Fill color set to black
                cylinder_patch = patches.Patch(color='black', label='Cylinder Fill')
                # plt.legend(handles=[cylinder_patch])
                
                ax.add_patch(arrow1_)
                ax.add_patch(arrow2_)
                cbar = fig.colorbar(contour, ax=ax, ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
                cbar.set_label('TurbVisc', rotation=270, labelpad=15)
                ax.set_xlabel(f'{coordinate1} Coordinate')
                ax.set_ylabel(f'{coordinate2} Coordinate')
                ax.set_xlim(lim_min1, lim_max1) 
                ax.set_ylim(lim_min2, lim_max2)
                ax.set_title(f'TurbVisc in the {plane} Plane for Wind Angle = {angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')
                plt.tight_layout()
                plt.savefig(savename_turbvisc_arrow)
                plt.close()

        fig, ax = plt.subplots(figsize=(16, 8), sharey=True)
        vmin, vmax, cmap, scatter_size, ticks = plotting_details(filtered_df['TurbVisc'].values)
        contour = ax.contourf(grid1, grid2, grid_turbvisc, levels=128, vmin=vmin, vmax=vmax, cmap=cmap)
        if config["plotting"]["make_pure_data_plots_quiver"]:
            ax.quiver(quiver_points1, quiver_points2, quiver_v1, quiver_v2, scale=200, color='white')
        ax.scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
        ax.tricontourf(tri.Triangulation(cylinderfill1, cylinderfill2), np.zeros(len(cylinderfill1)), colors='black', alpha=1)  # Fill color set to black
        cylinder_patch = patches.Patch(color='black', label='Cylinder Fill')
        # plt.legend(handles=[cylinder_patch])
        
        cbar = fig.colorbar(contour, ax=ax, ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
        cbar.set_label('TurbVisc', rotation=270, labelpad=15)
        ax.set_xlabel(f'{coordinate1} Coordinate')
        ax.set_ylabel(f'{coordinate2} Coordinate')
        ax.set_xlim(lim_min1, lim_max1) 
        ax.set_ylim(lim_min2, lim_max2)
        ax.set_title(f'TurbVisc in the {plane} Plane for Wind Angle = {angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')
        plt.tight_layout()
        plt.savefig(savename_turbvisc)
        plt.close()

def plot_data_predictions(df,angle,config,datafolder_path,savename_all,savename_total_velocity,savename_total_velocity_arrow,savename_vx,savename_vy,savename_vz,savename_pressure,savename_turbvisc,savename_turbvisc_arrow,plane,cut,tolerance):
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
        arrow_x1 = 500
        arrow_y1 = 500
        arrow_x2 = 500
        arrow_y2 = 570
        length = 10000000
        wind_angle_modf = 270 - wind_angle
        wind_angle_rad_modf = np.deg2rad(wind_angle_modf)
        arrow_dx1 = length*(np.cos(wind_angle_rad_modf))
        arrow_dy1 = length*(np.sin(wind_angle_rad_modf))
        arrow_dx2 = length*(np.cos(wind_angle_rad_modf))
        arrow_dy2 = length*(np.sin(wind_angle_rad_modf))
        arrow_x1, arrow_y1 = normalize_grids(plane, arrow_x1, arrow_y1)
        arrow_dx1, arrow_dy1 = normalize_grids(plane, arrow_dx1, arrow_dy1)
        arrow_x2, arrow_y2 = normalize_grids(plane, arrow_x2, arrow_y2)
        arrow_dx2, arrow_dy2 = normalize_grids(plane, arrow_dx2, arrow_dy2)
        arrow10 = patches.Arrow(arrow_x1, arrow_y1, arrow_dx1, arrow_dy1, width=0.05, color="white")
        arrow20 = patches.Arrow(arrow_x2, arrow_y2, arrow_dx2, arrow_dy2, width=0.05, color="white")
        arrow11 = patches.Arrow(arrow_x1, arrow_y1, arrow_dx1, arrow_dy1, width=0.05, color="white")
        arrow21 = patches.Arrow(arrow_x2, arrow_y2, arrow_dx2, arrow_dy2, width=0.05, color="white")
        arrow10_ = patches.Arrow(arrow_x1, arrow_y1, arrow_dx1, arrow_dy1, width=0.05, color="white")
        arrow20_ = patches.Arrow(arrow_x2, arrow_y2, arrow_dx2, arrow_dy2, width=0.05, color="white")
        arrow11_ = patches.Arrow(arrow_x1, arrow_y1, arrow_dx1, arrow_dy1, width=0.05, color="white")
        arrow21_ = patches.Arrow(arrow_x2, arrow_y2, arrow_dx2, arrow_dy2, width=0.05, color="white")

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
        if 'Velocity_X_Actual' in filtered_df.columns:
            grid_vx_actual = griddata(filtered_df[[points1, points2]].values, filtered_df['Velocity_X_Actual'].values, (grid1, grid2), method='linear', fill_value=0)
        if 'Velocity_Y_Actual' in filtered_df.columns:
            grid_vy_actual = griddata(filtered_df[[points1, points2]].values, filtered_df['Velocity_Y_Actual'].values, (grid1, grid2), method='linear', fill_value=0)
        if 'Velocity_Z_Actual' in filtered_df.columns:
            grid_vz_actual = griddata(filtered_df[[points1, points2]].values, filtered_df['Velocity_Z_Actual'].values, (grid1, grid2), method='linear', fill_value=0)
        if 'Pressure_Actual' in filtered_df.columns:
            grid_pressure_actual = griddata(filtered_df[[points1, points2]].values, filtered_df['Pressure_Actual'].values, (grid1, grid2), method='linear', fill_value=0)
        if 'TurbVisc_Actual' in filtered_df.columns:
            grid_turbvisc_actual = griddata(filtered_df[[points1, points2]].values, filtered_df['TurbVisc_Actual'].values, (grid1, grid2), method='linear', fill_value=0)
        if 'Velocity_X_Predicted' in filtered_df.columns:
            grid_vx_pred = griddata(filtered_df[[points1, points2]].values, filtered_df['Velocity_X_Predicted'].values, (grid1, grid2), method='linear', fill_value=0)
        if 'Velocity_Y_Predicted' in filtered_df.columns:
            grid_vy_pred = griddata(filtered_df[[points1, points2]].values, filtered_df['Velocity_Y_Predicted'].values, (grid1, grid2), method='linear', fill_value=0)
        if 'Velocity_Z_Predicted' in filtered_df.columns:    
            grid_vz_pred = griddata(filtered_df[[points1, points2]].values, filtered_df['Velocity_Z_Predicted'].values, (grid1, grid2), method='linear', fill_value=0)
        if 'Pressure_Predicted' in filtered_df.columns:
            grid_pressure_pred = griddata(filtered_df[[points1, points2]].values, filtered_df['Pressure_Predicted'].values, (grid1, grid2), method='linear', fill_value=0)
        if 'TurbVisc_Predicted' in filtered_df.columns:    
            grid_turbvisc_pred = griddata(filtered_df[[points1, points2]].values, filtered_df['TurbVisc_Predicted'].values, (grid1, grid2), method='linear', fill_value=0)
    except:
        print(f"not enough points")
        return

    # Calculate the total magnitude of the velocity on the grid
    if 'Velocity_X_Actual' in filtered_df.columns and 'Velocity_Y_Actual' in filtered_df.columns and 'Velocity_Z_Actual' in filtered_df.columns:
        grid_magnitude_actual = np.sqrt(grid_vx_actual**2 + grid_vy_actual**2 + grid_vz_actual**2)
    if 'Velocity_X_Predicted' in filtered_df.columns and 'Velocity_Y_Predicted' in filtered_df.columns and 'Velocity_Z_Predicted' in filtered_df.columns:
        grid_magnitude_pred = np.sqrt(grid_vx_pred**2 + grid_vy_pred**2 + grid_vz_pred**2)

    spherefill1, spherefill2, cylinderfill1, cylinderfill2 = geometry_coordinates(plane, datafolder_path)

    grid1, grid2 = normalize_grids(plane, grid1, grid2)
    spherefill1, spherefill2 = normalize_grids(plane, spherefill1, spherefill2)
    cylinderfill1, cylinderfill2 = normalize_grids(plane, cylinderfill1, cylinderfill2)
    
    cut, tolerance = normalize_cut_tolerance(plane, cut, tolerance)

    if config["plotting"]["make_comparison_plots_quiver"]:
        # Decide on quiver density (e.g., every 20th point)
        stride = 100
        quiver_points1 = grid1[::stride, ::stride]
        quiver_points2 = grid2[::stride, ::stride]
        quiver_vx_actual = grid_vx_actual[::stride, ::stride]
        quiver_vy_actual = grid_vy_actual[::stride, ::stride]
        quiver_vz_actual = grid_vz_actual[::stride, ::stride]
        quiver_vx_pred = grid_vx_pred[::stride, ::stride]
        quiver_vy_pred = grid_vy_pred[::stride, ::stride]
        quiver_vz_pred = grid_vz_pred[::stride, ::stride]
        if plane == 'X-Z':
            quiver_v1_actual = quiver_vx_actual
            quiver_v2_actual = quiver_vz_actual
            quiver_v1_pred = quiver_vx_pred
            quiver_v2_pred = quiver_vz_pred
        elif plane == 'Y-Z':
            quiver_v1_actual = quiver_vy_actual
            quiver_v2_actual = quiver_vz_actual
            quiver_v1_pred = quiver_vy_pred
            quiver_v2_pred = quiver_vz_pred
        elif plane == 'X-Y':
            quiver_v1_actual = quiver_vx_actual
            quiver_v2_actual = quiver_vy_actual
            quiver_v1_pred = quiver_vx_pred
            quiver_v2_pred = quiver_vy_pred

    if config["plotting"]["make_comparison_plots_all"]:
        fig, axs = plt.subplots(5, 2, figsize=(16, 40), sharey=True)
        fig.suptitle(f'Comparison of Actual vs. Predicted values with Wind Angle = {wind_angle} in the {plane} Plane with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')

        # For Total Velocity
        vmin, vmax, cmap, scatter_size, ticks = plotting_details(filtered_df['Velocity_Magnitude_Actual'].values)
        contour_actual = axs[0,0].contourf(grid1, grid2, grid_magnitude_actual, levels=128, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[0,0].scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
        axs[0,0].tricontourf(tri.Triangulation(cylinderfill1, cylinderfill2), np.zeros(len(cylinderfill1)), colors='black', alpha=1)  # Fill color set to black
        cylinder_patch = patches.Patch(color='black', label='Cylinder Fill')
        cbar = fig.colorbar(contour_actual, ax=axs[0,0], ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
        cbar.set_label('Velocity Magnitude - Actual', rotation=270, labelpad=15)
        axs[0,0].set_xlabel(f'{coordinate1} Coordinate')
        axs[0,0].set_ylabel(f'{coordinate2} Coordinate')
        axs[0,0].set_xlim(lim_min1, lim_max1) 
        axs[0,0].set_ylim(lim_min2, lim_max2)
        axs[0,0].set_title(f'Total Actual Velocity in the {plane} Plane for Wind Angle = {angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')
        
        contour_pred = axs[0,1].contourf(grid1, grid2, grid_magnitude_pred, levels=128, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[0,1].scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
        axs[0,1].tricontourf(tri.Triangulation(cylinderfill1, cylinderfill2), np.zeros(len(cylinderfill1)), colors='black', alpha=1)  # Fill color set to black
        cylinder_patch = patches.Patch(color='black', label='Cylinder Fill')
        cbar = fig.colorbar(contour_pred, ax=axs[0,1], ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
        cbar.set_label('Velocity Magnitude - Predicted', rotation=270, labelpad=15)
        axs[0,1].set_xlabel(f'{coordinate1} Coordinate')
        axs[0,1].set_ylabel(f'{coordinate2} Coordinate')
        axs[0,1].set_xlim(lim_min1, lim_max1) 
        axs[0,1].set_ylim(lim_min2, lim_max2)
        axs[0,1].set_title(f'Total Predicted Velocity in the {plane} Plane for Wind Angle = {angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')
        
        # For Velocity X
        vmin, vmax, cmap, scatter_size, ticks = plotting_details(filtered_df['Velocity_X_Actual'].values)
        contour_vx_actual = axs[1,0].contourf(grid1, grid2, grid_vx_actual, levels=128, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[1,0].scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
        axs[1,0].tricontourf(tri.Triangulation(cylinderfill1, cylinderfill2), np.zeros(len(cylinderfill1)), colors='black', alpha=1)  # Fill color set to black
        cylinder_patch = patches.Patch(color='black', label='Cylinder Fill')
        cbar = fig.colorbar(contour_vx_actual, ax=axs[1, 0], ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
        cbar.set_label('Velocity X - Actual', rotation=270, labelpad=15)
        axs[1,0].set_xlabel(f'{coordinate1} Coordinate')
        axs[1,0].set_ylabel(f'{coordinate2} Coordinate')
        axs[1,0].set_xlim(lim_min1, lim_max1) 
        axs[1,0].set_ylim(lim_min2, lim_max2)
        axs[1,0].set_title(f'Actual Velocity X in the {plane} Plane for Wind Angle = {angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')

        contour_vx_pred = axs[1,1].contourf(grid1, grid2, grid_vx_pred, levels=128, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[1,1].scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
        axs[1,1].tricontourf(tri.Triangulation(cylinderfill1, cylinderfill2), np.zeros(len(cylinderfill1)), colors='black', alpha=1)  # Fill color set to black
        cylinder_patch = patches.Patch(color='black', label='Cylinder Fill')
        cbar = fig.colorbar(contour_vx_pred, ax=axs[1, 1], ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
        cbar.set_label('Velocity X - Predicted', rotation=270, labelpad=15)
        axs[1,1].set_xlabel(f'{coordinate1} Coordinate')
        axs[1,1].set_ylabel(f'{coordinate2} Coordinate')
        axs[1,1].set_xlim(lim_min1, lim_max1) 
        axs[1,1].set_ylim(lim_min2, lim_max2)
        axs[1,1].set_title(f'Predicted Velocity X in the {plane} Plane for Wind Angle = {angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')
        
        # For Velocity Y
        vmin, vmax, cmap, scatter_size, ticks = plotting_details(filtered_df['Velocity_Y_Actual'].values)
        contour_vy_actual = axs[2,0].contourf(grid1, grid2, grid_vy_actual, levels=128, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[2,0].scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
        axs[2,0].tricontourf(tri.Triangulation(cylinderfill1, cylinderfill2), np.zeros(len(cylinderfill1)), colors='black', alpha=1)  # Fill color set to black
        cylinder_patch = patches.Patch(color='black', label='Cylinder Fill')
        cbar = fig.colorbar(contour_vy_actual, ax=axs[2, 0], ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
        cbar.set_label('Velocity X - Actual', rotation=270, labelpad=15)
        axs[2,0].set_xlabel(f'{coordinate1} Coordinate')
        axs[2,0].set_ylabel(f'{coordinate2} Coordinate')
        axs[2,0].set_xlim(lim_min1, lim_max1) 
        axs[2,0].set_ylim(lim_min2, lim_max2)
        axs[2,0].set_title(f'Actual Velocity Y in the {plane} Plane for Wind Angle = {angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')

        contour_vy_pred = axs[2,1].contourf(grid1, grid2, grid_vy_pred, levels=128, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[2,1].scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
        axs[2,1].tricontourf(tri.Triangulation(cylinderfill1, cylinderfill2), np.zeros(len(cylinderfill1)), colors='black', alpha=1)  # Fill color set to black
        cylinder_patch = patches.Patch(color='black', label='Cylinder Fill')
        cbar = fig.colorbar(contour_vy_pred, ax=axs[2, 1], ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
        cbar.set_label('Velocity X - Predicted', rotation=270, labelpad=15)
        axs[2,1].set_xlabel(f'{coordinate1} Coordinate')
        axs[2,1].set_ylabel(f'{coordinate2} Coordinate')
        axs[2,1].set_xlim(lim_min1, lim_max1) 
        axs[2,1].set_ylim(lim_min2, lim_max2)
        axs[2,1].set_title(f'Predicted Velocity Y in the {plane} Plane for Wind Angle = {angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')

        # For Velocity Z
        vmin, vmax, cmap, scatter_size, ticks = plotting_details(filtered_df['Velocity_Z_Actual'].values)
        contour_vz_actual = axs[3,0].contourf(grid1, grid2, grid_vz_actual, levels=128, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[3,0].scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
        axs[3,0].tricontourf(tri.Triangulation(cylinderfill1, cylinderfill2), np.zeros(len(cylinderfill1)), colors='black', alpha=1)  # Fill color set to black
        cylinder_patch = patches.Patch(color='black', label='Cylinder Fill')
        cbar = fig.colorbar(contour_vz_actual, ax=axs[3, 0], ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
        cbar.set_label('Velocity Z - Actual', rotation=270, labelpad=15)
        axs[3,0].set_xlabel(f'{coordinate1} Coordinate')
        axs[3,0].set_ylabel(f'{coordinate2} Coordinate')
        axs[3,0].set_xlim(lim_min1, lim_max1) 
        axs[3,0].set_ylim(lim_min2, lim_max2)
        axs[3,0].set_title(f'Actual Velocity Z in the {plane} Plane for Wind Angle = {angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')

        contour_vz_pred = axs[3,1].contourf(grid1, grid2, grid_vz_pred, levels=128, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[3,1].scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
        axs[3,1].tricontourf(tri.Triangulation(cylinderfill1, cylinderfill2), np.zeros(len(cylinderfill1)), colors='black', alpha=1)  # Fill color set to black
        cylinder_patch = patches.Patch(color='black', label='Cylinder Fill')
        cbar = fig.colorbar(contour_vz_pred, ax=axs[3, 1], ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
        cbar.set_label('Velocity Z - Predicted', rotation=270, labelpad=15)
        axs[3,1].set_xlabel(f'{coordinate1} Coordinate')
        axs[3,1].set_ylabel(f'{coordinate2} Coordinate')
        axs[3,1].set_xlim(lim_min1, lim_max1) 
        axs[3,1].set_ylim(lim_min2, lim_max2)
        axs[3,1].set_title(f'Predicted Velocity Z in the {plane} Plane for Wind Angle = {angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')
        
        # For Pressure
        vmin, vmax, cmap, scatter_size, ticks = plotting_details(filtered_df['Pressure_Actual'].values)
        contour_pressure_actual = axs[4,0].contourf(grid1, grid2, grid_pressure_actual, levels=128, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[4,0].scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
        axs[4,0].tricontourf(tri.Triangulation(cylinderfill1, cylinderfill2), np.zeros(len(cylinderfill1)), colors='black', alpha=1)  # Fill color set to black
        cylinder_patch = patches.Patch(color='black', label='Cylinder Fill')
        cbar = fig.colorbar(contour_pressure_actual, ax=axs[4, 0], ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
        cbar.set_label('Pressure - Actual', rotation=270, labelpad=15)
        axs[4,0].set_xlabel(f'{coordinate1} Coordinate')
        axs[4,0].set_ylabel(f'{coordinate2} Coordinate')
        axs[4,0].set_xlim(lim_min1, lim_max1) 
        axs[4,0].set_ylim(lim_min2, lim_max2)
        axs[4,0].set_title(f'Actual Pressure in the {plane} Plane for Wind Angle = {angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')

        contour_pressure_pred = axs[4,1].contourf(grid1, grid2, grid_pressure_pred, levels=128, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[4,1].scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
        axs[4,1].tricontourf(tri.Triangulation(cylinderfill1, cylinderfill2), np.zeros(len(cylinderfill1)), colors='black', alpha=1)  # Fill color set to black
        cylinder_patch = patches.Patch(color='black', label='Cylinder Fill')
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

    if config["plotting"]["make_comparison_plots_total_velocity"]:
        
        # For Total Velocity
        if plane == 'X-Y':
            if config["plotting"]["make_comparison_plots_total_velocity_arrow"]:
                fig, axs = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
                fig.suptitle(f'Comparison of Actual vs. Predicted values with Wind Angle = {wind_angle} in the {plane} Plane with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')
                
                vmin, vmax, cmap, scatter_size, ticks = plotting_details(filtered_df['Velocity_Magnitude_Actual'].values)
                contour_actual = axs[0].contourf(grid1, grid2, grid_magnitude_actual, levels=128, vmin=vmin, vmax=vmax, cmap=cmap)
                if config["plotting"]["make_comparison_plots_quiver"]:
                    axs[0].quiver(quiver_points1, quiver_points2, quiver_v1_actual, quiver_v2_actual, scale=200, color='white')
                axs[0].scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
                axs[0].tricontourf(tri.Triangulation(cylinderfill1, cylinderfill2), np.zeros(len(cylinderfill1)), colors='black', alpha=1)  # Fill color set to black
                cylinder_patch = patches.Patch(color='black', label='Cylinder Fill')
                axs[0].add_patch(arrow10)
                axs[0].add_patch(arrow20)
                cbar = fig.colorbar(contour_actual, ax=axs[0], ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
                cbar.set_label('Velocity Magnitude - Actual', rotation=270, labelpad=15)
                axs[0].set_xlabel(f'{coordinate1} Coordinate')
                axs[0].set_ylabel(f'{coordinate2} Coordinate')
                axs[0].set_xlim(lim_min1, lim_max1) 
                axs[0].set_ylim(lim_min2, lim_max2)
                axs[0].set_title(f'Total Actual Velocity in the {plane} Plane for Wind Angle = {angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')
        
                contour_pred = axs[1].contourf(grid1, grid2, grid_magnitude_pred, levels=128, vmin=vmin, vmax=vmax, cmap=cmap)
                if config["plotting"]["make_comparison_plots_quiver"]:
                    axs[1].quiver(quiver_points1, quiver_points2, quiver_v1_pred, quiver_v2_pred, scale=200, color='white')
                axs[1].scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
                axs[1].tricontourf(tri.Triangulation(cylinderfill1, cylinderfill2), np.zeros(len(cylinderfill1)), colors='black', alpha=1)  # Fill color set to black
                cylinder_patch = patches.Patch(color='black', label='Cylinder Fill')
                axs[1].add_patch(arrow11)
                axs[1].add_patch(arrow21)
                cbar = fig.colorbar(contour_pred, ax=axs[1], ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
                cbar.set_label('Velocity Magnitude - Predicted', rotation=270, labelpad=15)
                axs[1].set_xlabel(f'{coordinate1} Coordinate')
                axs[1].set_ylabel(f'{coordinate2} Coordinate')
                axs[1].set_xlim(lim_min1, lim_max1) 
                axs[1].set_ylim(lim_min2, lim_max2)
                axs[1].set_title(f'Total Predicted Velocity in the {plane} Plane for Wind Angle = {angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')
                plt.tight_layout()
                plt.savefig(savename_total_velocity_arrow)
                plt.close()

        # For Total Velocity
        fig, axs = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
        fig.suptitle(f'Comparison of Actual vs. Predicted values with Wind Angle = {wind_angle} in the {plane} Plane with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')
        vmin, vmax, cmap, scatter_size, ticks = plotting_details(filtered_df['Velocity_Magnitude_Actual'].values)
        contour_actual = axs[0].contourf(grid1, grid2, grid_magnitude_actual, levels=128, vmin=vmin, vmax=vmax, cmap=cmap)
        if config["plotting"]["make_comparison_plots_quiver"]:
            axs[0].quiver(quiver_points1, quiver_points2, quiver_v1_actual, quiver_v2_actual, scale=200, color='white')
        axs[0].scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
        axs[0].tricontourf(tri.Triangulation(cylinderfill1, cylinderfill2), np.zeros(len(cylinderfill1)), colors='black', alpha=1)  # Fill color set to black
        cylinder_patch = patches.Patch(color='black', label='Cylinder Fill')
        cbar = fig.colorbar(contour_actual, ax=axs[0], ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
        cbar.set_label('Velocity Magnitude - Actual', rotation=270, labelpad=15)
        axs[0].set_xlabel(f'{coordinate1} Coordinate')
        axs[0].set_ylabel(f'{coordinate2} Coordinate')
        axs[0].set_xlim(lim_min1, lim_max1) 
        axs[0].set_ylim(lim_min2, lim_max2)
        axs[0].set_title(f'Total Actual Velocity in the {plane} Plane for Wind Angle = {angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')
        
        contour_pred = axs[1].contourf(grid1, grid2, grid_magnitude_pred, levels=128, vmin=vmin, vmax=vmax, cmap=cmap)
        if config["plotting"]["make_comparison_plots_quiver"]:
            axs[1].quiver(quiver_points1, quiver_points2, quiver_v1_pred, quiver_v2_pred, scale=200, color='white')
        axs[1].scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
        axs[1].tricontourf(tri.Triangulation(cylinderfill1, cylinderfill2), np.zeros(len(cylinderfill1)), colors='black', alpha=1)  # Fill color set to black
        cylinder_patch = patches.Patch(color='black', label='Cylinder Fill')
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

    if config["plotting"]["make_comparison_plots_vx"]:
        fig, axs = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
        fig.suptitle(f'Comparison of Actual vs. Predicted values with Wind Angle = {wind_angle} in the {plane} Plane with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')
        
        # For Velocity X
        vmin, vmax, cmap, scatter_size, ticks = plotting_details(filtered_df['Velocity_X_Actual'].values)
        contour_vx_actual = axs[0].contourf(grid1, grid2, grid_vx_actual, levels=128, vmin=vmin, vmax=vmax, cmap=cmap)
        if config["plotting"]["make_comparison_plots_quiver"]:
            axs[0].quiver(quiver_points1, quiver_points2, quiver_v1_actual, quiver_v2_actual, scale=200, color='white')
        axs[0].scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
        axs[0].tricontourf(tri.Triangulation(cylinderfill1, cylinderfill2), np.zeros(len(cylinderfill1)), colors='black', alpha=1)  # Fill color set to black
        cylinder_patch = patches.Patch(color='black', label='Cylinder Fill')
        cbar = fig.colorbar(contour_vx_actual, ax=axs[0], ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
        cbar.set_label('Velocity X - Actual', rotation=270, labelpad=15)
        axs[0].set_xlabel(f'{coordinate1} Coordinate')
        axs[0].set_ylabel(f'{coordinate2} Coordinate')
        axs[0].set_xlim(lim_min1, lim_max1) 
        axs[0].set_ylim(lim_min2, lim_max2)
        axs[0].set_title(f'Actual Velocity X in the {plane} Plane for Wind Angle = {angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')

        contour_vx_pred = axs[1].contourf(grid1, grid2, grid_vx_pred, levels=128, vmin=vmin, vmax=vmax, cmap=cmap)
        if config["plotting"]["make_comparison_plots_quiver"]:
            axs[1].quiver(quiver_points1, quiver_points2, quiver_v1_actual, quiver_v2_actual, scale=200, color='white')
        axs[1].scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
        axs[1].tricontourf(tri.Triangulation(cylinderfill1, cylinderfill2), np.zeros(len(cylinderfill1)), colors='black', alpha=1)  # Fill color set to black
        cylinder_patch = patches.Patch(color='black', label='Cylinder Fill')
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

    if config["plotting"]["make_comparison_plots_vy"]:
        fig, axs = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
        fig.suptitle(f'Comparison of Actual vs. Predicted values with Wind Angle = {wind_angle} in the {plane} Plane with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')

        # For Velocity Y
        vmin, vmax, cmap, scatter_size, ticks = plotting_details(filtered_df['Velocity_Y_Actual'].values)
        contour_vy_actual = axs[0].contourf(grid1, grid2, grid_vy_actual, levels=128, vmin=vmin, vmax=vmax, cmap=cmap)
        if config["plotting"]["make_comparison_plots_quiver"]:
                axs[0].quiver(quiver_points1, quiver_points2, quiver_v1_actual, quiver_v2_actual, scale=200, color='white')
        axs[0].scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
        axs[0].tricontourf(tri.Triangulation(cylinderfill1, cylinderfill2), np.zeros(len(cylinderfill1)), colors='black', alpha=1)  # Fill color set to black
        cylinder_patch = patches.Patch(color='black', label='Cylinder Fill')
        cbar = fig.colorbar(contour_vy_actual, ax=axs[0], ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
        cbar.set_label('Velocity Y - Actual', rotation=270, labelpad=15)
        axs[0].set_xlabel(f'{coordinate1} Coordinate')
        axs[0].set_ylabel(f'{coordinate2} Coordinate')
        axs[0].set_xlim(lim_min1, lim_max1) 
        axs[0].set_ylim(lim_min2, lim_max2)
        axs[0].set_title(f'Actual Velocity Y in the {plane} Plane for Wind Angle = {angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')

        contour_vy_pred = axs[1].contourf(grid1, grid2, grid_vy_pred, levels=128, vmin=vmin, vmax=vmax, cmap=cmap)
        if config["plotting"]["make_comparison_plots_quiver"]:
                axs[1].quiver(quiver_points1, quiver_points2, quiver_v1_actual, quiver_v2_actual, scale=200, color='white')
        axs[1].scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
        axs[1].tricontourf(tri.Triangulation(cylinderfill1, cylinderfill2), np.zeros(len(cylinderfill1)), colors='black', alpha=1)  # Fill color set to black
        cylinder_patch = patches.Patch(color='black', label='Cylinder Fill')
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

    if config["plotting"]["make_comparison_plots_vz"]:
        fig, axs = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
        fig.suptitle(f'Comparison of Actual vs. Predicted values with Wind Angle = {wind_angle} in the {plane} Plane with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')

        # For Velocity Z
        vmin, vmax, cmap, scatter_size, ticks = plotting_details(filtered_df['Velocity_Z_Actual'].values)
        contour_vz_actual = axs[0].contourf(grid1, grid2, grid_vz_actual, levels=128, vmin=vmin, vmax=vmax, cmap=cmap)
        if config["plotting"]["make_comparison_plots_quiver"]:
                axs[0].quiver(quiver_points1, quiver_points2, quiver_v1_actual, quiver_v2_actual, scale=200, color='white')
        axs[0].scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
        axs[0].tricontourf(tri.Triangulation(cylinderfill1, cylinderfill2), np.zeros(len(cylinderfill1)), colors='black', alpha=1)  # Fill color set to black
        cylinder_patch = patches.Patch(color='black', label='Cylinder Fill')
        cbar = fig.colorbar(contour_vz_actual, ax=axs[0], ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
        cbar.set_label('Velocity Z - Actual', rotation=270, labelpad=15)
        axs[0].set_xlabel(f'{coordinate1} Coordinate')
        axs[0].set_ylabel(f'{coordinate2} Coordinate')
        axs[0].set_xlim(lim_min1, lim_max1) 
        axs[0].set_ylim(lim_min2, lim_max2)
        axs[0].set_title(f'Actual Velocity Z in the {plane} Plane for Wind Angle = {angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')

        contour_vz_pred = axs[1].contourf(grid1, grid2, grid_vz_pred, levels=128, vmin=vmin, vmax=vmax, cmap=cmap)
        if config["plotting"]["make_comparison_plots_quiver"]:
                axs[1].quiver(quiver_points1, quiver_points2, quiver_v1_actual, quiver_v2_actual, scale=200, color='white')
        axs[1].scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
        axs[1].tricontourf(tri.Triangulation(cylinderfill1, cylinderfill2), np.zeros(len(cylinderfill1)), colors='black', alpha=1)  # Fill color set to black
        cylinder_patch = patches.Patch(color='black', label='Cylinder Fill')
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

    if config["plotting"]["make_comparison_plots_pressure"]:
        if 'grid_pressure_actual' in locals():
            fig, axs = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
            fig.suptitle(f'Comparison of Actual vs. Predicted values with Wind Angle = {wind_angle} in the {plane} Plane with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')

            # For Pressure
            vmin, vmax, cmap, scatter_size, ticks = plotting_details(filtered_df['Pressure_Actual'].values)
            contour_pressure_actual = axs[0].contourf(grid1, grid2, grid_pressure_actual, levels=128, vmin=vmin, vmax=vmax, cmap=cmap)
            if config["plotting"]["make_comparison_plots_quiver"]:
                    axs[0].quiver(quiver_points1, quiver_points2, quiver_v1_actual, quiver_v2_actual, scale=200, color='white')
            axs[0].scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
            axs[0].tricontourf(tri.Triangulation(cylinderfill1, cylinderfill2), np.zeros(len(cylinderfill1)), colors='black', alpha=1)  # Fill color set to black
            cylinder_patch = patches.Patch(color='black', label='Cylinder Fill')
            cbar = fig.colorbar(contour_pressure_actual, ax=axs[0], ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
            cbar.set_label('Pressure - Actual', rotation=270, labelpad=15)
            axs[0].set_xlabel(f'{coordinate1} Coordinate')
            axs[0].set_ylabel(f'{coordinate2} Coordinate')
            axs[0].set_xlim(lim_min1, lim_max1) 
            axs[0].set_ylim(lim_min2, lim_max2)
            axs[0].set_title(f'Actual Pressure in the {plane} Plane for Wind Angle = {angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')

            contour_pressure_pred = axs[1].contourf(grid1, grid2, grid_pressure_pred, levels=128, vmin=vmin, vmax=vmax, cmap=cmap)
            if config["plotting"]["make_comparison_plots_quiver"]:
                    axs[1].quiver(quiver_points1, quiver_points2, quiver_v1_actual, quiver_v2_actual, scale=200, color='white')
            axs[1].scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
            axs[1].tricontourf(tri.Triangulation(cylinderfill1, cylinderfill2), np.zeros(len(cylinderfill1)), colors='black', alpha=1)  # Fill color set to black
            cylinder_patch = patches.Patch(color='black', label='Cylinder Fill')
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
        else:
            pass

    if config["plotting"]["make_comparison_plots_turbvisc"]:
        if 'grid_turbvisc_actual' in locals():

            # For TurbVisc
            if plane == 'X-Y':
                if config["plotting"]["make_comparison_plots_turbvisc_arrow"]:

                    # For TurbVisc
                    fig, axs = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
                    fig.suptitle(f'Comparison of Actual vs. Predicted values with Wind Angle = {wind_angle} in the {plane} Plane with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')
                    vmin, vmax, cmap, scatter_size, ticks = plotting_details(filtered_df['TurbVisc_Actual'].values)
                    contour_turbvisc_actual = axs[0].contourf(grid1, grid2, grid_turbvisc_actual, levels=128, vmin=vmin, vmax=vmax, cmap=cmap)
                    if config["plotting"]["make_comparison_plots_quiver"]:
                        axs[0].quiver(quiver_points1, quiver_points2, quiver_v1_actual, quiver_v2_actual, scale=200, color='white')
                    axs[0].scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
                    axs[0].tricontourf(tri.Triangulation(cylinderfill1, cylinderfill2), np.zeros(len(cylinderfill1)), colors='black', alpha=1)  # Fill color set to black
                    cylinder_patch = patches.Patch(color='black', label='Cylinder Fill')
                    axs[0].add_patch(arrow10_)
                    axs[0].add_patch(arrow20_)
                    cbar = fig.colorbar(contour_turbvisc_actual, ax=axs[0], ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
                    cbar.set_label('TurbVisc - Actual', rotation=270, labelpad=15)
                    axs[0].set_xlabel(f'{coordinate1} Coordinate')
                    axs[0].set_ylabel(f'{coordinate2} Coordinate')
                    axs[0].set_xlim(lim_min1, lim_max1) 
                    axs[0].set_ylim(lim_min2, lim_max2)
                    axs[0].set_title(f'Actual TurbVisc in the {plane} Plane for Wind Angle = {angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')

                    contour_turbvisc_pred = axs[1].contourf(grid1, grid2, grid_turbvisc_pred, levels=128, vmin=vmin, vmax=vmax, cmap=cmap)
                    if config["plotting"]["make_comparison_plots_quiver"]:
                        axs[1].quiver(quiver_points1, quiver_points2, quiver_v1_actual, quiver_v2_actual, scale=200, color='white')
                    axs[1].scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
                    axs[1].tricontourf(tri.Triangulation(cylinderfill1, cylinderfill2), np.zeros(len(cylinderfill1)), colors='black', alpha=1)  # Fill color set to black
                    cylinder_patch = patches.Patch(color='black', label='Cylinder Fill')
                    axs[1].add_patch(arrow11_)
                    axs[1].add_patch(arrow21_)
                    cbar = fig.colorbar(contour_turbvisc_pred, ax=axs[1], ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
                    cbar.set_label('TurbVisc - Predicted', rotation=270, labelpad=15)
                    axs[1].set_xlabel(f'{coordinate1} Coordinate')
                    axs[1].set_ylabel(f'{coordinate2} Coordinate')
                    axs[1].set_xlim(lim_min1, lim_max1) 
                    axs[1].set_ylim(lim_min2, lim_max2)
                    axs[1].set_title(f'Predicted TurbVisc in the {plane} Plane for Wind Angle = {angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')
                    plt.tight_layout()
                    plt.savefig(savename_turbvisc_arrow)
                    plt.close()

            # For TurbVisc
            fig, axs = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
            fig.suptitle(f'Comparison of Actual vs. Predicted values with Wind Angle = {wind_angle} in the {plane} Plane with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')
            vmin, vmax, cmap, scatter_size, ticks = plotting_details(filtered_df['TurbVisc_Actual'].values)
            contour_turbvisc_actual = axs[0].contourf(grid1, grid2, grid_turbvisc_actual, levels=128, vmin=vmin, vmax=vmax, cmap=cmap)
            if config["plotting"]["make_comparison_plots_quiver"]:
                axs[0].quiver(quiver_points1, quiver_points2, quiver_v1_actual, quiver_v2_actual, scale=200, color='white')
            axs[0].scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
            axs[0].tricontourf(tri.Triangulation(cylinderfill1, cylinderfill2), np.zeros(len(cylinderfill1)), colors='black', alpha=1)  # Fill color set to black
            cylinder_patch = patches.Patch(color='black', label='Cylinder Fill')
            cbar = fig.colorbar(contour_turbvisc_actual, ax=axs[0], ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
            cbar.set_label('TurbVisc - Actual', rotation=270, labelpad=15)
            axs[0].set_xlabel(f'{coordinate1} Coordinate')
            axs[0].set_ylabel(f'{coordinate2} Coordinate')
            axs[0].set_xlim(lim_min1, lim_max1) 
            axs[0].set_ylim(lim_min2, lim_max2)
            axs[0].set_title(f'Actual TurbVisc in the {plane} Plane for Wind Angle = {angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')

            contour_turbvisc_pred = axs[1].contourf(grid1, grid2, grid_turbvisc_pred, levels=128, vmin=vmin, vmax=vmax, cmap=cmap)
            if config["plotting"]["make_comparison_plots_quiver"]:
                axs[1].quiver(quiver_points1, quiver_points2, quiver_v1_actual, quiver_v2_actual, scale=200, color='white')
            axs[1].scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
            axs[1].tricontourf(tri.Triangulation(cylinderfill1, cylinderfill2), np.zeros(len(cylinderfill1)), colors='black', alpha=1)  # Fill color set to black
            cylinder_patch = patches.Patch(color='black', label='Cylinder Fill')
            cbar = fig.colorbar(contour_turbvisc_pred, ax=axs[1], ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
            cbar.set_label('TurbVisc - Predicted', rotation=270, labelpad=15)
            axs[1].set_xlabel(f'{coordinate1} Coordinate')
            axs[1].set_ylabel(f'{coordinate2} Coordinate')
            axs[1].set_xlim(lim_min1, lim_max1) 
            axs[1].set_ylim(lim_min2, lim_max2)
            axs[1].set_title(f'Predicted TurbVisc in the {plane} Plane for Wind Angle = {angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')
            plt.tight_layout()
            plt.savefig(savename_turbvisc)
            plt.close()

def plot_data_new_angles(df,angle,config,datafolder_path,savename_total_velocity,savename_total_velocity_arrow,savename_vx,savename_vy,savename_vz,savename_pressure,savename_turbvisc,savename_turbvisc_arrow,plane,cut,tolerance):
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
        arrow_x1 = 500
        arrow_y1 = 500
        arrow_x2 = 500
        arrow_y2 = 570
        length = 10000000
        wind_angle_modf = 270 - wind_angle
        wind_angle_rad_modf = np.deg2rad(wind_angle_modf)
        arrow_dx1 = length*(np.cos(wind_angle_rad_modf))
        arrow_dy1 = length*(np.sin(wind_angle_rad_modf))
        arrow_dx2 = length*(np.cos(wind_angle_rad_modf))
        arrow_dy2 = length*(np.sin(wind_angle_rad_modf))
        arrow_x1, arrow_y1 = normalize_grids(plane, arrow_x1, arrow_y1)
        arrow_dx1, arrow_dy1 = normalize_grids(plane, arrow_dx1, arrow_dy1)
        arrow_x2, arrow_y2 = normalize_grids(plane, arrow_x2, arrow_y2)
        arrow_dx2, arrow_dy2 = normalize_grids(plane, arrow_dx2, arrow_dy2)
        arrow1 = patches.Arrow(arrow_x1, arrow_y1, arrow_dx1, arrow_dy1, width=0.05, color="white")
        arrow2 = patches.Arrow(arrow_x2, arrow_y2, arrow_dx2, arrow_dy2, width=0.05, color="white")
        arrow1_ = patches.Arrow(arrow_x1, arrow_y1, arrow_dx1, arrow_dy1, width=0.05, color="white")
        arrow2_ = patches.Arrow(arrow_x2, arrow_y2, arrow_dx2, arrow_dy2, width=0.05, color="white")

    points1 = points[0]
    points2 = points[1]

    coordinate1 = plane.split('-')[0]
    coordinate2 = plane.split('-')[1]

    # Filter the data to focus on the y-z plane/...
    filtered_df = df[(df[noplotdata] >= cut - tolerance) & (df[noplotdata] <= cut + tolerance)]

    # Define a regular grid covering the range of y and z coordinates/...
    grid1, grid2 = np.mgrid[filtered_df[points1].min():filtered_df[points1].max():1000j, filtered_df[points2].min():filtered_df[points2].max():1000j]

    # Interpolate all velocity components onto the grid
    try:
        grid_vx = griddata(filtered_df[[points1, points2]].values, filtered_df['Velocity_X'].values, (grid1, grid2), method='linear', fill_value=0)
        grid_vy = griddata(filtered_df[[points1, points2]].values, filtered_df['Velocity_Y'].values, (grid1, grid2), method='linear', fill_value=0)
        grid_vz = griddata(filtered_df[[points1, points2]].values, filtered_df['Velocity_Z'].values, (grid1, grid2), method='linear', fill_value=0)
        if 'Pressure' in filtered_df.columns:
            grid_pressure = griddata(filtered_df[[points1, points2]].values, filtered_df['Pressure'].values, (grid1, grid2), method='linear', fill_value=0)
        if 'TurbVisc' in filtered_df.columns:
            grid_turbvisc = griddata(filtered_df[[points1, points2]].values, filtered_df['TurbVisc'].values, (grid1, grid2), method='linear', fill_value=0)
    except:
        print(f"not enough points")
        return

    # Calculate the total magnitude of the velocity on the grid
    grid_magnitude = np.sqrt(grid_vx**2 + grid_vy**2 + grid_vz**2)

    spherefill1, spherefill2, cylinderfill1, cylinderfill2 = geometry_coordinates(plane, datafolder_path)

    grid1, grid2 = normalize_grids(plane, grid1, grid2)
    spherefill1, spherefill2 = normalize_grids(plane, spherefill1, spherefill2)
    cylinderfill1, cylinderfill2 = normalize_grids(plane, cylinderfill1, cylinderfill2)
    
    cut, tolerance = normalize_cut_tolerance(plane, cut, tolerance)

    if config["plotting"]["make_new_angle_plots_quiver"]:
        # Decide on quiver density (e.g., every 20th point)
        stride = 100
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

    if config["plotting"]["make_new_angle_plots_total_velocity"] and not Path(savename_total_velocity).exists():

        # For Total Velocity
        if plane == 'X-Y':
            if config["plotting"]["make_new_angle_plots_total_velocity_arrow"]:
                fig, ax = plt.subplots(figsize=(16, 8), sharey=True)
                vmin, vmax, cmap, scatter_size, ticks = plotting_details(filtered_df['Velocity_Magnitude'].values)
                contour = ax.contourf(grid1, grid2, grid_magnitude, levels=128, vmin=vmin, vmax=vmax, cmap=cmap)
                if config["plotting"]["make_new_angle_plots_quiver"]:
                    ax.quiver(quiver_points1, quiver_points2, quiver_v1, quiver_v2, scale=200, color='white')
                ax.scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
                ax.tricontourf(tri.Triangulation(cylinderfill1, cylinderfill2), np.zeros(len(cylinderfill1)), colors='black', alpha=1)  # Fill color set to black
                cylinder_patch = patches.Patch(color='black', label='Cylinder Fill')
                # plt.legend(handles=[cylinder_patch])
                
                ax.add_patch(arrow1)
                ax.add_patch(arrow2)
                cbar = fig.colorbar(contour, ax=ax, ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
                cbar.set_label('Velocity Magnitude', rotation=270, labelpad=15)
                ax.set_xlabel(f'{coordinate1} Coordinate')
                ax.set_ylabel(f'{coordinate2} Coordinate')
                ax.set_xlim(lim_min1, lim_max1) 
                ax.set_ylim(lim_min2, lim_max2)
                ax.set_title(f'Total Velocity in the {plane} Plane for Wind Angle = {angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')
                plt.tight_layout()
                plt.savefig(savename_total_velocity_arrow)
                plt.close()

        # For Total Velocity
        fig, ax = plt.subplots(figsize=(16, 8), sharey=True)
        vmin, vmax, cmap, scatter_size, ticks = plotting_details(filtered_df['Velocity_Magnitude'].values)
        contour = ax.contourf(grid1, grid2, grid_magnitude, levels=128, vmin=vmin, vmax=vmax, cmap=cmap)
        if config["plotting"]["make_new_angle_plots_quiver"]:
            ax.quiver(quiver_points1, quiver_points2, quiver_v1, quiver_v2, scale=200, color='white')
        ax.scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
        ax.tricontourf(tri.Triangulation(cylinderfill1, cylinderfill2), np.zeros(len(cylinderfill1)), colors='black', alpha=1)  # Fill color set to black
        cylinder_patch = patches.Patch(color='black', label='Cylinder Fill')
        # plt.legend(handles=[cylinder_patch])
        
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

    if config["plotting"]["make_new_angle_plots_vx"] and not Path(savename_vx).exists():
        fig, ax = plt.subplots(figsize=(16, 8), sharey=True)

        # For Velocity_X
        vmin, vmax, cmap, scatter_size, ticks = plotting_details(filtered_df['Velocity_X'].values)
        contour = ax.contourf(grid1, grid2, grid_vx, levels=128, vmin=vmin, vmax=vmax, cmap=cmap)
        if config["plotting"]["make_new_angle_plots_quiver"]:
            ax.quiver(quiver_points1, quiver_points2, quiver_v1, quiver_v2, scale=200, color='white')
        ax.scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
        ax.tricontourf(tri.Triangulation(cylinderfill1, cylinderfill2), np.zeros(len(cylinderfill1)), colors='black', alpha=1)  # Fill color set to black
        cylinder_patch = patches.Patch(color='black', label='Cylinder Fill')
        # plt.legend(handles=[cylinder_patch])
        
        cbar = fig.colorbar(contour, ax=ax, ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
        cbar.set_label('Velocity X', rotation=270, labelpad=15)
        ax.set_xlabel(f'{coordinate1} Coordinate')
        ax.set_ylabel(f'{coordinate2} Coordinate')
        ax.set_xlim(lim_min1, lim_max1) 
        ax.set_ylim(lim_min2, lim_max2)
        ax.set_title(f'Velocity X in the {plane} Plane for Wind Angle = {angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')
        
        plt.tight_layout()
        plt.savefig(savename_vx)
        plt.close()

    if config["plotting"]["make_new_angle_plots_vy"] and not Path(savename_vy).exists():
        fig, ax = plt.subplots(figsize=(16, 8), sharey=True)

        # For Velocity_Y
        vmin, vmax, cmap, scatter_size, ticks = plotting_details(filtered_df['Velocity_Y'].values)
        contour = ax.contourf(grid1, grid2, grid_vy, levels=128, vmin=vmin, vmax=vmax, cmap=cmap)
        if config["plotting"]["make_new_angle_plots_quiver"]:
            ax.quiver(quiver_points1, quiver_points2, quiver_v1, quiver_v2, scale=200, color='white')
        ax.scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
        ax.tricontourf(tri.Triangulation(cylinderfill1, cylinderfill2), np.zeros(len(cylinderfill1)), colors='black', alpha=1)  # Fill color set to black
        cylinder_patch = patches.Patch(color='black', label='Cylinder Fill')
        # plt.legend(handles=[cylinder_patch])
        
        cbar = fig.colorbar(contour, ax=ax, ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
        cbar.set_label('Velocity Y', rotation=270, labelpad=15)
        ax.set_xlabel(f'{coordinate1} Coordinate')
        ax.set_ylabel(f'{coordinate2} Coordinate')
        ax.set_xlim(lim_min1, lim_max1) 
        ax.set_ylim(lim_min2, lim_max2)
        ax.set_title(f'Velocity Y in the {plane} Plane for Wind Angle = {angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')
        
        plt.tight_layout()
        plt.savefig(savename_vy)
        plt.close()

    if config["plotting"]["make_new_angle_plots_vz"] and not Path(savename_vz).exists():
        fig, ax = plt.subplots(figsize=(16, 8), sharey=True)

        # For Velocity_Z
        vmin, vmax, cmap, scatter_size, ticks = plotting_details(filtered_df['Velocity_Z'].values)
        contour = ax.contourf(grid1, grid2, grid_vz, levels=128, vmin=vmin, vmax=vmax, cmap=cmap)
        if config["plotting"]["make_new_angle_plots_quiver"]:
            ax.quiver(quiver_points1, quiver_points2, quiver_v1, quiver_v2, scale=200, color='white')
        ax.scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
        ax.tricontourf(tri.Triangulation(cylinderfill1, cylinderfill2), np.zeros(len(cylinderfill1)), colors='black', alpha=1)  # Fill color set to black
        cylinder_patch = patches.Patch(color='black', label='Cylinder Fill')
        # plt.legend(handles=[cylinder_patch])
        
        cbar = fig.colorbar(contour, ax=ax, ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
        cbar.set_label('Velocity Z', rotation=270, labelpad=15)
        ax.set_xlabel(f'{coordinate1} Coordinate')
        ax.set_ylabel(f'{coordinate2} Coordinate')
        ax.set_xlim(lim_min1, lim_max1) 
        ax.set_ylim(lim_min2, lim_max2)
        ax.set_title(f'Velocity Z in the {plane} Plane for Wind Angle = {angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')
        
        plt.tight_layout()
        plt.savefig(savename_vz)
        plt.close()

    if config["plotting"]["make_new_angle_plots_pressure"] and not Path(savename_pressure).exists():
        if 'grid_pressure' in locals():
            fig, ax = plt.subplots(figsize=(16, 8), sharey=True)

            # For Pressure
            vmin, vmax, cmap, scatter_size, ticks = plotting_details(filtered_df['Pressure'].values)
            contour = ax.contourf(grid1, grid2, grid_pressure, levels=128, vmin=vmin, vmax=vmax, cmap=cmap)
            if config["plotting"]["make_new_angle_plots_quiver"]:
                ax.quiver(quiver_points1, quiver_points2, quiver_v1, quiver_v2, scale=200, color='white')
            ax.scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
            ax.tricontourf(tri.Triangulation(cylinderfill1, cylinderfill2), np.zeros(len(cylinderfill1)), colors='black', alpha=1)  # Fill color set to black
            cylinder_patch = patches.Patch(color='black', label='Cylinder Fill')
            # plt.legend(handles=[cylinder_patch])
            
            cbar = fig.colorbar(contour, ax=ax, ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
            cbar.set_label('Pressure', rotation=270, labelpad=15)
            ax.set_xlabel(f'{coordinate1} Coordinate')
            ax.set_ylabel(f'{coordinate2} Coordinate')
            ax.set_xlim(lim_min1, lim_max1) 
            ax.set_ylim(lim_min2, lim_max2)
            ax.set_title(f'Pressure in the {plane} Plane for Wind Angle = {angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')
            
            plt.tight_layout()
            plt.savefig(savename_pressure)
            plt.close()

    if config["plotting"]["make_new_angle_plots_turbvisc"] and not Path(savename_turbvisc).exists():
        if 'grid_turbvisc' in locals():

            # For TurbVisc
            if plane == 'X-Y':
                if config["plotting"]["make_new_angle_plots_turbvisc_arrow"]:
                    fig, ax = plt.subplots(figsize=(16, 8), sharey=True)
                    vmin, vmax, cmap, scatter_size, ticks = plotting_details(filtered_df['TurbVisc'].values)
                    contour = ax.contourf(grid1, grid2, grid_turbvisc, levels=128, vmin=vmin, vmax=vmax, cmap=cmap)
                    if config["plotting"]["make_new_angle_plots_quiver"]:
                        ax.quiver(quiver_points1, quiver_points2, quiver_v1, quiver_v2, scale=200, color='white')
                    ax.scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
                    ax.tricontourf(tri.Triangulation(cylinderfill1, cylinderfill2), np.zeros(len(cylinderfill1)), colors='black', alpha=1)  # Fill color set to black
                    cylinder_patch = patches.Patch(color='black', label='Cylinder Fill')
                    # plt.legend(handles=[cylinder_patch])
                    
                    ax.add_patch(arrow1_)
                    ax.add_patch(arrow2_)
                    cbar = fig.colorbar(contour, ax=ax, ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
                    cbar.set_label('TurbVisc', rotation=270, labelpad=15)
                    ax.set_xlabel(f'{coordinate1} Coordinate')
                    ax.set_ylabel(f'{coordinate2} Coordinate')
                    ax.set_xlim(lim_min1, lim_max1) 
                    ax.set_ylim(lim_min2, lim_max2)
                    ax.set_title(f'TurbVisc in the {plane} Plane for Wind Angle = {angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')
                    plt.tight_layout()
                    plt.savefig(savename_turbvisc)
                    plt.close()
            
            # For TurbVisc
            fig, ax = plt.subplots(figsize=(16, 8), sharey=True)
            vmin, vmax, cmap, scatter_size, ticks = plotting_details(filtered_df['TurbVisc'].values)
            contour = ax.contourf(grid1, grid2, grid_turbvisc, levels=128, vmin=vmin, vmax=vmax, cmap=cmap)
            if config["plotting"]["make_new_angle_plots_quiver"]:
                ax.quiver(quiver_points1, quiver_points2, quiver_v1, quiver_v2, scale=200, color='white')
            ax.scatter(spherefill1, spherefill2, c='black', s=1, label='Sphere Fill')
            ax.tricontourf(tri.Triangulation(cylinderfill1, cylinderfill2), np.zeros(len(cylinderfill1)), colors='black', alpha=1)  # Fill color set to black
            cylinder_patch = patches.Patch(color='black', label='Cylinder Fill')
            # plt.legend(handles=[cylinder_patch])
            
            cbar = fig.colorbar(contour, ax=ax, ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
            cbar.set_label('TurbVisc', rotation=270, labelpad=15)
            ax.set_xlabel(f'{coordinate1} Coordinate')
            ax.set_ylabel(f'{coordinate2} Coordinate')
            ax.set_xlim(lim_min1, lim_max1) 
            ax.set_ylim(lim_min2, lim_max2)
            ax.set_title(f'TurbVisc in the {plane} Plane for Wind Angle = {angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')
            plt.tight_layout()
            plt.savefig(savename_turbvisc)
            plt.close()

def plot_data_2d(df,angle,config,plot_folder):
    params = [['X-Z',570,5],['Y-Z',500,5],['X-Y',50,5]]
    chosen_machine_key = config["chosen_machine"]
    datafolder_path = os.path.join(config["machine"][chosen_machine_key], config["data"]["data_folder_name"])
    for j in params:
        plane = j[0]
        cut = j[1]
        tolerance = j[2]
        savename_total_velocity = os.path.join(plot_folder,f'{plane}_totalvelocity_{angle}.png')
        savename_total_velocity_arrow = os.path.join(plot_folder,f'{plane}_totalvelocityarrow_{angle}.png')
        savename_vx = os.path.join(plot_folder,f'{plane}_vx_{angle}.png')
        savename_vy = os.path.join(plot_folder,f'{plane}_vy_{angle}.png')
        savename_vz = os.path.join(plot_folder,f'{plane}_vz_{angle}.png')
        savename_pressure = os.path.join(plot_folder,f'{plane}_pressure_{angle}.png')
        savename_turbvisc = os.path.join(plot_folder,f'{plane}_turbvisc_{angle}.png')
        savename_turbvisc_arrow = os.path.join(plot_folder,f'{plane}_turbviscarrow_{angle}.png')
        plot_data(df,angle,config,datafolder_path,savename_total_velocity,savename_total_velocity_arrow,savename_vx,savename_vy,savename_vz,savename_pressure,savename_turbvisc,savename_turbvisc_arrow,plane,cut,tolerance)

def plot_prediction_2d(df,angle,config,plot_folder):
    params = [['X-Z',570,5],['Y-Z',500,5],['X-Y',50,5]]
    chosen_machine_key = config["chosen_machine"]
    datafolder_path = os.path.join(config["machine"][chosen_machine_key], config["data"]["data_folder_name"])
    for j in params:
        plane = j[0]
        cut = j[1]
        tolerance = j[2]
        savename_all = os.path.join(plot_folder,f'{plane}_allplots_{angle}.png')
        savename_total_velocity = os.path.join(plot_folder,f'{plane}_totalvelocity_{angle}.png')
        savename_total_velocity_arrow = os.path.join(plot_folder,f'{plane}_totalvelocityarrow_{angle}.png')
        savename_vx = os.path.join(plot_folder,f'{plane}_vx_{angle}.png')
        savename_vy = os.path.join(plot_folder,f'{plane}_vy_{angle}.png')
        savename_vz = os.path.join(plot_folder,f'{plane}_vz_{angle}.png')
        savename_pressure = os.path.join(plot_folder,f'{plane}_pressure_{angle}.png')
        savename_turbvisc = os.path.join(plot_folder,f'{plane}_turbvisc_{angle}.png')
        savename_turbvisc_arrow = os.path.join(plot_folder,f'{plane}_turbviscarrow_{angle}.png')
        plot_data_predictions(df,angle,config,datafolder_path,savename_all,savename_total_velocity,savename_total_velocity_arrow,savename_vx,savename_vy,savename_vz,savename_pressure,savename_turbvisc,savename_turbvisc_arrow,plane,cut,tolerance)

def plot_new_angles_2d(df,angle,config,plot_folder):
    params = [['X-Z',570,5],['Y-Z',500,5],['X-Y',50,5]]
    chosen_machine_key = config["chosen_machine"]
    datafolder_path = os.path.join(config["machine"][chosen_machine_key], config["data"]["data_folder_name"])
    for j in params:
        plane = j[0]
        cut = j[1]
        tolerance = j[2]
        savename_total_velocity = os.path.join(plot_folder,f'{plane}_totalvelocity_{angle}.png')
        savename_total_velocity_arrow = os.path.join(plot_folder,f'{plane}_totalvelocityarrow_{angle}.png')
        savename_vx = os.path.join(plot_folder,f'{plane}_vx_{angle}.png')
        savename_vy = os.path.join(plot_folder,f'{plane}_vy_{angle}.png')
        savename_vz = os.path.join(plot_folder,f'{plane}_vz_{angle}.png')
        savename_pressure = os.path.join(plot_folder,f'{plane}_pressure_{angle}.png')
        savename_turbvisc = os.path.join(plot_folder,f'{plane}_turbvisc_{angle}.png')
        savename_turbvisc_arrow = os.path.join(plot_folder,f'{plane}_turbviscarrow_{angle}.png')
        plot_data_new_angles(df,angle,config,datafolder_path,savename_total_velocity,savename_total_velocity_arrow,savename_vx,savename_vy,savename_vz,savename_pressure,savename_turbvisc,savename_turbvisc_arrow,plane,cut,tolerance)

def make_logging_plots(directory, df, current_time, num_epochs, epoch_time, save_plot, save_plot_xlog, save_plot_ylog, save_plot_ylog_xlog, xlog, ylog):

    # Selecting the columns for epoch and the losses
    epoch = df['Epoch']
    losses = df.loc[:, 'Data Loss':'Total Loss']

    # Define a list of colors to cycle through
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    # Plotting with color cycling
    plt.figure(figsize=(16, 8))
    for i, column in enumerate(losses.columns):
        color = colors[i % len(colors)]  # Cycle through the colors
        plt.plot(epoch, losses[column], label=column, color=color)

    plt.title(f'Epoch vs Loss - Time Taken = {current_time:.2f} hours for {num_epochs} Epochs; Time Taken Per Epoch = {epoch_time:.2f} hours')
    plt.legend()
    plt.grid(True)

    if xlog and not ylog:
        plt.xlabel('Epoch (log)')
        plt.ylabel('Loss')
        plt.xscale('log')
        plt.savefig(os.path.join(directory, save_plot_xlog))
        plt.close()
    elif ylog and not xlog:
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log)')
        plt.yscale('log')
        plt.savefig(os.path.join(directory, save_plot_ylog))
        plt.close()
    elif xlog and ylog:
        plt.xlabel('Epoch (log)')
        plt.ylabel('Loss (log)')
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig(os.path.join(directory, save_plot_ylog_xlog))
        plt.close()
    elif not xlog and not ylog:
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(directory, save_plot))
        plt.close()

def make_test_plots(directory, filename, savefigname):

    df = filter_trainingloss_file(directory, filename)
    epoch = df['Epoch']
    df_mse = df.loc[:, 'MSE_Velocity_X':'MSE_Velocity_Z']
    df_r2 = df.loc[:, 'R2_Velocity_X':'R2_Velocity_Z']

    # Define a list of colors to cycle through
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    # Plotting with color cycling
    fig, axs = plt.subplots(1, 2, figsize=(16, 8), sharey=False)
    axs[0].set_title(f'MSE Loss')
    axs[1].set_title(f'R2')
    for i, column in enumerate(df_mse.columns):
        color = colors[i % len(colors)]  # Cycle through the colors
        axs[0].plot(epoch, df_mse[column], label=column, color=color)
    for i, column in enumerate(df_r2.columns):
        color = colors[i % len(colors)]  # Cycle through the colors
        axs[1].plot(epoch, df_r2[column], label=column, color=color)

    axs[0].legend()
    axs[1].legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(directory,savefigname))
    plt.close()

def make_all_logging_plots(directory, df, current_time, total_epochs):
    save_plot = 'save_logging_plot.png'
    save_plot_xlog = 'save_logging_plot_xlog.png'
    save_plot_ylog = 'save_logging_plot_ylog.png'
    save_plot_ylog_xlog = 'save_logging_plot_ylog_xlog.png'

    info_test = 'info_test.csv'
    info_skipped = 'info_skipped.csv'

    save_testing_loss = 'save_testing_loss.png'
    save_skipped_loss = 'save_skipped_loss.png'

    xlog_ylog = [[True, True], [True, False], [False, True], [False, False]]

    epoch_time = current_time/total_epochs

    for xlog, ylog in xlog_ylog:
        make_logging_plots(directory, df, current_time, total_epochs, epoch_time, save_plot, save_plot_xlog, save_plot_ylog, save_plot_ylog_xlog, xlog, ylog)

    make_test_plots(directory, info_test, save_testing_loss)
    make_test_plots(directory, info_skipped, save_skipped_loss)