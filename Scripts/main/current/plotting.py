from definitions import *
from plotting_definitions import *

def make_scatter_plot(figsize, z, grid1, grid2, grid3, geometry, plane, wind_angle, cut, tolerance, cbar_label, savename):
    coordinate1, coordinate2, coordinate3, lim_min1, lim_max1, lim_min2, lim_max2 = get_plane_config(plane)
    fig, ax = plt.subplots(figsize=figsize, sharey=True)
    vmin, vmax, levels, cmap, scatter_size, ticks = plotting_details(z)
    scatter = ax.scatter(grid1, grid2, c=grid3, s=scatter_size, vmin=min(z), vmax=max(z), cmap=cmap)
    scatter_cbar = fig.colorbar(scatter, ax=ax)
    scatter_cbar.set_label(f'Scatter {cbar_label}', rotation=270, labelpad=15)
    if config["plotting"]["plot_geometry"]:
        plot_geometry(plane, geometry, scatter_size, ax)
    ax.set_xlabel(f'{coordinate1} Coordinate')
    ax.set_ylabel(f'{coordinate2} Coordinate')
    ax.set_xlim(lim_min1, lim_max1) 
    ax.set_ylim(lim_min2, lim_max2)
    ax.set_title(f'{cbar_label} in the {plane} Plane for Wind Angle = {wind_angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f} with mean = {np.mean(z):.2f} and std dev = {np.std(z):.2f}')
    plt.tight_layout()
    plt.savefig(savename)
    plt.close()

def make_singular_plot(figsize, z, grid1, grid2, grid3, geometry, quiver_points1, quiver_points2, quiver_v1, quiver_v2, plane, wind_angle, cut, tolerance, cbar_label, savename, arrow1= None, arrow2 = None):
    coordinate1, coordinate2, coordinate3, lim_min1, lim_max1, lim_min2, lim_max2 = get_plane_config(plane)
    fig, ax = plt.subplots(figsize=figsize, sharey=True)
    vmin, vmax, levels, cmap, scatter_size, ticks = plotting_details(z)
    contour = ax.contourf(grid1, grid2, grid3, levels=levels, vmin=np.min(z_actual), vmax=np.max(z_actual), cmap=cmap)
    ax.quiver(quiver_points1, quiver_points2, quiver_v1, quiver_v2, scale=200, color='white')
    plot_geometry(plane, geometry, scatter_size, ax)
    if arrow1 is not None and arrow2 is not None:
        ax.add_patch(arrow1)
        ax.add_patch(arrow2)
    cbar = fig.colorbar(contour, ax=ax, ticks=ticks)
    cbar.set_label(cbar_label, rotation=270, labelpad=15)
    ax.set_xlabel(f'{coordinate1} Coordinate')
    ax.set_ylabel(f'{coordinate2} Coordinate')
    ax.set_xlim(lim_min1, lim_max1) 
    ax.set_ylim(lim_min2, lim_max2)
    ax.set_title(f'{cbar_label} in the {plane} Plane for Wind Angle = {wind_angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')
    plt.tight_layout()
    plt.savefig(savename)
    plt.close()

def make_double_plots(figsize, z_actual, z_pred, grid1, grid2, grid3_actual, grid3_pred, geometry, quiver_points1, quiver_points2, quiver_v1_actual, quiver_v2_actual, quiver_v1_pred, quiver_v2_pred, plane, wind_angle, cut, tolerance, cbar_label, savename, arrow10 = None, arrow20 = None, arrow11 = None, arrow21 = None):
    coordinate1, coordinate2, coordinate3, lim_min1, lim_max1, lim_min2, lim_max2 = get_plane_config(plane)
    fig, axs = plt.subplots(1, 2, figsize=figsize, sharey=True)
    fig.suptitle(f'Comparison of Actual vs. Predicted values with Wind Angle = {wind_angle} in the {plane} Plane with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')
    vmin, vmax, levels, cmap, scatter_size, ticks = plotting_details(z_actual)
    
    contour_actual = axs[0].contourf(grid1, grid2, grid3_actual, levels=levels, vmin=np.min(z_actual), vmax=np.max(z_actual), cmap=cmap)
    axs[0].quiver(quiver_points1, quiver_points2, quiver_v1_actual, quiver_v2_actual, scale=200, color='white')
    plot_geometry(plane, geometry, scatter_size, axs[0])
    if arrow10 is not None and arrow20 is not None:
        axs[0].add_patch(arrow10)
        axs[0].add_patch(arrow20)
    cbar = fig.colorbar(contour_actual, ax=axs[0], ticks=ticks)
    cbar.set_label(f'{cbar_label} - Actual', rotation=270, labelpad=15)
    axs[0].set_xlabel(f'{coordinate1} Coordinate')
    axs[0].set_ylabel(f'{coordinate2} Coordinate')
    axs[0].set_xlim(lim_min1, lim_max1) 
    axs[0].set_ylim(lim_min2, lim_max2)
    axs[0].set_title(f'Actual {cbar_label} in the {plane} Plane for Wind Angle = {wind_angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f} \n Mean = {np.mean(z_actual):.2f} and Standard Deviation = {np.std(z_actual):.2f}')
    
    contour_pred = axs[1].contourf(grid1, grid2, grid3_pred, levels=levels, vmin=np.min(z_actual), vmax=np.max(z_actual), cmap=cmap)
    axs[1].quiver(quiver_points1, quiver_points2, quiver_v1_pred, quiver_v2_pred, scale=200, color='white')
    plot_geometry(plane, geometry, scatter_size, axs[1])
    if arrow11 is not None and arrow21 is not None:
        axs[1].add_patch(arrow11)
        axs[1].add_patch(arrow21)
    cbar = fig.colorbar(contour_pred, ax=axs[1], ticks=ticks)
    cbar.set_label(f'{cbar_label} - Predicted', rotation=270, labelpad=15)
    axs[1].set_xlabel(f'{coordinate1} Coordinate')
    axs[1].set_ylabel(f'{coordinate2} Coordinate')
    axs[1].set_xlim(lim_min1, lim_max1) 
    axs[1].set_ylim(lim_min2, lim_max2)
    axs[1].set_title(f'Predicted {cbar_label} in the {plane} Plane for Wind Angle = {wind_angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f} \n Mean = {np.mean(z_pred):.2f} and Standard Deviation = {np.std(z_pred):.2f}')

    plt.tight_layout()
    plt.savefig(savename)
    plt.close()

def make_double_scatter_plots(figsize, z_actual, z_pred, grid1, grid2, grid3_actual, grid3_pred, geometry, plane, wind_angle, cut, tolerance, cbar_label, savename, arrow10 = None, arrow20 = None, arrow11 = None, arrow21 = None):
    coordinate1, coordinate2, coordinate3, lim_min1, lim_max1, lim_min2, lim_max2 = get_plane_config(plane)
    fig, axs = plt.subplots(1, 2, figsize=figsize, sharey=True)
    fig.suptitle(f'Comparison of Actual vs. Predicted values with Wind Angle = {wind_angle} in the {plane} Plane with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')
    vmin, vmax, levels, cmap, scatter_size, ticks = plotting_details(z_actual)
    
    scatter_actual = axs[0].scatter(grid1, grid2, c=grid3_actual, s=scatter_size, vmin=np.min(z_actual), vmax=np.max(z_actual), cmap=cmap)
    scatter_cbar = fig.colorbar(scatter_actual, ax=axs[0])
    scatter_cbar.set_label(f'Scatter {cbar_label}', rotation=270, labelpad=15)
    plot_geometry(plane, geometry, scatter_size, axs[0])
    if arrow10 is not None and arrow20 is not None:
        axs[0].add_patch(arrow10)
        axs[0].add_patch(arrow20)
    axs[0].set_xlabel(f'{coordinate1} Coordinate')
    axs[0].set_ylabel(f'{coordinate2} Coordinate')
    axs[0].set_xlim(lim_min1, lim_max1) 
    axs[0].set_ylim(lim_min2, lim_max2)
    axs[0].set_title(f'Actual {cbar_label} in the {plane} Plane for Wind Angle = {wind_angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f} \n Mean = {np.mean(z_actual):.2f} and Standard Deviation = {np.std(z_actual):.2f}')
    
    scatter_pred = axs[1].scatter(grid1, grid2, c=grid3_pred, s=scatter_size, vmin=np.min(z_actual), vmax=np.max(z_actual), cmap=cmap)
    scatter_cbar = fig.colorbar(scatter_pred, ax=axs[1])
    scatter_cbar.set_label(f'Scatter {cbar_label}', rotation=270, labelpad=15)
    plot_geometry(plane, geometry, scatter_size, axs[1])
    if arrow11 is not None and arrow21 is not None:
        axs[1].add_patch(arrow11)
        axs[1].add_patch(arrow21)
    axs[1].set_xlabel(f'{coordinate1} Coordinate')
    axs[1].set_ylabel(f'{coordinate2} Coordinate')
    axs[1].set_xlim(lim_min1, lim_max1) 
    axs[1].set_ylim(lim_min2, lim_max2)
    axs[1].set_title(f'Predicted {cbar_label} in the {plane} Plane for Wind Angle = {wind_angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f} \n Mean = {np.mean(z_pred):.2f} and Standard Deviation = {np.std(z_pred):.2f}')

    plt.tight_layout()
    plt.savefig(savename)
    plt.close()

def plot_differences(config, df, plane, wind_angle, cut, tolerance, geometry_filename, savenames):
    coordinate1, coordinate2, coordinate3, lim_min1, lim_max1, lim_min2, lim_max2 = get_plane_config(plane)
    filtered_df = filter_dataframe(config, df, wind_angle, coordinate3, cut, tolerance)
    grid1, grid2 = define_scatter_grid(filtered_df, coordinate1, coordinate2)
    filtered_df['Velocity_X_Difference'] = filtered_df['Velocity_X_Actual'] - filtered_df['Velocity_X_Predicted']
    filtered_df['Velocity_Y_Difference'] = filtered_df['Velocity_Y_Actual'] - filtered_df['Velocity_Y_Predicted']
    filtered_df['Velocity_Z_Difference'] = filtered_df['Velocity_Z_Actual'] - filtered_df['Velocity_Z_Predicted']
    filtered_df['Velocity_Magnitude_Difference'] = filtered_df['Velocity_Magnitude_Actual'] - filtered_df['Velocity_Magnitude_Predicted']
    if 'Pressure_Actual' in filtered_df.columns and 'TurbVisc_Actual' in filtered_df.columns:
        filtered_df['Pressure_Difference'] = filtered_df['Pressure_Actual'] - filtered_df['Pressure_Predicted']
        filtered_df['TurbVisc_Difference'] = filtered_df['TurbVisc_Actual'] - filtered_df['TurbVisc_Predicted']
        all_grids = [filtered_df['Velocity_Magnitude_Difference'], filtered_df['Velocity_X_Difference'], filtered_df['Velocity_Y_Difference'], filtered_df['Velocity_Z_Difference'], filtered_df['Pressure_Difference'], filtered_df['TurbVisc_Difference']]
    else:
        all_grids = [filtered_df['Velocity_Magnitude_Difference'], filtered_df['Velocity_X_Difference'], filtered_df['Velocity_Y_Difference'], filtered_df['Velocity_Z_Difference']] 
    geometry = get_geometry(plane, geometry_filename)
    grid1, grid2 = normalize_grids(plane, grid1, grid2)
    cut, tolerance = normalize_cut_tolerance(plane, cut, tolerance)
    counter = 0
    for grid3 in all_grids:
        make_scatter_plot((32, 16), filtered_df[savenames[counter][0]].values, grid1, grid2, grid3, geometry, plane, wind_angle, cut, tolerance, savenames[counter][1], savenames[counter][2])
        counter += 1

def plot_data(config, df, plane, wind_angle, cut, tolerance, geometry_filename, savenames):
    coordinate1, coordinate2, coordinate3, lim_min1, lim_max1, lim_min2, lim_max2 = get_plane_config(plane)
    filtered_df = filter_dataframe(config, df, wind_angle, coordinate3, cut, tolerance)
    grid1, grid2 = define_grid(filtered_df, coordinate1, coordinate2)
    if 'Pressure' in filtered_df.columns and 'TurbVisc' in filtered_df.columns:
        grid_vx, grid_vy, grid_vz, grid_magnitude, grid_pressure, grid_turbvisc = get_all_grids(grid1, grid2, coordinate1, coordinate2, filtered_df)
        all_grids = [grid_magnitude, grid_vx, grid_vy, grid_vz, grid_pressure, grid_turbvisc]
    else:
        grid_vx, grid_vy, grid_vz, grid_magnitude = get_velocity_grids(grid1, grid2, coordinate1, coordinate2, filtered_df)
        all_grids = [grid_magnitude, grid_vx, grid_vy, grid_vz]
    geometry = get_geometry(plane, geometry_filename)
    grid1, grid2 = normalize_grids(plane, grid1, grid2)
    quiver_points1, quiver_points2, quiver_v1, quiver_v2 = make_quiver_grid(plane, grid1, grid2, grid_vx, grid_vy, grid_vz, stride=100)
    cut, tolerance = normalize_cut_tolerance(plane, cut, tolerance)
    counter = 0
    for grid3 in all_grids:
        make_singular_plot((32, 16), filtered_df[savenames[counter][0]].values, grid1, grid2, grid3, geometry, quiver_points1, quiver_points2, quiver_v1, quiver_v2, plane, wind_angle, cut, tolerance, savenames[counter][1], savenames[counter][2])
        if counter == 0 and plane == 'X-Y' and config["plotting"]["arrow"][0]:
            arrow1 = arrow(plane,config["plotting"]["arrow"][1][0][0],config["plotting"]["arrow"][1][0][1],wind_angle)
            arrow2 = arrow(plane,config["plotting"]["arrow"][1][1][0],config["plotting"]["arrow"][1][1][1],wind_angle)
            make_singular_plot((32, 16), filtered_df[savenames[counter][0]].values, grid1, grid2, grid3, geometry, quiver_points1, quiver_points2, quiver_v1, quiver_v2, plane, wind_angle, cut, tolerance, savenames[counter][1], f'{savenames[counter][2]}_arrow.png', arrow1, arrow2)
        else:
            pass
        counter += 1

def plot_data_scatter(config, df, plane, wind_angle, cut, tolerance, geometry_filename, savenames):
    coordinate1, coordinate2, coordinate3, lim_min1, lim_max1, lim_min2, lim_max2 = get_plane_config(plane)
    filtered_df = filter_dataframe(config, df, wind_angle, coordinate3, cut, tolerance)
    grid1, grid2 = define_scatter_grid(filtered_df, coordinate1, coordinate2)
    if 'Pressure' in filtered_df.columns and 'TurbVisc' in filtered_df.columns:
        all_grids = [filtered_df['Velocity_Magnitude'],filtered_df['Velocity_X'],filtered_df['Velocity_Y'],filtered_df['Velocity_Z'],filtered_df['Pressure'],filtered_df['TurbVisc']]
    else:
        all_grids = [filtered_df['Velocity_Magnitude'],filtered_df['Velocity_X'],filtered_df['Velocity_Y'],filtered_df['Velocity_Z']]
    geometry = get_geometry(plane, geometry_filename)
    grid1, grid2 = normalize_grids(plane, grid1, grid2)
    cut, tolerance = normalize_cut_tolerance(plane, cut, tolerance)
    counter = 0
    for grid3 in all_grids:
        while counter < len(savenames):
            make_scatter_plot((32, 16), filtered_df[savenames[counter][0]].values, grid1, grid2, grid3, geometry, plane, wind_angle, cut, tolerance, savenames[counter][1], savenames[counter][2])
            counter += 1

def plot_data_predictions(config, df, plane, wind_angle, cut, tolerance, geometry_filename, savenames):
    coordinate1, coordinate2, coordinate3, lim_min1, lim_max1, lim_min2, lim_max2 = get_plane_config(plane)
    filtered_df = filter_dataframe(config, df, wind_angle, coordinate3, cut, tolerance)
    grid1, grid2 = define_grid(filtered_df, coordinate1, coordinate2)
    if 'Pressure_Actual' in filtered_df.columns and 'TurbVisc_Actual' in filtered_df.columns:
        grid_vx_actual, grid_vy_actual, grid_vz_actual, grid_magnitude_actual, grid_pressure_actual, grid_turbvisc_actual = get_all_grids(grid1, grid2, coordinate1, coordinate2, filtered_df, 'Actual')
        grid_vx_pred, grid_vy_pred, grid_vz_pred, grid_magnitude_pred, grid_pressure_pred, grid_turbvisc_pred = get_all_grids(grid1, grid2, coordinate1, coordinate2, filtered_df, 'Predicted')
        all_grids = [[grid_magnitude_actual,grid_magnitude_pred], [grid_vx_actual,grid_vx_pred], [grid_vy_actual,grid_vy_pred], [grid_vz_actual,grid_vz_pred], [grid_pressure_actual,grid_pressure_pred], [grid_turbvisc_actual,grid_turbvisc_pred]]
    else:
        grid_vx_actual, grid_vy_actual, grid_vz_actual, grid_magnitude_actual = get_velocity_grids(grid1, grid2, coordinate1, coordinate2, filtered_df, 'Actual')
        grid_vx_pred, grid_vy_pred, grid_vz_pred, grid_magnitude_pred = get_velocity_grids(grid1, grid2, coordinate1, coordinate2, filtered_df, 'Predicted')
        all_grids = [[grid_magnitude_actual,grid_magnitude_pred], [grid_vx_actual,grid_vx_pred], [grid_vy_actual,grid_vy_pred], [grid_vz_actual,grid_vz_pred]]
    geometry = get_geometry(plane, geometry_filename)
    grid1, grid2 = normalize_grids(plane, grid1, grid2)
    quiver_points1, quiver_points2, quiver_v1_actual, quiver_v2_actual = make_quiver_grid(plane, grid1, grid2, grid_vx_actual, grid_vy_actual, grid_vz_actual, stride=100)
    quiver_points1, quiver_points2, quiver_v1_pred, quiver_v2_pred = make_quiver_grid(plane, grid1, grid2, grid_vx_pred, grid_vy_pred, grid_vz_pred, stride=100)
    cut, tolerance = normalize_cut_tolerance(plane, cut, tolerance)
    counter = 0
    for grid3_actual, grid3_pred in all_grids:
        make_double_plots((32, 16), filtered_df[savenames[counter][0]].values, filtered_df[savenames[counter][0].replace("_Actual", "_Predicted")].values, grid1, grid2, grid3_actual, grid3_pred, geometry, quiver_points1, quiver_points2, quiver_v1_actual, quiver_v2_actual, quiver_v1_pred, quiver_v2_pred, plane, wind_angle, cut, tolerance, savenames[counter][1], savenames[counter][2], arrow10 = None, arrow20 = None, arrow11 = None, arrow21 = None)
        if counter == 0 and plane == 'X-Y' and config["plotting"]["arrow"][0]:
            arrow10 = arrow(plane,config["plotting"]["arrow"][1][0][0],config["plotting"]["arrow"][1][0][1],wind_angle)
            arrow20 = arrow(plane,config["plotting"]["arrow"][1][1][0],config["plotting"]["arrow"][1][1][1],wind_angle)
            arrow11 = arrow(plane,config["plotting"]["arrow"][1][0][0],config["plotting"]["arrow"][1][0][1],wind_angle)
            arrow21 = arrow(plane,config["plotting"]["arrow"][1][1][0],config["plotting"]["arrow"][1][1][1],wind_angle)
            make_double_plots((32, 16), filtered_df[savenames[counter][0]].values, filtered_df[savenames[counter][0].replace("_Actual", "_Predicted")].values, grid1, grid2, grid3_actual, grid3_pred, geometry, quiver_points1, quiver_points2, quiver_v1_actual, quiver_v2_actual, quiver_v1_pred, quiver_v2_pred, plane, wind_angle, cut, tolerance, savenames[counter][1], f'{savenames[counter][2]}_arrow.png', arrow10, arrow20, arrow11, arrow21)
        else:
            pass
        counter += 1

def plot_data_scatter_predictions(config, df, plane, wind_angle, cut, tolerance, geometry_filename, savenames):
    coordinate1, coordinate2, coordinate3, lim_min1, lim_max1, lim_min2, lim_max2 = get_plane_config(plane)
    filtered_df = filter_dataframe(config, df, wind_angle, coordinate3, cut, tolerance)
    grid1, grid2 = define_scatter_grid(filtered_df, coordinate1, coordinate2)
    if 'Pressure_Actual' in filtered_df.columns and 'TurbVisc_Actual' in filtered_df.columns:
        all_grids = [[filtered_df['Velocity_Magnitude_Actual'], filtered_df['Velocity_Magnitude_Predicted']], [filtered_df['Velocity_X_Actual'], filtered_df['Velocity_X_Predicted']], [filtered_df['Velocity_Y_Actual'], filtered_df['Velocity_Y_Predicted']], [filtered_df['Velocity_Z_Actual'], filtered_df['Velocity_Z_Predicted']], [filtered_df['Pressure_Actual'], filtered_df['Pressure_Predicted']], [filtered_df['TurbVisc_Actual'], filtered_df['TurbVisc_Predicted']]]
    else:
        all_grids = [[filtered_df['Velocity_Magnitude_Actual'], filtered_df['Velocity_Magnitude_Predicted']], [filtered_df['Velocity_X_Actual'], filtered_df['Velocity_X_Predicted']], [filtered_df['Velocity_Y_Actual'], filtered_df['Velocity_Y_Predicted']], [filtered_df['Velocity_Z_Actual'], filtered_df['Velocity_Z_Predicted']]]
    geometry = get_geometry(plane, geometry_filename)
    grid1, grid2 = normalize_grids(plane, grid1, grid2)
    cut, tolerance = normalize_cut_tolerance(plane, cut, tolerance)
    counter = 0
    for grid3_actual, grid3_pred in all_grids:
        make_double_scatter_plots((32, 16), filtered_df[savenames[counter][0]].values, filtered_df[savenames[counter][0].replace("_Actual", "_Predicted")].values, grid1, grid2, grid3_actual, grid3_pred, geometry, plane, wind_angle, cut, tolerance, savenames[counter][1], savenames[counter][2], arrow10 = None, arrow20 = None, arrow11 = None, arrow21 = None)
        if counter == 0 and plane == 'X-Y' and config["plotting"]["arrow"][0]:
            arrow10 = arrow(plane,config["plotting"]["arrow"][1][0][0],config["plotting"]["arrow"][1][0][1],wind_angle)
            arrow20 = arrow(plane,config["plotting"]["arrow"][1][1][0],config["plotting"]["arrow"][1][1][1],wind_angle)
            arrow11 = arrow(plane,config["plotting"]["arrow"][1][0][0],config["plotting"]["arrow"][1][0][1],wind_angle)
            arrow21 = arrow(plane,config["plotting"]["arrow"][1][1][0],config["plotting"]["arrow"][1][1][1],wind_angle)
            make_double_scatter_plots((32, 16), filtered_df[savenames[counter][0]].values, filtered_df[savenames[counter][0].replace("_Actual", "_Predicted")].values, grid1, grid2, grid3_actual, grid3_pred, geometry, plane, wind_angle, cut, tolerance, savenames[counter][1], f'{savenames[counter][2]}_arrow.png', arrow10, arrow20, arrow11, arrow21)
        else:
            pass
        counter += 1

def plot_data_2d(config,df,wind_angles,geometry_filename,plot_folder,single):
    params = config["plotting"]["plotting_params"]
    os.makedirs(plot_folder, exist_ok=True)
    for wind_angle in wind_angles:
        for j in params:
            plane = j[0]
            cut = j[1]
            tolerance = j[2]
            savenames_single = [['Velocity_Magnitude', 'Velocity Magnitude', os.path.join(plot_folder,f'{plane}_totalvelocity_{wind_angle}.png')],
            ['Velocity_X', 'Velocity X', os.path.join(plot_folder,f'{plane}_vx_{wind_angle}.png')],
            ['Velocity_Y', 'Velocity Y', os.path.join(plot_folder,f'{plane}_vy_{wind_angle}.png')],
            ['Velocity_Z', 'Velocity Z', os.path.join(plot_folder,f'{plane}_vz_{wind_angle}.png')],
            ['Pressure', 'Pressure', os.path.join(plot_folder,f'{plane}_pressure_{wind_angle}.png')],
            ['TurbVisc', 'TurbVisc', os.path.join(plot_folder,f'{plane}_turbvisc_{wind_angle}.png')]]
            savenames_scatter_single = [['Velocity_Magnitude', 'Velocity Magnitude', os.path.join(plot_folder,f'{plane}_totalvelocity_scatter_{wind_angle}.png')],
            ['Velocity_X', 'Velocity X', os.path.join(plot_folder,f'{plane}_vx_scatter_{wind_angle}.png')],
            ['Velocity_Y', 'Velocity Y', os.path.join(plot_folder,f'{plane}_vy_scatter_{wind_angle}.png')],
            ['Velocity_Z', 'Velocity Z', os.path.join(plot_folder,f'{plane}_vz_scatter_{wind_angle}.png')],
            ['Pressure', 'Pressure', os.path.join(plot_folder,f'{plane}_pressure_scatter_{wind_angle}.png')],
            ['TurbVisc', 'TurbVisc', os.path.join(plot_folder,f'{plane}_turbvisc_scatter_{wind_angle}.png')]]
            savenames_double = [['Velocity_Magnitude_Actual', 'Velocity Magnitude', os.path.join(plot_folder,f'{plane}_totalvelocity_{wind_angle}.png')],
            ['Velocity_X_Actual', 'Velocity X', os.path.join(plot_folder,f'{plane}_vx_{wind_angle}.png')],
            ['Velocity_Y_Actual', 'Velocity Y', os.path.join(plot_folder,f'{plane}_vy_{wind_angle}.png')],
            ['Velocity_Z_Actual', 'Velocity Z', os.path.join(plot_folder,f'{plane}_vz_{wind_angle}.png')],
            ['Pressure_Actual', 'Pressure', os.path.join(plot_folder,f'{plane}_pressure_{wind_angle}.png')],
            ['TurbVisc_Actual', 'TurbVisc', os.path.join(plot_folder,f'{plane}_turbvisc_{wind_angle}.png')]]
            savenames_scatter_double = [['Velocity_Magnitude_Actual', 'Velocity Magnitude', os.path.join(plot_folder,f'{plane}_totalvelocity_scatter_{wind_angle}.png')],
            ['Velocity_X_Actual', 'Velocity X', os.path.join(plot_folder,f'{plane}_vx_scatter_{wind_angle}.png')],
            ['Velocity_Y_Actual', 'Velocity Y', os.path.join(plot_folder,f'{plane}_vy_scatter_{wind_angle}.png')],
            ['Velocity_Z_Actual', 'Velocity Z', os.path.join(plot_folder,f'{plane}_vz_scatter_{wind_angle}.png')],
            ['Pressure_Actual', 'Pressure', os.path.join(plot_folder,f'{plane}_pressure_scatter_{wind_angle}.png')],
            ['TurbVisc_Actual', 'TurbVisc', os.path.join(plot_folder,f'{plane}_turbvisc_scatter_{wind_angle}.png')]]
            if single:
                plot_data(config, df, plane, wind_angle, cut, tolerance, geometry_filename, savenames_single)
                plot_data_scatter(config, df, plane, wind_angle, cut, tolerance, geometry_filename, savenames_scatter_single)
            else:
                plot_data_predictions(config, df, plane, wind_angle, cut, tolerance, geometry_filename, savenames_double)
                plot_data_scatter_predictions(config, df, plane, wind_angle, cut, tolerance, geometry_filename, savenames_scatter_double)

def plot_diff_2d(config,df,wind_angles,geometry_filename,plot_folder):
    params = config["plotting"]["plotting_params"]
    os.makedirs(plot_folder, exist_ok=True)
    for wind_angle in wind_angles:
        for j in params:
            plane = j[0]
            cut = j[1]
            tolerance = j[2]
            savenames_scatter_single = [['Velocity_Magnitude_Difference', 'Velocity Magnitude_Difference', os.path.join(plot_folder,f'{plane}_totalvelocity_scatter_diff_{wind_angle}.png')],
            ['Velocity_X_Difference', 'Velocity X_Difference', os.path.join(plot_folder,f'{plane}_vx_scatter_diff_{wind_angle}.png')],
            ['Velocity_Y_Difference', 'Velocity Y_Difference', os.path.join(plot_folder,f'{plane}_vy_scatter_diff_{wind_angle}.png')],
            ['Velocity_Z_Difference', 'Velocity Z_Difference', os.path.join(plot_folder,f'{plane}_vz_scatter_diff_{wind_angle}.png')],
            ['Pressure_Difference', 'Pressure_Difference', os.path.join(plot_folder,f'{plane}_pressure_scatter_diff_{wind_angle}.png')],
            ['TurbVisc_Difference', 'TurbVisc_Difference', os.path.join(plot_folder,f'{plane}_turbvisc_scatter_diff_{wind_angle}.png')]]
            plot_differences(config, df, plane, wind_angle, cut, tolerance, geometry_filename, savenames_scatter_single)

def make_logging_plots(directory, df, current_time, num_epochs, epoch_time, save_plot, save_plot_xlog, save_plot_ylog, save_plot_ylog_xlog, xlog, ylog):
    epoch = df['Epoch']
    losses = df.iloc[:, 3:]
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    plt.figure(figsize=(32, 16))
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

def make_test_plots(directory, filename, current_time, num_epochs, epoch_time, save_plot, save_plot_xlog, save_plot_ylog, save_plot_ylog_xlog, xlog, ylog):
    df = filter_trainingloss_file(directory, filename)
    epoch = df['Epoch']
    df_mse = df.filter(regex='^MSE_')
    df_r2 = df.filter(regex='^R2_')
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    fig, axs = plt.subplots(1, 2, figsize=(32, 16), sharey=False)
    fig.suptitle(f'Time Taken = {current_time:.2f} hours for {num_epochs} Epochs; Time Taken Per Epoch = {epoch_time:.2f} hours')
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

    if xlog and not ylog:
        axs[0].set_xlabel('Epoch (log)')
        axs[0].set_ylabel('MSE Value')
        axs[1].set_xlabel('Epoch (log)')
        axs[1].set_ylabel('R2 Value')
        axs[0].set_xscale('log')
        axs[1].set_xscale('log')
        plt.savefig(os.path.join(directory, save_plot_xlog))
        plt.close()
    elif ylog and not xlog:
        if (df_r2 > 0).all().all():
            axs[0].set_xlabel('Epoch')
            axs[0].set_ylabel('MSE Value (log)')
            axs[1].set_xlabel('Epoch')
            axs[1].set_ylabel('R2 Value (log)')
            axs[0].set_yscale('log')
            axs[1].set_yscale('log')
            plt.savefig(os.path.join(directory, save_plot_ylog))
            plt.close()
    elif xlog and ylog:
        if (df_r2 > 0).all().all():
            axs[0].set_xlabel('Epoch (log)')
            axs[0].set_ylabel('MSE Value (log)')
            axs[1].set_xlabel('Epoch (log)')
            axs[1].set_ylabel('R2 Value (log)')
            axs[0].set_xscale('log')
            axs[1].set_xscale('log')
            axs[0].set_yscale('log')
            axs[1].set_yscale('log')
            plt.savefig(os.path.join(directory, save_plot_ylog_xlog))
            plt.close()
    elif not xlog and not ylog:
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('MSE Value')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('R2 Value')
        plt.savefig(os.path.join(directory, save_plot))
        plt.close()

def make_all_logging_plots(directory, df, current_time, total_epochs):
    save_plot = 'save_logging_plot.png'
    save_plot_xlog = 'save_logging_plot_xlog.png'
    save_plot_ylog = 'save_logging_plot_ylog.png'
    save_plot_ylog_xlog = 'save_logging_plot_ylog_xlog.png'
    info_test = 'info_test.csv'
    info_skipped = 'info_skipped.csv'
    save_testing_loss = 'save_testing_loss.png'
    save_testing_loss_xlog = 'save_testing_loss_xlog.png'
    save_testing_loss_ylog = 'save_testing_loss_ylog.png'
    save_testing_loss_ylog_xlog = 'save_testing_loss_ylog_xlog.png'
    save_skipped_loss = 'save_skipped_loss.png'
    save_skipped_loss_xlog = 'save_skipped_loss_xlog.png'
    save_skipped_loss_ylog = 'save_skipped_loss_ylog.png'
    save_skipped_loss_ylog_xlog = 'save_skipped_loss_ylog_xlog.png'
    xlog_ylog = [[True, True], [True, False], [False, True], [False, False]]
    epoch_time = current_time/total_epochs
    for xlog, ylog in xlog_ylog:
        make_logging_plots(directory, df, current_time, total_epochs, epoch_time, save_plot, save_plot_xlog, save_plot_ylog, save_plot_ylog_xlog, xlog, ylog)
        make_test_plots(directory, info_test, current_time, total_epochs, epoch_time, save_testing_loss, save_testing_loss_xlog, save_testing_loss_ylog, save_testing_loss_ylog_xlog, xlog, ylog)
        make_test_plots(directory, info_skipped, current_time, total_epochs, epoch_time, save_skipped_loss, save_skipped_loss_xlog, save_skipped_loss_ylog, save_skipped_loss_ylog_xlog, xlog, ylog)