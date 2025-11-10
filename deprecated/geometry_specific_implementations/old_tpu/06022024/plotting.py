from definitions import *
from plotting_definitions import *

def make_singular_plot(figsize, z, grid1, grid2, grid3, geometry1, geometry2, quiver_points1, quiver_points2, quiver_v1, quiver_v2, plane, wind_angle, cut, tolerance, cbar_label, savename, arrow1= None, arrow2 = None):
    coordinate1, coordinate2, coordinate3, lim_min1, lim_max1, lim_min2, lim_max2 = get_plane_config(plane)
    fig, ax = plt.subplots(figsize=figsize, sharey=True)
    vmin, vmax, levels, cmap, scatter_size, ticks = plotting_details(z)
    contour = ax.contourf(grid1, grid2, grid3, levels=levels, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.quiver(quiver_points1, quiver_points2, quiver_v1, quiver_v2, scale=200, color='white')
    ax.scatter(geometry1, geometry2, c='black', s=scatter_size, label='Geometry')
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

def make_double_plots(figsize, z, grid1, grid2, grid3_actual, grid3_pred, geometry1, geometry2, quiver_points1, quiver_points2, quiver_v1_actual, quiver_v2_actual, quiver_v1_pred, quiver_v2_pred, plane, wind_angle, cut, tolerance, cbar_label, savename, arrow10 = None, arrow20 = None, arrow11 = None, arrow21 = None):
    coordinate1, coordinate2, coordinate3, lim_min1, lim_max1, lim_min2, lim_max2 = get_plane_config(plane)
    fig, axs = plt.subplots(1, 2, figsize=figsize, sharey=True)
    fig.suptitle(f'Comparison of Actual vs. Predicted values with Wind Angle = {wind_angle} in the {plane} Plane with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')
    vmin, vmax, levels, cmap, scatter_size, ticks = plotting_details(z)
    
    contour_actual = axs[0].contourf(grid1, grid2, grid3_actual, levels=levels, vmin=vmin, vmax=vmax, cmap=cmap)
    axs[0].quiver(quiver_points1, quiver_points2, quiver_v1_actual, quiver_v2_actual, scale=200, color='white')
    axs[0].scatter(geometry1, geometry2, c='black', s=scatter_size, label='Geometry')
    if arrow10 is not None and arrow20 is not None:
        axs[0].add_patch(arrow10)
        axs[0].add_patch(arrow20)
    cbar = fig.colorbar(contour_actual, ax=axs[0], ticks=ticks)
    cbar.set_label(f'{cbar_label} - Actual', rotation=270, labelpad=15)
    axs[0].set_xlabel(f'{coordinate1} Coordinate')
    axs[0].set_ylabel(f'{coordinate2} Coordinate')
    axs[0].set_xlim(lim_min1, lim_max1) 
    axs[0].set_ylim(lim_min2, lim_max2)
    axs[0].set_title(f'Actual {cbar_label} in the {plane} Plane for Wind Angle = {wind_angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')
    
    contour_pred = axs[1].contourf(grid1, grid2, grid3_pred, levels=levels, vmin=vmin, vmax=vmax, cmap=cmap)
    axs[1].quiver(quiver_points1, quiver_points2, quiver_v1_pred, quiver_v2_pred, scale=200, color='white')
    axs[1].scatter(geometry1, geometry2, c='black', s=scatter_size, label='Geometry')
    if arrow11 is not None and arrow21 is not None:
        axs[1].add_patch(arrow11)
        axs[1].add_patch(arrow21)
    cbar = fig.colorbar(contour_pred, ax=axs[1], ticks=ticks)
    cbar.set_label(f'{cbar_label} - Predicted', rotation=270, labelpad=15)
    axs[1].set_xlabel(f'{coordinate1} Coordinate')
    axs[1].set_ylabel(f'{coordinate2} Coordinate')
    axs[1].set_xlim(lim_min1, lim_max1) 
    axs[1].set_ylim(lim_min2, lim_max2)
    axs[1].set_title(f'Predicted {cbar_label} in the {plane} Plane for Wind Angle = {wind_angle} with a cut at {coordinate3} = {cut:.2f} +/- {tolerance:.2f}')

    plt.tight_layout()
    plt.savefig(savename)
    plt.close()

def plot_data(df, plane, wind_angle, cut, tolerance, geometry_filename, savenames):
    coordinate1, coordinate2, coordinate3, lim_min1, lim_max1, lim_min2, lim_max2 = get_plane_config(plane)
    filtered_df = filter_dataframe(df, wind_angle, coordinate3, cut, tolerance)
    grid1, grid2 = define_grid(filtered_df, coordinate1, coordinate2)
    if 'Pressure' in filtered_df.columns and 'TurbVisc' in filtered_df.columns:
        grid_vx, grid_vy, grid_vz, grid_magnitude, grid_pressure, grid_turbvisc = get_all_grids(grid1, grid2, coordinate1, coordinate2, filtered_df)
        all_grids = [grid_magnitude, grid_vx, grid_vy, grid_vz, grid_pressure, grid_turbvisc]
    else:
        grid_vx, grid_vy, grid_vz, grid_magnitude = get_velocity_grids(grid1, grid2, coordinate1, coordinate2, filtered_df)
        all_grids = [grid_magnitude, grid_vx, grid_vy, grid_vz]
    geometry1, geometry2 = get_geometry(plane, geometry_filename)
    grid1, grid2 = normalize_grids(plane, grid1, grid2)
    geometry1, geometry2 = normalize_grids(plane, geometry1, geometry2)
    quiver_points1, quiver_points2, quiver_v1, quiver_v2 = make_quiver_grid(plane, grid1, grid2, grid_vx, grid_vy, grid_vz, stride=100)
    cut, tolerance = normalize_cut_tolerance(plane, cut, tolerance)
    counter = 0
    for grid3 in all_grids:
        make_singular_plot((16, 8), filtered_df[savenames[counter][0]].values, grid1, grid2, grid3, geometry1, geometry2, quiver_points1, quiver_points2, quiver_v1, quiver_v2, plane, wind_angle, cut, tolerance, savenames[counter][1], savenames[counter][2])
        if counter == 0 and plane == 'X-Y':
            arrow1 = make_arrow(plane,500,500,wind_angle)
            arrow2 = make_arrow(plane,500,570,wind_angle)
            make_singular_plot((16, 8), filtered_df[savenames[counter][0]].values, grid1, grid2, grid3, geometry1, geometry2, quiver_points1, quiver_points2, quiver_v1, quiver_v2, plane, wind_angle, cut, tolerance, savenames[counter][1], f'{savenames[counter][2]}_arrow.png', arrow1, arrow2)
        else:
            pass
        counter += 1

def plot_data_predictions(df, plane, wind_angle, cut, tolerance, geometry_filename, savenames):
    coordinate1, coordinate2, coordinate3, lim_min1, lim_max1, lim_min2, lim_max2 = get_plane_config(plane)
    filtered_df = filter_dataframe(df, wind_angle, coordinate3, cut, tolerance)
    grid1, grid2 = define_grid(filtered_df, coordinate1, coordinate2)
    if 'Pressure_Actual' in filtered_df.columns and 'TurbVisc_Actual' in filtered_df.columns:
        grid_vx_actual, grid_vy_actual, grid_vz_actual, grid_magnitude_actual, grid_pressure_actual, grid_turbvisc_actual = get_all_grids(grid1, grid2, coordinate1, coordinate2, filtered_df, 'Actual')
        grid_vx_pred, grid_vy_pred, grid_vz_pred, grid_magnitude_pred, grid_pressure_pred, grid_turbvisc_pred = get_all_grids(grid1, grid2, coordinate1, coordinate2, filtered_df, 'Predicted')
        all_grids = [[grid_magnitude_actual,grid_magnitude_pred], [grid_vx_actual,grid_vx_pred], [grid_vy_actual,grid_vy_pred], [grid_vz_actual,grid_vz_pred], [grid_pressure_actual,grid_pressure_pred], [grid_turbvisc_actual,grid_turbvisc_pred]]
    else:
        grid_vx_actual, grid_vy_actual, grid_vz_actual, grid_magnitude_actual = get_velocity_grids(grid1, grid2, coordinate1, coordinate2, filtered_df, 'Actual')
        grid_vx_pred, grid_vy_pred, grid_vz_pred, grid_magnitude_pred = get_velocity_grids(grid1, grid2, coordinate1, coordinate2, filtered_df, 'Predicted')
        all_grids = [[grid_magnitude_actual,grid_magnitude_pred], [grid_vx_actual,grid_vx_pred], [grid_vy_actual,grid_vy_pred], [grid_vz_actual,grid_vz_pred]]
    geometry1, geometry2 = get_geometry(plane, geometry_filename)
    grid1, grid2 = normalize_grids(plane, grid1, grid2)
    geometry1, geometry2 = normalize_grids(plane, geometry1, geometry2)
    quiver_points1, quiver_points2, quiver_v1_actual, quiver_v2_actual = make_quiver_grid(plane, grid1, grid2, grid_vx_actual, grid_vy_actual, grid_vz_actual, stride=100)
    quiver_points1, quiver_points2, quiver_v1_pred, quiver_v2_pred = make_quiver_grid(plane, grid1, grid2, grid_vx_pred, grid_vy_pred, grid_vz_pred, stride=100)
    cut, tolerance = normalize_cut_tolerance(plane, cut, tolerance)
    counter = 0
    for grid3_actual, grid3_pred in all_grids:
        make_double_plots((16, 8), filtered_df[savenames[counter][0]].values, grid1, grid2, grid3_actual, grid3_pred, geometry1, geometry2, quiver_points1, quiver_points2, quiver_v1_actual, quiver_v2_actual, quiver_v1_pred, quiver_v2_pred, plane, wind_angle, cut, tolerance, savenames[counter][1], savenames[counter][2], arrow10 = None, arrow20 = None, arrow11 = None, arrow21 = None)
        if counter == 0 and plane == 'X-Y':
            arrow10 = make_arrow(plane,500,500,wind_angle)
            arrow20 = make_arrow(plane,500,570,wind_angle)
            arrow11 = make_arrow(plane,500,500,wind_angle)
            arrow21 = make_arrow(plane,500,570,wind_angle)
            make_double_plots((16, 8), filtered_df[savenames[counter][0]].values, grid1, grid2, grid3_actual, grid3_pred, geometry1, geometry2, quiver_points1, quiver_points2, quiver_v1_actual, quiver_v2_actual, quiver_v1_pred, quiver_v2_pred, plane, wind_angle, cut, tolerance, savenames[counter][1], f'{savenames[counter][2]}_arrow.png', arrow10, arrow20, arrow11, arrow21)
        else:
            pass
        counter += 1

def plot_data_2d(df,wind_angles,geometry_filename,plot_folder,single):
    params = [['X-Z',570,5],['Y-Z',500,5],['X-Y',50,5]]
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
            savenames_double = [['Velocity_Magnitude_Actual', 'Velocity Magnitude', os.path.join(plot_folder,f'{plane}_totalvelocity_{wind_angle}.png')],
            ['Velocity_X_Actual', 'Velocity X', os.path.join(plot_folder,f'{plane}_vx_{wind_angle}.png')],
            ['Velocity_Y_Actual', 'Velocity Y', os.path.join(plot_folder,f'{plane}_vy_{wind_angle}.png')],
            ['Velocity_Z_Actual', 'Velocity Z', os.path.join(plot_folder,f'{plane}_vz_{wind_angle}.png')],
            ['Pressure_Actual', 'Pressure', os.path.join(plot_folder,f'{plane}_pressure_{wind_angle}.png')],
            ['TurbVisc_Actual', 'TurbVisc', os.path.join(plot_folder,f'{plane}_turbvisc_{wind_angle}.png')]]
            if single:
                plot_data(df, plane, wind_angle, cut, tolerance, geometry_filename, savenames_single)
            else:
                plot_data_predictions(df, plane, wind_angle, cut, tolerance, geometry_filename, savenames_double)

def make_logging_plots(directory, df, current_time, num_epochs, epoch_time, save_plot, save_plot_xlog, save_plot_ylog, save_plot_ylog_xlog, xlog, ylog):
    epoch = df['Epoch']
    losses = df.loc[:, 'Data Loss':'Total Loss']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
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
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
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