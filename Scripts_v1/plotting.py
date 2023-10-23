from definitions import *
from training import *
from PINN import *

def plot_predictions(y_test, predictions, output_folder, variables):
    for i, var in enumerate(variables):
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(y_test)), y_test[:, i], label='Test', s=5)
        plt.scatter(range(len(predictions)), predictions[:, i], label='Prediction', s=5)
        plt.title(f'Test vs Prediction for {var}')
        plt.legend()
        
        safe_var_name = var.replace(':', '_')
        figure_file_path = os.path.join(output_folder, f'{safe_var_name}_test_vs_prediction.png')
        
        plt.savefig(figure_file_path)
        plt.close()

def plot_scatter_3d(features, actual, predicted, output_folder, variables):
    x = features['Points:0'].to_numpy()
    y = features['Points:1'].to_numpy()
    z = features['Points:2'].to_numpy()

    actual = actual
    predicted = predicted
    
    for i, var in enumerate(variables):
        # Create a subplot
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=(f'Actual {var}', f'Predicted {var}'),
                            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]])
        
        # Actual scatter
        scatter_actual = go.Scatter3d(x=x, y=y, z=z, mode='markers',
                                      marker=dict(size=2, color=actual[:, i], colorscale='Viridis', opacity=0.8, colorbar=dict(title=var, x=-0.07)),
                                      name='Actual')
        
        # Predicted scatter
        scatter_pred = go.Scatter3d(x=x, y=y, z=z, mode='markers',
                                    marker=dict(size=2, color=predicted[:, i], colorscale='Viridis', opacity=0.8, colorbar=dict(title=var, x=1.07)),
                                    name='Predicted')
        
        fig.add_trace(scatter_actual, row=1, col=1)
        fig.add_trace(scatter_pred, row=1, col=2)
        
        fig.update_layout(title=f"Comparison of Actual vs. Predicted {var} values")
        
        # To save the figure as an interactive HTML
        safe_var_name = var.replace(':', '_')
        fig.write_html(os.path.join(output_folder, f"{safe_var_name}_figure.html"))

def individual_plot_predictions(wind_angle, y_test, predictions, output_folder, variables):
    for i, var in enumerate(variables):
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(y_test)), y_test[:, i], label='Test', s=5)
        plt.scatter(range(len(predictions)), predictions[:, i], label='Prediction', s=5)
        plt.title(f'Test vs Prediction for {var} with Wind Angle = {wind_angle}')
        plt.legend()

        plot_folder = os.path.join(output_folder, f'plots_{wind_angle}')
        os.makedirs(plot_folder, exist_ok=True)
        
        safe_var_name = var.replace(':', '_')
        figure_file_path = os.path.join(plot_folder, f'{safe_var_name}_test_vs_prediction_for_wind_angle_{wind_angle}.png')
        
        plt.savefig(figure_file_path)
        plt.close()

def individual_plot_scatter_3d(wind_angle, positions, actual, predicted, output_folder, variables):

    x = positions['Points:0'].to_numpy()
    y = positions['Points:1'].to_numpy()
    z = positions['Points:2'].to_numpy()

    actual = actual
    predicted = predicted
    
    for i, var in enumerate(variables):
        # Create a subplot
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=(f'Actual {var}', f'Predicted {var}'),
                            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]])
        
        # Actual scatter
        scatter_actual = go.Scatter3d(x=x, y=y, z=z, mode='markers',
                                      marker=dict(size=2, color=actual[:, i], colorscale='Viridis', opacity=0.8, colorbar=dict(title=var, x=-0.07)),
                                      name='Actual')
        
        # Predicted scatter
        scatter_pred = go.Scatter3d(x=x, y=y, z=z, mode='markers',
                                    marker=dict(size=2, color=predicted[:, i], colorscale='Viridis', opacity=0.8, colorbar=dict(title=var, x=1.07)),
                                    name='Predicted')
        
        fig.add_trace(scatter_actual, row=1, col=1)
        fig.add_trace(scatter_pred, row=1, col=2)
        
        fig.update_layout(title=f"Comparison of Actual vs. Predicted {var} values with Wind Angle = {wind_angle}")

        plot_folder = os.path.join(output_folder, f'plots_{wind_angle}')
        os.makedirs(plot_folder, exist_ok=True)
        
        # To save the figure as an interactive HTML
        safe_var_name = var.replace(':', '_')
        fig.write_html(os.path.join(plot_folder, f"{safe_var_name}_figure_for_wind_angle_{wind_angle}.html"))

def plot_scatter_2d(features, actual, predicted, idx_test, output_folder, variables, variables_to_plot):  
    x_feature, y_feature = variables_to_plot[0]
    variable_to_plot = variables_to_plot[1]
    
    if variable_to_plot not in variables:
        raise ValueError(f"{variable_to_plot} not in provided variables")

    x_plot, y_plot = get_x_y_for_plot(x_feature, y_feature, features, idx_test)

    actual = actual
    predicted = predicted

    z_actual = actual[:, variables.index(variable_to_plot)]
    z_predicted = predicted[:, variables.index(variable_to_plot)]

    fig, axs = prepare_2d_subplots(x_feature, y_feature, z_actual, variable_to_plot)
    vmin, vmax, cmap, scatter_size, ticks = plotting_details(z_actual)
    c_actual = axs[0].scatter(x_plot, y_plot, c=z_actual, cmap=cmap, s=scatter_size,vmin=vmin,vmax=vmax)
    fig.colorbar(c_actual, ax=axs[0], orientation="vertical", ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
    c_predicted = axs[1].scatter(x_plot, y_plot, c=z_predicted, cmap=cmap, s=scatter_size, vmin=vmin,vmax=vmax)
    fig.colorbar(c_predicted, ax=axs[1], orientation="vertical", ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
    axes_limits_and_labels(axs, x_feature, y_feature, variable_to_plot)
    save_2d_plot(output_folder, x_feature, y_feature, variable_to_plot)

def plot_scatter_2d_total_velocity(features, actual, predicted, idx_test, output_folder, variables, variables_to_plot):
    x_feature, y_feature = variables_to_plot
    variable_to_plot = 'Total Velocity'

    x_plot, y_plot = get_x_y_for_plot(x_feature, y_feature, features, idx_test)
        
    actual = actual
    predicted = predicted

    z_actual, z_predicted = compute_total_velocity(actual, predicted, variables)
    
    fig, axs = prepare_2d_subplots(x_feature, y_feature, z_actual, variable_to_plot)
    vmin, vmax, cmap, scatter_size, ticks = plotting_details(z_actual)
    c_actual = axs[0].scatter(x_plot, y_plot, c=z_actual, cmap=cmap, s=scatter_size,vmin=vmin,vmax=vmax)
    fig.colorbar(c_actual, ax=axs[0], orientation="vertical", ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
    c_predicted = axs[1].scatter(x_plot, y_plot, c=z_predicted, cmap=cmap, s=scatter_size, vmin=vmin,vmax=vmax)
    fig.colorbar(c_predicted, ax=axs[1], orientation="vertical", ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
    axes_limits_and_labels(axs, x_feature, y_feature, variable_to_plot)
    save_2d_plot(output_folder, x_feature, y_feature, variable_to_plot)

def individual_plot_scatter_2d(wind_angle, features, actual, predicted, output_folder, variables, variables_to_plot):  
    x_feature, y_feature = variables_to_plot[0]
    variable_to_plot = variables_to_plot[1]
    
    if variable_to_plot not in variables:
        raise ValueError(f"{variable_to_plot} not in provided variables")

    x_plot, y_plot = get_x_y_for_plot(x_feature, y_feature, features)

    actual = actual
    predicted = predicted

    z_actual = actual[:, variables.index(variable_to_plot)]
    z_predicted = predicted[:, variables.index(variable_to_plot)]

    fig, axs = prepare_2d_subplots(x_feature, y_feature, z_actual, variable_to_plot, wind_angle)
    vmin, vmax, cmap, scatter_size, ticks = plotting_details(z_actual)
    c_actual = axs[0].scatter(x_plot, y_plot, c=z_actual, cmap=cmap, s=scatter_size,vmin=vmin,vmax=vmax)
    fig.colorbar(c_actual, ax=axs[0], orientation="vertical", ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
    c_predicted = axs[1].scatter(x_plot, y_plot, c=z_predicted, cmap=cmap, s=scatter_size, vmin=vmin,vmax=vmax)
    fig.colorbar(c_predicted, ax=axs[1], orientation="vertical", ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
    axes_limits_and_labels(axs, x_feature, y_feature, variable_to_plot, wind_angle)
    save_2d_plot(output_folder, x_feature, y_feature, variable_to_plot, wind_angle)

def individual_plot_scatter_2d_total_velocity(wind_angle, positions, actual, predicted, output_folder, variables, variables_to_plot):    
    x_feature, y_feature = variables_to_plot
    variable_to_plot = 'Total Velocity'

    x_plot, y_plot = get_x_y_for_plot(x_feature, y_feature, positions)
        
    actual = actual
    predicted = predicted

    z_actual, z_predicted = compute_total_velocity(actual, predicted, variables)
    
    fig, axs = prepare_2d_subplots(x_feature, y_feature, z_actual, variable_to_plot, wind_angle)
    vmin, vmax, cmap, scatter_size, ticks = plotting_details(z_actual)
    c_actual = axs[0].scatter(x_plot, y_plot, c=z_actual, cmap=cmap, s=scatter_size,vmin=vmin,vmax=vmax)
    fig.colorbar(c_actual, ax=axs[0], orientation="vertical", ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
    c_predicted = axs[1].scatter(x_plot, y_plot, c=z_predicted, cmap=cmap, s=scatter_size, vmin=vmin,vmax=vmax)
    fig.colorbar(c_predicted, ax=axs[1], orientation="vertical", ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
    axes_limits_and_labels(axs, x_feature, y_feature, variable_to_plot, wind_angle)
    save_2d_plot(output_folder, x_feature, y_feature, variable_to_plot, wind_angle)

def data_plot_scatter_2d(filenames, datafolder_path, wind_angle, output_folder, variables, variables_to_plot):  
    x_feature, y_feature = variables_to_plot[0]
    variable_to_plot = variables_to_plot[1]

    if variable_to_plot not in variables:
        raise ValueError(f"{variable_to_plot} not in provided variables")

    x_plot, y_plot, z_actual = get_pure_data(filenames, datafolder_path, wind_angle, variables, variable_to_plot, x_feature, y_feature)

    fig, ax = prepare_2d_plots(x_feature, y_feature, z_actual, variable_to_plot, wind_angle)
    vmin, vmax, cmap, scatter_size, ticks = plotting_details(z_actual)
    c_actual = ax.scatter(x_plot, y_plot, c=z_actual, cmap=cmap, s=scatter_size,vmin=vmin,vmax=vmax)
    fig.colorbar(c_actual, ax=ax, orientation="vertical", ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
    axis_limits_and_labels(ax, x_feature, y_feature, variable_to_plot, wind_angle)
    save_2d_plot(output_folder, x_feature, y_feature, variable_to_plot, wind_angle)

def data_plot_scatter_2d_total_velocity(filenames, datafolder_path, wind_angle, output_folder, variables, variables_to_plot):    
    x_feature, y_feature = variables_to_plot
    variable_to_plot = 'Total Velocity'
    variables.append(variable_to_plot)

    x_plot, y_plot, z_actual = get_pure_data_velocity(filenames, datafolder_path, wind_angle, variables, variable_to_plot, x_feature, y_feature)
    
    fig, ax = prepare_2d_plots(x_feature, y_feature, z_actual, variable_to_plot, wind_angle)
    vmin, vmax, cmap, scatter_size, ticks = plotting_details(z_actual)
    c_actual = ax.scatter(x_plot, y_plot, c=z_actual, cmap=cmap, s=scatter_size,vmin=vmin,vmax=vmax)
    fig.colorbar(c_actual, ax=ax, orientation="vertical", ticks=np.linspace(vmin,vmax,ticks,endpoint=True))
    axis_limits_and_labels(ax, x_feature, y_feature, variable_to_plot, wind_angle)
    save_2d_plot(output_folder, x_feature, y_feature, variable_to_plot, wind_angle)

def plot_scatter_2d_total_velocity(X_test_dataframe,y_test_dataframe,predictions_dataframe, plot_folder):
    thetas = [[1,0], [2,30], [3,60], [4,90] , [5,120], [6,135], [7,150], [8,180]]
    params = [['X-Z',570,5],['Y-Z',500,5],['X-Y',50,5]]
    cmap = 'turbo'
    for i in thetas:
        for j in params:
            angle = i[1]
            plane = j[0]
            cut = j[1]
            tolerance = j[2]
            savename = os.path.join(plot_folder,f'{plane}_total_velocity_{cmap}_{angle}')
            plot_data_predictions(X_test_dataframe,y_test_dataframe,predictions_dataframe,angle,savename,plane,cut,tolerance,cmap)


def data_plot_scatter_2d_total_velocity(datafolder,plot_folder):    
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

def plot_prediction_2d_total_velocity(df,angle, plot_folder):
    params = [['X-Z',570,5],['Y-Z',500,5],['X-Y',50,5]]
    cmap = 'jet'
    for j in params:
        plane = j[0]
        cut = j[1]
        tolerance = j[2]
        savename_all = os.path.join(plot_folder,f'{plane}_allplots_{cmap}_{angle}')
        savename_total_velocity = os.path.join(plot_folder,f'{plane}_totalvelocity_{cmap}_{angle}')
        savename_vx = os.path.join(plot_folder,f'{plane}_vx_{cmap}_{angle}')
        savename_vy = os.path.join(plot_folder,f'{plane}_vy_{cmap}_{angle}')
        savename_vz = os.path.join(plot_folder,f'{plane}_vz_{cmap}_{angle}')
        savename_pressure = os.path.join(plot_folder,f'{plane}_pressure_{cmap}_{angle}')
        plot_data_predictions(df,angle,savename_all,savename_total_velocity,savename_vx,savename_vy,savename_vz,savename_pressure,plane,cut,tolerance,cmap)