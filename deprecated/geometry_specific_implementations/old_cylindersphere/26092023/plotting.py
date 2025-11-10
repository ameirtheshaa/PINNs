import os
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import numpy as np
from scipy.interpolate import griddata

def plot_predictions(y_test, predictions, output_folder, variables):
    for i, var in enumerate(variables):
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(y_test)), y_test.cpu().numpy()[:, i], label='Test', s=5)
        plt.scatter(range(len(predictions.cpu())), predictions.cpu().numpy()[:, i], label='Prediction', s=5)
        plt.title(f'Test vs Prediction for {var}')
        plt.legend()
        
        safe_var_name = var.replace(':', '_')
        figure_file_path = os.path.join(output_folder, f'{safe_var_name}_test_vs_prediction.png')
        
        plt.savefig(figure_file_path)
        plt.close()

def plot_3d_scatter_comparison(features, actual, predicted, output_folder, variables):
    x = features['Points:0'].to_numpy()
    y = features['Points:1'].to_numpy()
    z = features['Points:2'].to_numpy()
    
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

def plot_2d_contour_comparison(features, actual, predicted, idx_test, output_folder, variables, variables_to_plot):  
    x_feature, y_feature = variables_to_plot[0]
    variable_to_plot = variables_to_plot[1]
    
    if variable_to_plot not in variables:
        raise ValueError(f"{variable_to_plot} not in provided variables")
    
    x = features['Points:0'].to_numpy()
    y = features['Points:1'].to_numpy()
    z = features['Points:2'].to_numpy()

    x = x[idx_test]
    y = y[idx_test]
    z = z[idx_test]
    
    # Apply mask based on the domain and the idx_test
    mask = (x >= 400) & (x <= 600) & (y >= 400) & (y <= 600) & (z >= 0) & (z <= 100)
    
    # Get the indices where the mask is True
    indices = np.where(mask)[0]

    # Intersect indices with idx_test
    indices = np.intersect1d(indices, idx_test)

    # Use the indices to filter x, y, z, z_actual, and z_predicted
    x = x[indices]
    y = y[indices]
    z = z[indices]
    z_actual = actual[indices, variables.index(variable_to_plot)].cpu().numpy()
    z_predicted = predicted[indices, variables.index(variable_to_plot)].cpu().numpy()

    # Creating a 2D grid and interpolating z values for each grid point
    if x_feature == 'Points:0':
        x_plot = x
    elif x_feature == 'Points:1':
        x_plot = y
    elif x_feature == 'Points:2':
        x_plot = z
    if y_feature == 'Points:0':
        y_plot = x
    elif y_feature == 'Points:1':
        y_plot = y
    elif y_feature == 'Points:2':
        y_plot = z
    xi, yi = np.linspace(x_plot.min(), x_plot.max(), 256), np.linspace(y_plot.min(), y_plot.max(), 256)
    xi, yi = np.meshgrid(xi, yi)
    
    zi_actual = griddata((x_plot, y_plot), z_actual, (xi, yi), method='cubic')
    zi_predicted = griddata((x_plot, y_plot), z_predicted, (xi, yi), method='cubic')
    
    # Creating the contour plots side by side
    fig, axs = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    
    # Actual contour plot
    c_actual = axs[0].contourf(xi, yi, zi_actual, extend='both', cmap=plt.cm.RdBu_r, levels=np.linspace(0,20,1000))
    axs[0].set_title(f'Actual {variable_to_plot}')
    fig.colorbar(c_actual, ax=axs[0], orientation="vertical", ticks=np.linspace(0,20,10, endpoint=True))
    
    # Predicted contour plot
    c_predicted = axs[1].contourf(xi, yi, zi_predicted, extend='both', cmap=plt.cm.RdBu_r, levels=np.linspace(0,20,1000))
    axs[1].set_title(f'Predicted {variable_to_plot}')
    fig.colorbar(c_predicted, ax=axs[1], orientation="vertical", ticks=np.linspace(0,20,10, endpoint=True))
    
    # Setting labels and title
    axs[0].set_xlabel(x_feature)
    axs[0].set_ylabel(y_feature)
    axs[1].set_xlabel(x_feature)
    axs[1].set_ylabel(y_feature)
    fig.suptitle(f'Comparison of Actual vs. Predicted {variable_to_plot} values')
    
    # Saving the figure
    safe_var_name = variable_to_plot.replace(':', '_')
    safe_x_feature = x_feature.replace(':', '')
    safe_y_feature = y_feature.replace(':', '')
    plt.savefig(os.path.join(output_folder, f"{safe_x_feature}_{safe_y_feature}_{safe_var_name}_contour_comparison.png"))
    
    # plt.show()
    
    plt.close()

def plot_total_velocity(features, actual, predicted, idx_test, output_folder, variables, variables_to_plot):    
    x_feature, y_feature = variables_to_plot[0]

    u_index = variables.index('Velocity:0')
    v_index = variables.index('Velocity:1')
    w_index = variables.index('Velocity:2')
    
    x = features['Points:0'].to_numpy()
    y = features['Points:1'].to_numpy()
    z = features['Points:2'].to_numpy()

    x = x[idx_test]
    y = y[idx_test]
    z = z[idx_test]
    
    # Apply mask based on the domain and the idx_test
    mask = (x >= 400) & (x <= 600) & (y >= 400) & (y <= 600) & (z >= 0) & (z <= 100)
    
    # Get the indices where the mask is True
    indices = np.where(mask)[0]

    # Intersect indices with idx_test
    indices = np.intersect1d(indices, idx_test)

    # Use the indices to filter x, y, z, and velocity components
    x = x[indices]
    y = y[indices]
    z = z[indices]
    u_actual = actual[indices, u_index].cpu().numpy()
    v_actual = actual[indices, v_index].cpu().numpy()
    w_actual = actual[indices, w_index].cpu().numpy()
    u_predicted = predicted[indices, u_index].cpu().numpy()
    v_predicted = predicted[indices, v_index].cpu().numpy()
    w_predicted = predicted[indices, w_index].cpu().numpy()
    
    # Compute the magnitude of the velocity vector
    velocity_actual = np.sqrt(u_actual**2 + v_actual**2 + w_actual**2)
    velocity_predicted = np.sqrt(u_predicted**2 + v_predicted**2 + w_predicted**2)

    z_actual = velocity_actual
    z_predicted = velocity_predicted

    variable_to_plot = 'Total Velocity'

    # Creating a 2D grid and interpolating z values for each grid point
    if x_feature == 'Points:0':
        x_plot = x
    elif x_feature == 'Points:1':
        x_plot = y
    elif x_feature == 'Points:2':
        x_plot = z
    if y_feature == 'Points:0':
        y_plot = x
    elif y_feature == 'Points:1':
        y_plot = y
    elif y_feature == 'Points:2':
        y_plot = z
    xi, yi = np.linspace(x_plot.min(), x_plot.max(), 256), np.linspace(y_plot.min(), y_plot.max(), 256)
    xi, yi = np.meshgrid(xi, yi)
    
    zi_actual = griddata((x_plot, y_plot), z_actual, (xi, yi), method='cubic')
    zi_predicted = griddata((x_plot, y_plot), z_predicted, (xi, yi), method='cubic')
    
    # Creating the contour plots side by side
    fig, axs = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    
    # Actual contour plot
    c_actual = axs[0].contourf(xi, yi, zi_actual, extend='both', cmap=plt.cm.RdBu_r, levels=np.linspace(0,20,1000))
    axs[0].set_title(f'Actual {variable_to_plot}')
    fig.colorbar(c_actual, ax=axs[0], orientation="vertical", ticks=np.linspace(0,20,10, endpoint=True))
    
    # Predicted contour plot
    c_predicted = axs[1].contourf(xi, yi, zi_predicted, extend='both', cmap=plt.cm.RdBu_r, levels=np.linspace(0,20,1000))
    axs[1].set_title(f'Predicted {variable_to_plot}')
    fig.colorbar(c_predicted, ax=axs[1], orientation="vertical", ticks=np.linspace(0,20,10, endpoint=True))
    
    # Setting labels and title
    axs[0].set_xlabel(x_feature)
    axs[0].set_ylabel(y_feature)
    axs[1].set_xlabel(x_feature)
    axs[1].set_ylabel(y_feature)
    fig.suptitle(f'Comparison of Actual vs. Predicted {variable_to_plot} values')
    
    # Saving the figure
    safe_var_name = variable_to_plot.replace(':', '_')
    safe_x_feature = x_feature.replace(':', '')
    safe_y_feature = y_feature.replace(':', '')
    plt.savefig(os.path.join(output_folder, f"{safe_x_feature}_{safe_y_feature}_{safe_var_name}_contour_comparison.png"))
    
    # plt.show()
    
    plt.close()