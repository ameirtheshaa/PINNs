from definitions import *
from training import *
from PINN import *

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
    cmap = 'turbo'
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

def plot_prediction_2d_total_velocity(X_test_dataframe,y_test_dataframe,predictions_dataframe,angle, plot_folder):
    params = [['X-Z',570,5],['Y-Z',500,5],['X-Y',50,5]]
    cmap = 'turbo'
    for j in params:
        plane = j[0]
        cut = j[1]
        tolerance = j[2]
        savename = os.path.join(plot_folder,f'{plane}_total_velocity_{cmap}_{angle}')
        plot_data_predictions(X_test_dataframe,y_test_dataframe,predictions_dataframe,angle,savename,plane,cut,tolerance,cmap)