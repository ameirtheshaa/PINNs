import os
import pandas as pd
import matplotlib.pyplot as plt

# Base directory
base_directory = "/Users/ameirshaa/Dropbox/School/Graduate/CERN/Temp_Files/nonlineardynamics/ameir_PINNs/cylinder_cell"

folders_to_compare = [["24092023_adam_datalossonly","26092023_lbgfs_datalossonly"]]

def read_csv_from_folder(base_dir, folder_name, simulation_folder):
    file_path = os.path.join(base_dir, folder_name, simulation_folder, "metrics.csv")
    return pd.read_csv(file_path)

def compare_results(adam_df, lbgfs_df):
    comparison_results = pd.DataFrame()
    comparison_results['Variable'] = adam_df['Variable']
    comparison_results['MSE_Ratio'] = adam_df['MSE'] / lbgfs_df['MSE']
    comparison_results['Better_Optimizer_MSE'] = comparison_results['MSE_Ratio'].apply(lambda x: 'Adam' if x < 1 else 'LBGFS')
    comparison_results['RMSE_Ratio'] = adam_df['RMSE'] / lbgfs_df['RMSE']
    comparison_results['Better_Optimizer_RMSE'] = comparison_results['RMSE_Ratio'].apply(lambda x: 'Adam' if x < 1 else 'LBGFS')
    comparison_results['MAE_Ratio'] = adam_df['MAE'] / lbgfs_df['MAE']
    comparison_results['Better_Optimizer_MAE'] = comparison_results['MAE_Ratio'].apply(lambda x: 'Adam' if x < 1 else 'LBGFS')
    comparison_results['R2_Ratio'] = adam_df['R2'] / lbgfs_df['R2']
    comparison_results['Better_Optimizer_R2'] = comparison_results['R2_Ratio'].apply(lambda x: 'Adam' if x > 1 else 'LBGFS')
    return comparison_results

# Save comparison results to a CSV file and save the plot to the output folder
for folders in folders_to_compare:
    for i in range(1, 8):  # For x in CFD_cell_data_simulation_x where x ranges from 1 to 7
        folder_adam = folders[0]
        folder_lbgfs = folders[1]
        date = folder_lbgfs.split('_')[0]
        type_ = folder_adam.split('_')[2]  # Adjusted to get the correct 'type_' value
        base_filename = f"comparison_{date}_{type_}"
        output_folder = os.path.join(base_directory, base_filename)
        os.makedirs(output_folder, exist_ok=True)
        
        simulation_folder = f"CFD_cell_data_simulation_{i}"
        
        # Read the CSV files from the corresponding folders
        results_adam = read_csv_from_folder(base_directory, folder_adam, simulation_folder)
        results_lbgfs = read_csv_from_folder(base_directory, folder_lbgfs, simulation_folder)
        
        # Compare the MSE results and print/display the comparison
        comparison_results = compare_results(results_adam, results_lbgfs)
        print(comparison_results)
        
        # Save the comparison results to a CSV file in the output folder
        comparison_results.to_csv(os.path.join(output_folder, f"comparison_results_{i}.csv"), index=False)
        
        # Visualization
        fig, ax = plt.subplots(figsize=(10,6))
        width = 0.35  # the width of the bars
        r1 = range(len(comparison_results['Variable']))
        r2 = [x + width for x in r1]
        
        plt.bar(r1, results_adam['MSE'], width = width, color = 'b', edgecolor = 'grey', label='Adam')
        plt.bar(r2, results_lbgfs['MSE'], width = width, color = 'r', edgecolor = 'grey', label='LBGFS')
        plt.xlabel('Variables', fontweight='bold', fontsize = 15)
        plt.ylabel('MSE Values', fontweight='bold', fontsize = 15)
        plt.xticks([r + width/2 for r in range(len(comparison_results['Variable']))], comparison_results['Variable'])
        plt.title('Comparison of MSE Values between Adam and LBGFS Optimizers')
        plt.legend()
        
        # Save the plot to the output folder
        plt.savefig(os.path.join(output_folder, f"comparison_plot_{i}.png"))
        plt.close(fig)  # Close the figure after saving to free up memory