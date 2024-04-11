import paraview.simple as pvs
import os
import pandas as pd

angles = [0,15,30,45,60,75,90,105,120,135,150,165,180]

for angle in angles:

    # Load your case file
    base = os.path.join('Z:\\', "ladefense", "data", "core_data", "core_data_sampled_70_70_10")
    base2 = os.path.join('Z:\\', "ladefense", "data")
    # base = os.path.join(os.path.expanduser("~"), "Dropbox", "School", "Graduate", "CERN", "Temp_Files", "nonlineardynamics", "ameir_PINNs", "cylinder_cell", "data", "core_data", "core_data", f"deg_{angle}")
    # base2 = os.path.join(os.path.expanduser("~"), "Dropbox", "School", "Graduate", "CERN", "Temp_Files", "nonlineardynamics", "ameir_PINNs", "cylinder_cell", "data")
    case_file = os.path.join(base, f"sampled_data_{angle}.vtk")
    data_source = pvs.OpenDataFile(case_file)

    cell_centers = pvs.CellCenters(Input=data_source)

    cell_centers_filename = os.path.join(base2, f"Sampled_70_70_10_CFD_cell_data_simulation_{angle}.csv")
    pvs.SaveData(cell_centers_filename, proxy=cell_centers)

    pvs.UpdatePipeline()

    # # # First-order gradients
    # # velocity_gradient = pvs.Gradient(Input=data_source, ScalarArray=['POINTS', 'Velocity'], ResultArrayName='GradientVelocity')
    # # pvs.UpdatePipeline()
    # # pressure_gradient = pvs.Gradient(Input=data_source, ScalarArray=['POINTS', 'Pressure'], ResultArrayName='GradientPressure')
    # # pvs.UpdatePipeline()
    # # turbvisc_gradient = pvs.Gradient(Input=data_source, ScalarArray=['POINTS', 'TurbVisc'], ResultArrayName='GradientTurbVisc')
    # # pvs.UpdatePipeline()

    # # # Second-order gradients
    # # velocity_second_gradient = pvs.Gradient(Input=velocity_gradient, ScalarArray=['POINTS', 'GradientVelocity'], ResultArrayName='SecondGradientVelocity')
    # # pvs.UpdatePipeline()
    # # pressure_second_gradient = pvs.Gradient(Input=pressure_gradient, ScalarArray=['POINTS', 'GradientPressure'], ResultArrayName='SecondGradientPressure')
    # # pvs.UpdatePipeline()

    # # First-order gradients
    # velocity_gradient = pvs.Gradient(Input=data_source, ScalarArray=['CELLS', 'Velocity'], ResultArrayName='GradientVelocity')
    # pvs.UpdatePipeline()
    # pressure_gradient = pvs.Gradient(Input=data_source, ScalarArray=['CELLS', 'Pressure'], ResultArrayName='GradientPressure')
    # pvs.UpdatePipeline()
    # turbvisc_gradient = pvs.Gradient(Input=data_source, ScalarArray=['CELLS', 'TurbVisc'], ResultArrayName='GradientTurbVisc')
    # pvs.UpdatePipeline()

    # # Second-order gradients
    # velocity_second_gradient = pvs.Gradient(Input=velocity_gradient, ScalarArray=['CELLS', 'GradientVelocity'], ResultArrayName='SecondGradientVelocity')
    # pvs.UpdatePipeline()
    # pressure_second_gradient = pvs.Gradient(Input=pressure_gradient, ScalarArray=['CELLS', 'GradientPressure'], ResultArrayName='SecondGradientPressure')
    # pvs.UpdatePipeline()

    # # # First-order gradients
    # # velocity_gradient = pvs.Gradient(Input=cell_centers, ScalarArray=['CELLS', 'Velocity'], ResultArrayName='GradientVelocity')
    # # pvs.UpdatePipeline()
    # # pressure_gradient = pvs.Gradient(Input=cell_centers, ScalarArray=['CELLS', 'Pressure'], ResultArrayName='GradientPressure')
    # # pvs.UpdatePipeline()
    # # turbvisc_gradient = pvs.Gradient(Input=cell_centers, ScalarArray=['CELLS', 'TurbVisc'], ResultArrayName='GradientTurbVisc')
    # # pvs.UpdatePipeline()

    # # # Second-order gradients
    # # velocity_second_gradient = pvs.Gradient(Input=velocity_gradient, ScalarArray=['CELLS', 'GradientVelocity'], ResultArrayName='SecondGradientVelocity')
    # # pvs.UpdatePipeline()
    # # pressure_second_gradient = pvs.Gradient(Input=pressure_gradient, ScalarArray=['CELLS', 'GradientPressure'], ResultArrayName='SecondGradientPressure')
    # # pvs.UpdatePipeline()

    # # Saving first-order gradient data
    # velocity_gradient_filename = os.path.join(base, "velocity_gradient.csv")
    # pvs.SaveData(velocity_gradient_filename, proxy=velocity_gradient)

    # pressure_gradient_filename = os.path.join(base, "pressure_gradient.csv")
    # pvs.SaveData(pressure_gradient_filename, proxy=pressure_gradient)

    # turbvisc_gradient_filename = os.path.join(base, "turbvisc_gradient.csv")
    # pvs.SaveData(turbvisc_gradient_filename, proxy=turbvisc_gradient)

    # # Saving second-order gradient data
    # velocity_second_gradient_filename = os.path.join(base, "velocity_second_gradient.csv")
    # pvs.SaveData(velocity_second_gradient_filename, proxy=velocity_second_gradient)

    # pressure_second_gradient_filename = os.path.join(base, "pressure_second_gradient.csv")
    # pvs.SaveData(pressure_second_gradient_filename, proxy=pressure_second_gradient)

    # data_source_vtk = pvs.servermanager.Fetch(data_source)
    # velocity_gradient_vtk = pvs.servermanager.Fetch(velocity_gradient)

    # print (f'wind angle = {angle} deg')

    # # Print the number of points and cells in the original data source
    # print("Number of points in data_source:", data_source_vtk.GetNumberOfPoints())
    # print("Number of cells in data_source:", data_source_vtk.GetNumberOfCells())

    # # Print the number of points and cells in the gradient
    # print("Number of points in velocity_gradient:", velocity_gradient_vtk.GetNumberOfPoints())
    # print("Number of cells in velocity_gradient:", velocity_gradient_vtk.GetNumberOfCells())

    # # Read CSV files using pandas
    # df_velocity = pd.read_csv(velocity_second_gradient_filename)
    # df_pressure = pd.read_csv(pressure_gradient_filename)
    # df_turbvisc = pd.read_csv(turbvisc_gradient_filename)
    # df_cellcenters = pd.read_csv(cell_centers_filename)

    # # Concatenate the two DataFrames
    # df_combined = pd.concat([df_velocity, df_pressure, df_turbvisc, df_cellcenters], axis=1)

    # # Save the combined DataFrame to a new CSV file
    # combined_filename = os.path.join(base, f"all_data_{angle}.csv")
    # df_combined.to_csv(combined_filename, index=False)

    # print (len(df_combined), len(df_combined.columns))