import paraview.simple as pvs
import os
import pandas as pd

angles = [0,15,30,45,60,75,90,105,120,135,150,165,180]

res = 1000
point1 = [480, 0, 50]
point2 = [480, 1000, 50]

for angle in angles:

    # base = os.path.join('Z:\\', "cylinder_cell", "data", "core_data", "core_data", f"deg_{angle}")
    base = os.path.join(os.path.expanduser("~"), "Dropbox", "School", "Graduate", "CERN", "Temp_Files", "nonlineardynamics", "ameir_PINNs", "cylinder_cell", "data", "core_data", "core_data", f"deg_{angle}")
    base2 = os.path.join(os.path.expanduser("~"), "Dropbox", "School", "Graduate", "CERN", "Temp_Files", "nonlineardynamics", "ameir_PINNs", "cylinder_cell", "data")
    case_file = os.path.join(base, "RESULTS_FLUID_DOMAIN.case")
    data_source = pvs.OpenDataFile(case_file)

    plot_over_line = pvs.PlotOverLine(Input=data_source, Point1=point1, Point2=point2, Resolution=res, SamplingPattern='Sample Uniformly')
    pvs.UpdatePipeline()

    plot_over_line_filename = os.path.join(base, "plot_over_line.csv")
    pvs.SaveData(plot_over_line_filename, proxy=plot_over_line)

    velocity_gradient = pvs.Gradient(Input=plot_over_line, ScalarArray=['Velocity'], ResultArrayName='GradientVelocity')
    pvs.UpdatePipeline()

    velocity_gradient_filename = os.path.join(base, "velocity_gradient_line.csv")
    pvs.SaveData(velocity_gradient_filename, proxy=velocity_gradient)

    pressure_gradient = pvs.Gradient(Input=plot_over_line, ScalarArray=['Pressure'], ResultArrayName='GradientPressure')
    pvs.UpdatePipeline()

    pressure_gradient_filename = os.path.join(base, "pressure_gradient_line.csv")
    pvs.SaveData(pressure_gradient_filename, proxy=pressure_gradient)

    turbvisc_gradient = pvs.Gradient(Input=plot_over_line, ScalarArray=['TurbVisc'], ResultArrayName='GradientTurbVisc')
    pvs.UpdatePipeline()

    turbvisc_gradient_filename = os.path.join(base, "turbvisc_gradient_line.csv")
    pvs.SaveData(turbvisc_gradient_filename, proxy=turbvisc_gradient)

    velocity_second_gradient = pvs.Gradient(Input=velocity_gradient, ScalarArray=['CELLS', 'GradientVelocity'], ResultArrayName='SecondGradientVelocity')
    pvs.UpdatePipeline()

    velocity_second_gradient_filename = os.path.join(base, "velocity_second_gradient_line.csv")
    pvs.SaveData(velocity_second_gradient_filename, proxy=velocity_second_gradient)

    pressure_second_gradient = pvs.Gradient(Input=pressure_gradient, ScalarArray=['GradientPressure'], ResultArrayName='SecondGradientPressure')
    pvs.UpdatePipeline()

    pressure_second_gradient_filename = os.path.join(base, "pressure_second_gradient_line.csv")
    pvs.SaveData(pressure_second_gradient_filename, proxy=pressure_second_gradient)

    df_velocity = pd.read_csv(velocity_second_gradient_filename)
    df_pressure = pd.read_csv(pressure_gradient_filename)
    df_turbvisc = pd.read_csv(turbvisc_gradient_filename)

    df_combined = pd.concat([df_velocity, df_pressure, df_turbvisc], axis=1)

    combined_filename = os.path.join(base2, f"line_data_{angle}.csv")
    df_combined.to_csv(combined_filename, index=False)

    print (len(df_combined), len(df_combined.columns))