import paraview.simple as pvs
import os
import pandas as pd
import numpy as np

angles = [0,15,30,45,60,75,90,105,120,135,150,165,180]
angles = [0]

binssss = [[10,10,10],[20,20,10],[30,30,10],[40,40,10],[50,50,10],[60,60,10]]
binssss = [[10,10,10]]


for i in binssss:
    sampling_bin_size_x = i[0]
    sampling_bin_size_y = i[1]
    sampling_bin_size_z = i[2]

    for angle in angles:

        # Load your case file
        # base = os.path.join('Z:\\', "cylinder_cell", "data", "core_data", "core_data", f"deg_{angle}")
        base = os.path.join('Z:\\', "ladefense", "data", "core_data", "core_data", f"deg_{angle}")
        base2 = os.path.join('Z:\\', "ladefense", "data")
        # base = os.path.join(os.path.expanduser("~"), "Dropbox", "School", "Graduate", "CERN", "Temp_Files", "nonlineardynamics", "ameir_PINNs", "cylinder_cell", "data", "core_data", "core_data", f"deg_{angle}")
        # base2 = os.path.join(os.path.expanduser("~"), "Dropbox", "School", "Graduate", "CERN", "Temp_Files", "nonlineardynamics", "ameir_PINNs", "cylinder_cell", "data")
        case_file = os.path.join(base, "RESULTS_FLUID_DOMAIN.case")
        data_source = pvs.OpenDataFile(case_file)

        cells = pvs.CellCenters(Input=data_source)
        cells.UpdatePipeline()

        # Extract the output of the CellDatatoPointData filter
        output = cells.GetPointDataInformation()

        print (output.ListProperties())

        for i in output:
            print (i)
        
        # # Extract x, y, z arrays
        # coords = output.GetPoints().GetData()
        # x_array = np.array(coords.GetArray(0))
        # y_array = np.array(coords.GetArray(1))
        # z_array = np.array(coords.GetArray(2))