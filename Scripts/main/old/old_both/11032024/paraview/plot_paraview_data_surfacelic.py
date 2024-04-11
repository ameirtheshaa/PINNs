import paraview.simple as pvs
import os

angles = [0,15,30,45,60,75,90,105,120,135,150,165,180]
angles = [0]

for angle in angles:
    base = os.path.join('Z:\\', "cylinder_cell", "data", "core_data", "core_data", f"deg_{angle}")
    base2 = os.path.join('Z:\\', "cylinder_cell", "data")
    # base = os.path.join(os.path.expanduser("~"), "Dropbox", "School", "Graduate", "CERN", "Temp_Files", "nonlineardynamics", "ameir_PINNs", "cylinder_cell", "data", "core_data", "core_data", f"deg_{angle}")
    # base2 = os.path.join(os.path.expanduser("~"), "Dropbox", "School", "Graduate", "CERN", "Temp_Files", "nonlineardynamics", "ameir_PINNs", "cylinder_cell", "data")
    case_file = os.path.join(base, "RESULTS_FLUID_DOMAIN.case")
    data_source = pvs.OpenDataFile(case_file)

    renderView1 = pvs.GetActiveViewOrCreate('RenderView')
    renderView1.ViewSize = [3840, 2160]

    # Convert cell data to point data.
    cell_to_point = pvs.CellDatatoPointData(Input=data_source)

    # Take a slice of the data at a specific height.
    sliceFilter = pvs.Slice(Input=cell_to_point)
    sliceFilter.SliceType = 'Plane'
    sliceFilter.SliceOffsetValues = [0.0]
    sliceFilter.SliceType.Origin = [500, 535, 50] #Slice in the middle of the sphere
    sliceFilter.SliceType.Normal = [0.0, 0.0, 1.0]

    # Show the slice with SurfaceLIC representation.
    slice_display = pvs.Show(sliceFilter, renderView1)
    slice_display.Representation = 'Surface LIC'
    pvs.ColorBy(slice_display, ('POINTS', 'Velocity'))

    # Adjust Camera position and focal point for looking down.
    camera = renderView1.GetActiveCamera()
    camera.SetPosition(500, 535, 500)  # Position the camera wayyyy above the slice.
    camera.SetFocalPoint(500, 535, 50)  # Focus on the center of the slice.
    camera.SetViewUp(0, 1, 0)  # Adjust view-up vector. view up is y since we are in the x-y plane.

    pvs.Render()
    screenshot_path = os.path.join(base, "slice_surfaceLIC_visualization.png")
    screenshot_path2 = os.path.join(base2, f"slice_surfaceLIC_visualization_{angle}.png")
    pvs.SaveScreenshot(screenshot_path, renderView1)
    pvs.SaveScreenshot(screenshot_path2, renderView1)
    state_path = os.path.join(base, "visualization_state_slice_surfaceLIC.pvsm")
    pvs.SaveState(state_path)