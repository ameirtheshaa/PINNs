import paraview.simple as pvs
import os

angles = [0,15,30,45,60,75,90,105,120,135,150,165,180]

angles = [0]  # Simplified for the example.

for angle in angles:
    base = os.path.join('Z:\\', "cylinder_cell", "data", "core_data", "core_data", f"deg_{angle}")
    case_file = os.path.join(base, "RESULTS_FLUID_DOMAIN.case")
    data_source = pvs.OpenDataFile(case_file)

    renderView1 = pvs.GetActiveViewOrCreate('RenderView')
    renderView1.ViewSize = [3840, 2160]

    # Take a slice of the data at a specific height.
    sliceFilter = pvs.Slice(Input=data_source)
    sliceFilter.SliceType = 'Plane'
    sliceFilter.SliceOffsetValues = [0.0]
    sliceFilter.SliceType.Origin = [500, 535, 50]  # Assuming 70 is the height at which you want the slice.
    sliceFilter.SliceType.Normal = [0.0, 0.0, 1.0]

    # Show the slice with SurfaceLIC representation.
    slice_display = pvs.Show(sliceFilter, renderView1)
    slice_display.Representation = 'Surface LIC'
    pvs.ColorBy(slice_display, ('CELLS', 'Velocity'))

    # Adjust Camera position and focal point for looking down.
    camera = renderView1.GetActiveCamera()
    camera.SetPosition(500, 535, 500)  # Position the camera above the slice.
    camera.SetFocalPoint(500, 535, 50)  # Focus on the center of the slice.
    camera.SetViewUp(0, 1, 0)  # Adjust view-up vector.

    pvs.Render()
    screenshot_path = os.path.join(base, "slice_surfaceLIC_visualization.png")
    pvs.SaveScreenshot(screenshot_path, renderView1)
    state_path = os.path.join(base, "visualization_state_slice_surfaceLIC.pvsm")
    pvs.SaveState(state_path)