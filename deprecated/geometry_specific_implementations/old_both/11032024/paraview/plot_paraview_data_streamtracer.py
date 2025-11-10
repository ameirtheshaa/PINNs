import paraview.simple as pvs
import os

angles = [0,15,30,45,60,75,90,105,120,135,150,165,180]

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

    # Apply Surface Vectors Filter
    surfaceVectors = pvs.SurfaceVectors(Input=sliceFilter)
    surfaceVectors.SelectInputVectors = ['POINTS', 'Velocity']  # Adjust 'Velocity' based on your vector field name

    # Apply Mask Points Filter with On Ratio = 23
    maskPoints = pvs.MaskPoints(Input=surfaceVectors)
    maskPoints.OnRatio = 23

    # Apply Stream Tracer with the Mask Points as the seed source
    streamTracer = pvs.StreamTracerWithCustomSource(Input=surfaceVectors,
                                    SeedSource=maskPoints)
    streamTracer.Vectors = ['POINTS', 'Velocity']  # Adjust 'Velocity' based on your vector field name
    streamTracer.MaximumStreamlineLength = 1000.0  # Adjust based on your dataset

    # Color the streamlines by velocity magnitude (or another scalar/vector property)
    surfaceVectorsDisplay = pvs.Show(surfaceVectors, renderView1)
    pvs.ColorBy(surfaceVectorsDisplay, ('POINTS', 'Velocity'), True)
    surfaceVectorsDisplay.SetScalarBarVisibility(renderView1, True)

    streamTracerDisplay = pvs.Show(streamTracer, renderView1)
    pvs.ColorBy(streamTracerDisplay, ('POINTS', 'Velocity'), True)
    streamlineLUT = pvs.GetColorTransferFunction('VelocityMagnitude')
    streamlineLUT.ApplyPreset('X Ray', True)  # Choose a preset or customize your color map
    streamTracerDisplay.LookupTable = streamlineLUT
    streamTracerDisplay.SetScalarBarVisibility(renderView1, True)

    # Update the view to visualize the changes
    pvs.Render(renderView1)

    # Adjust Camera position and focal point for looking down.
    camera = renderView1.GetActiveCamera()
    camera.SetPosition(500, 535, 300)  # Position the camera wayyyy above the slice.
    camera.SetFocalPoint(500, 535, 50)  # Focus on the center of the slice.
    camera.SetViewUp(0, 1, 0)  # Adjust view-up vector. view up is y since we are in the x-y plane.

    pvs.Render()
    screenshot_path = os.path.join(base, "slice_streamTracer_visualization.png")
    screenshot_path2 = os.path.join(base2, f"slice_streamTracer_visualization_{angle}.png")
    pvs.SaveScreenshot(screenshot_path, renderView1)
    pvs.SaveScreenshot(screenshot_path2, renderView1)
    state_path = os.path.join(base, "visualization_state_slice_streamTracer.pvsm")
    pvs.SaveState(state_path)

    pvs.ResetSession()