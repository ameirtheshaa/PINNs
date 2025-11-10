angles = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]

for i in angles:
    # print(f'cd deg_{i} ; rm -rf *csv ; cd ../; ') 
    print(f'ffmpeg -framerate 24 -i wind_data_visualization_frame_%03d_{i}.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p wind_data_visualization_{i}.mp4') 


# import paraview.simple as pvs
# import os
# import pandas as pd
# import math

# def render_animation(base, renderView, camera, focalPoint, radius, num_frames, filename, wind_angle):
#     for i in range(num_frames):
#         angle = 2 * math.pi * i / num_frames
#         x = focalPoint[0] + radius * math.cos(angle)
#         y = focalPoint[1] + radius * math.sin(angle)
#         z = camera.GetPosition()[2]  # Keeping the Z constant

#         camera.SetPosition(x, y, z)
#         camera.SetFocalPoint(focalPoint)

#         pvs.Render()
#         frame_filename = os.path.join(base, f"{filename}_{i:03d}_{wind_angle}.png")
#         pvs.SaveScreenshot(frame_filename, renderView)

# angles = [0,15,30,45,60,75,90,105,120,135,150,165,180]

# for angle in angles:
# 	# base = os.path.join('Z:\\', "cylinder_cell", "data", "core_data", "core_data", f"deg_{angle}")
# 	base = os.path.join(os.path.expanduser("~"), "Dropbox", "School", "Graduate", "CERN", "Temp_Files", "nonlineardynamics", "ameir_PINNs", "cylinder_cell", "data", "core_data", "core_data", f"deg_{angle}")
# 	base2 = os.path.join(os.path.expanduser("~"), "Dropbox", "School", "Graduate", "CERN", "Temp_Files", "nonlineardynamics", "ameir_PINNs", "cylinder_cell", "data", "3d_screenshots")
# 	case_file = os.path.join(base, "RESULTS_FLUID_DOMAIN.case")
# 	data_source = pvs.OpenDataFile(case_file)
# 	# plotting_stuff = data_source

# 	cellDataArrays = data_source.CellData.keys()
# 	print("\nAvailable Cell Data Arrays:")
# 	for arrayName in cellDataArrays:
# 	    print(arrayName)

# 	# # A contour filter
# 	# contour = pvs.Contour(Input=data_source)
# 	# contour.ContourBy = ['CELLS', 'Velocity']
# 	# plotting_stuff = contour

# 	# # A StreamTracer with point cloud seed type
# 	# streamTracer = pvs.StreamTracer(Input=data_source)
# 	# streamTracer.SeedType = 'Point Cloud'
# 	# streamTracer.Vectors = ['Points', 'Velocity']  # Replace 'velocity' with your vector field name
# 	# streamTracer.SeedType.Center = [500, 500, 50]  # Sphere center coordinates
# 	# streamTracer.SeedType.NumberOfPoints = 1000  # Sphere radius
# 	# plotting_stuff = streamTracer

# 	# A StreamTracer with line seed type
# 	streamTracer = pvs.StreamTracer(Input=data_source)
# 	streamTracer.SeedType = 'Line'
# 	streamTracer.Vectors = ['CELLS', 'Velocity']  # Replace 'velocity' with your vector field name
# 	streamTracer.SeedType.Point1 = [500, 0, 50]  # Starting point of the line
# 	streamTracer.SeedType.Point2 = [500, 1000, 50]  # Ending point of the line
# 	streamTracer.SeedType.Resolution = 1000  # Number of seed points along the line
# 	streamTracer.MaximumStreamlineLength = 1000  # Set your maximum streamline length
# 	plotting_stuff = streamTracer

# 	renderView1 = pvs.GetActiveViewOrCreate('RenderView')
# 	renderView1.ViewSize = [3840, 2160]
# 	pvs.UpdatePipeline(time=None, proxy=plotting_stuff)
	
# 	original_display = pvs.Show(data_source, renderView1)
# 	pvs.ColorBy(original_display, ('CELLS', 'Velocity'))
# 	plot = pvs.Show(plotting_stuff, renderView1)
# 	pvs.ColorBy(plot, ('Points', 'Velocity'))

# 	camera = renderView1.GetActiveCamera()
# 	# camera.SetPosition(500, 535, 50)  # Replace with your desired camera position
# 	# camera.SetFocalPoint(500, 500, 50)   # Replace with the focal point of your scene
# 	# camera.SetViewUp(0, 0, 1)   
# 	camera.SetPosition(238, 836, 196)  # Replace with your desired camera position
# 	camera.SetFocalPoint(479, 526, 55)   # Replace with the focal point of your scene
# 	camera.SetViewUp(0, 0, 1)     

# 	pvs.Render()
# 	pvs.SaveScreenshot(os.path.join(base, "wind_data_visualization.png"), renderView1)
# 	pvs.SaveState(os.path.join(base, "visualization_state.pvsm"))

# 	focalPoint = [479, 526, 55]
# 	radius = 300
# 	num_frames = 360

# 	render_animation(base2, renderView1, camera, focalPoint, radius, num_frames, "wind_data_visualization_frame", angle)

# 	pvs.ResetSession()