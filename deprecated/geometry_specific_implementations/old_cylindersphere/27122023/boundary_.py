import numpy as np
import plotly.graph_objects as go
from stl import mesh
import matplotlib.pyplot as plt
from definitions import *

filename = 'scaled_cylinder_sphere.stl'
your_mesh = mesh.Mesh.from_file(filename)
flattened = your_mesh.vectors.reshape(-1, 3)

faces = restructure_data(flattened)
subfaces = random.sample(faces, 10000)
no_slip_normals = [compute_normal(*face) for face in subfaces]

print (len(flattened), len(faces), len(subfaces), len(no_slip_normals))