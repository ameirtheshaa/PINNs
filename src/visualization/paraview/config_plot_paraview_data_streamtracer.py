import paraview.simple as pvs

config = {
    "training": {
        "angles_to_leave_out": [135],
        "angles_to_train": [0,15,30,45,60,75,90,105,120,150,165,180],
        "all_angles": [0,15,30,45,60,75,90,105,120,135,150,165,180],
        "boundary": [[0,1000,0,1000,0,1000],100], #[[-2520,2520,-2520,2520,0,1000],100]
    },
    "machine": {
        "mac": os.path.join(os.path.expanduser("~"), "Dropbox", "School", "Graduate", "CERN", "Temp_Files", "nonlineardynamics", "ameir_PINNs", "cylinder_cell"),
        "CREATE": os.path.join('Z:\\', "cylinder_cell"),
        "google": f"/content/drive/Othercomputers/MacMini/cylinder_cell",
    },
    "data": {
        "geometry": 'scaled_cylinder_sphere.stl' #'ladefense.stl'
    },
    "train_test": {
        "new_angles": [0,15,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,75,90,105,120,135,150,165,180]
    },
    "chosen_machine": "mac",
    "chosen_optimizer": "adam_optimizer",
    "base_folder_names": "test",
    "base_folder_names": ["test"]
}

# Launch ParaView server
pvs.Connect()

# The path to your macro or script
macro_path = "/path/to/your/macro.py"

# Add or run the macro
# There isn't a direct API call to add a macro in ParaView's simple API,
# but you can execute a script directly which can be considered as running a macro.
exec(open(macro_path).read())

# Additional commands to manipulate ParaView can go here

# Disconnect from the server once done
pvs.Disconnect()