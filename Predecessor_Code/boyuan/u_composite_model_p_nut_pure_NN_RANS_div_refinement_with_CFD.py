import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time

gpus = tf.config.list_physical_devices("GPU")
print(gpus)
device = gpus[0]
# tf.config.set_visible_devices(device)

# ---------------------- #

# data type
data_type = tf.float64
tf.keras.backend.set_floatx('float64')

# ---------------------- #

# constants
U = 4.043564066
rho = 1.23
h = 300.0

# load meteo data
df_meteo = pd.read_csv("../../cylinder/meteo.csv")

# load CFD data position
file_path = "../../cylinder_cell_data/"
df = pd.read_csv(file_path + f"CFD_cell_data_simulation_central_region_subsampled_1.csv")
x_train = tf.cast(tf.reshape((df.to_numpy()[:, 0] - 500.0) / 300.0, [-1, 1]), dtype=data_type)
y_train = tf.cast(tf.reshape((df.to_numpy()[:, 1] - 500.0) / 300.0, [-1, 1]), dtype=data_type)
z_train = tf.cast(tf.reshape(df.to_numpy()[:, 2] / 300.0, [-1, 1]), dtype=data_type)

train_data_list = []
PDE_train_angle = np.arange(270, 90-15, -15)
for wind_angle in PDE_train_angle:
    wind_angle = np.deg2rad(wind_angle)
    theta_train = tf.ones_like(x_train, dtype=data_type) * wind_angle
    train_data = tf.concat([x_train, y_train, z_train, theta_train], axis=1)
    train_data_list.append(train_data)
    
train_data_list = tf.concat(train_data_list, axis=0)
print("unsupervised part training data:")
print(train_data_list)

train_dataset_PDE = tf.data.Dataset.from_tensor_slices(train_data_list)
train_dataset_PDE = train_dataset_PDE.shuffle(100000).batch(6500)


# simulation data
file_path = "../../cylinder_cell_data/"

train_data = []
train_label = []
for i in range(1, 8):
    df = pd.read_csv(file_path + f"CFD_cell_data_simulation_central_region_subsampled_{i}.csv")
    
    # train data
    pos = df.to_numpy()[:, 0:3]
    
    # scaling
    # x, y to [-1, 1]
    # z to [0, 1]
    pos[:, 0] = (pos[:, 0] - 500.0) / 300.0
    pos[:, 1] = (pos[:, 1] - 500.0) / 300.0
    pos[:, 2] = pos[:, 2] / 300.0
    
    theta = df_meteo["Orientation"][i-1]
    theta = np.ones((pos.shape[0], 1)) * theta
    pos = np.hstack((pos, theta))
    
    # train label
    ux = df.to_numpy()[:, 5]
    uy = df.to_numpy()[:, 6]
    uz = df.to_numpy()[:, 7]
    p = df.to_numpy()[:, 3]
    nut = df.to_numpy()[:, 4]

    # scaling
    ux = ux / U
    uy = uy / U
    uz = uz / U
    p = p / (rho * U**2)
    nut = nut / (U * h)
    
    flow_attributes = np.vstack((ux, uy, uz, p, nut)).T
    
    train_data.append(pos)
    train_label.append(flow_attributes)
    
train_data = tf.concat(train_data, axis=0)
train_label = tf.concat(train_label, axis=0)


print("supervised part training data and label:")
print(train_data, train_label)

train_dataset_CFD = tf.data.Dataset.from_tensor_slices((train_data, train_label))
train_dataset_CFD = train_dataset_CFD.shuffle(50000).batch(3500)


# ---------------------- #

# define composite model
NN_mask = tf.keras.models.load_model("../../model_mask/model_11")

# load pretrained NN for ux, uy, uz, p, nut
NN_ux = tf.keras.models.load_model("../models/composite_model_simpler_cell_data_w50_hl10_tanh_ux")
NN_uy = tf.keras.models.load_model("../models/composite_model_simpler_cell_data_w50_hl10_tanh_uy")
NN_uz = tf.keras.models.load_model("../models/composite_model_simpler_cell_data_w50_hl10_tanh_uz")
NN_p = tf.keras.models.load_model("../models/p_pure_nn_w50_hl10_tanh")
NN_nut = tf.keras.models.load_model("../models/nut_pure_nn_w50_hl10_tanh")

# def get_NN():
#     width = 50
#     num_hl = 10
#     activation_type = "tanh"
#     inputs = tf.keras.Input(shape=(4,))
#     for i in range(num_hl):
#         if i == 0:
#             x = tf.keras.layers.Dense(width, activation=activation_type, use_bias=True)(inputs)
#         x = tf.keras.layers.Dense(width, activation=activation_type, use_bias=True)(x)
        
#     outputs = tf.keras.layers.Dense(1, activation="linear", use_bias=True)(x)
#     model = tf.keras.Model(inputs=inputs, outputs=outputs)
#     return model
# NN_ux = get_NN()
# NN_uy = get_NN()
# NN_uz = get_NN()
# NN_p = get_NN()
# NN_nut = get_NN()


h = tf.constant(300.0, dtype=data_type)
U = 4.043564066
z_ref = df_meteo["Height"][0] / h

def input_convert(theta, x, y, z):
    x_ = tf.cast(tf.reshape(x, [-1,1]), dtype=data_type)
    y_ = tf.cast(tf.reshape(y, [-1,1]), dtype=data_type)
    z_ = tf.cast(tf.reshape(z, [-1,1]), dtype=data_type)
    theta_ = tf.cast(tf.reshape(theta, [-1,1]), dtype=data_type)
    
    return theta_, x_, y_, z_


def velocity_mean_x(varphi, x, y, z): 
    """
    mean flow to ensure boundary conditions
    """
    varphi_, x_, y_, z_ = input_convert(varphi, x, y, z)
    z_ref = 100.0 / h  # make it variables
    log_profile = tf.math.log(z_ + 1) / tf.math.log(z_ref + 1)
    U_x = tf.math.cos(varphi_) * log_profile

    return U_x


def velocity_mean_y(varphi, x, y, z):
    """
    mean flow to ensure boundary conditions
    """
    varphi_, x_, y_, z_ = input_convert(varphi, x, y, z)
    z_ref = 100.0 / h  # make it variables
    log_profile = tf.math.log(z_ + 1) / tf.math.log(z_ref + 1)
    U_y = tf.math.sin(varphi_) * log_profile

    return U_y


def velocity_mean_z(varphi, x, y, z): 
    """
    mean flow to ensure boundary conditions
    """
    varphi_, x_, y_, z_ = input_convert(varphi, x, y, z)
    U_z = 0 * z_

    return U_z


def ux_model(data):
    # data: N by 4
    # [x, y, z, theta]
    
    U_x = velocity_mean_x(data[:, 3], data[:, 0], data[:, 1], data[:, 2])
#     print(U_x.shape)
    
    g_z = tf.reshape(tf.math.sin(np.pi * data[:, 2]), [-1, 1])
#     print(g_z.shape)
    
    ux = NN_mask(data[:, 0:3]) * (U_x + g_z * NN_ux(data))
    
    return ux

def uy_model(data):
    # data: N by 4
    # [x, y, z, theta]
    
    U_y = velocity_mean_y(data[:, 3], data[:, 0], data[:, 1], data[:, 2])
#     print(U_y.shape)
    
    g_z = tf.reshape(tf.math.sin(np.pi * data[:, 2]), [-1, 1])
#     print(g_z.shape)
    
    uy = NN_mask(data[:, 0:3]) * (U_y + g_z * NN_uy(data))
    
    return uy

def uz_model(data):
    # data: N by 4
    # [x, y, z, theta]
    
    U_z = velocity_mean_z(data[:, 3], data[:, 0], data[:, 1], data[:, 2])
#     print(U_z.shape)
    
    g_z = tf.reshape(tf.math.sin(np.pi * data[:, 2]), [-1, 1])
#     print(g_z.shape)
    
    uz = NN_mask(data[:, 0:3]) * (U_z + g_z * NN_uz(data))
    
    return uz

# ---------------------- #

def compute_PDE_residues(x_test, y_test, z_test, theta_test):
    # compute the RANS residue for multiple points at one wind angle
    with tf.GradientTape(persistent=True) as tape1:
        tape1.watch(x_test)
        tape1.watch(y_test)
        tape1.watch(z_test)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x_test)
            tape.watch(y_test)
            tape.watch(z_test)
            
            # for NN inputs
            points_test = tf.concat([x_test, y_test, z_test, theta_test], axis=1)
            
            # forward pass
            ux = ux_model(points_test)
            uy = uy_model(points_test)
            uz = uz_model(points_test)
            p = NN_p(points_test)
            nut = NN_nut(points_test)
            
        dux_dx = tape.gradient(ux, x_test)
        dux_dy = tape.gradient(ux, y_test)
        dux_dz = tape.gradient(ux, z_test)
        
        duy_dx = tape.gradient(uy, x_test)
        duy_dy = tape.gradient(uy, y_test)
        duy_dz = tape.gradient(uy, z_test)
        
        duz_dx = tape.gradient(uz, x_test)
        duz_dy = tape.gradient(uz, y_test)
        duz_dz = tape.gradient(uz, z_test)
        
        dp_dx = tape.gradient(p, x_test)
        dp_dy = tape.gradient(p, y_test)
        dp_dz = tape.gradient(p, z_test)
        
        dnut_dx = tape.gradient(nut, x_test)
        dnut_dy = tape.gradient(nut, y_test)
        dnut_dz = tape.gradient(nut, z_test)
    
    d2ux_dx2 = tape1.gradient(dux_dx, x_test)
    d2ux_dy2 = tape1.gradient(dux_dy, y_test)
    d2ux_dz2 = tape1.gradient(dux_dz, z_test)
    
    d2uy_dx2 = tape1.gradient(duy_dx, x_test)
    d2uy_dy2 = tape1.gradient(duy_dy, y_test)
    d2uy_dz2 = tape1.gradient(duy_dz, z_test)
    
    d2uz_dx2 = tape1.gradient(duz_dx, x_test)
    d2uz_dy2 = tape1.gradient(duz_dy, y_test)
    d2uz_dz2 = tape1.gradient(duz_dz, z_test)
        
    # term 1
    term_1_x = dux_dx * ux + dux_dy * uy + dux_dz * uz
    term_1_y = duy_dx * ux + duy_dy * uy + duy_dz * uz
    term_1_z = duz_dx * ux + duz_dy * uy + duz_dz * uz
    
    # term 2
    term_2_x = -dp_dx
    term_2_y = -dp_dy
    term_2_z = -dp_dz
    
    # term 3
    term_3_x = nut * (d2ux_dx2 + d2ux_dy2 + d2ux_dz2)
    term_3_y = nut * (d2uy_dx2 + d2uy_dy2 + d2uy_dz2)
    term_3_z = nut * (d2uz_dx2 + d2uz_dy2 + d2uz_dz2)

    # term 4
    term_4_x = dnut_dx * (dux_dx + dux_dx) + dnut_dy * (dux_dy + duy_dx) + dnut_dz * (dux_dz + duz_dx)
    term_4_y = dnut_dx * (duy_dx + dux_dy) + dnut_dy * (duy_dy + duy_dy) + dnut_dz * (duy_dz + duz_dy)
    term_4_z = dnut_dx * (duz_dx + dux_dz) + dnut_dy * (duz_dy + duy_dz) + dnut_dz * (duz_dz + duz_dz)
    
    # RANS
    loss_rans_x = term_1_x - term_2_x - term_3_x - term_4_x
    loss_rans_y = term_1_y - term_2_y - term_3_y - term_4_y
    loss_rans_z = term_1_z - term_2_z - term_3_z - term_4_z
    loss_rans = tf.math.sqrt(tf.square(loss_rans_x) + tf.square(loss_rans_y) + tf.square(loss_rans_z))
    
    # div
    div = dux_dx + duy_dy + duz_dz
    loss_div = div
    
    
    del tape
    del tape1
    
    return loss_rans, loss_div

# ---------------------- #

# check before training

# sample on uniform grid and plot the results
# test on xy plane
resolution = 256
test_height = 50  # unit: meter

theta_range = np.arange(270, 90-15, -15)

for theta in theta_range:
    test_theta = np.deg2rad(theta)
    
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    x_grid, y_grid = np.meshgrid(x, y)
    x_test = np.reshape(x_grid, (-1, 1))
    y_test = np.reshape(y_grid, (-1, 1))
    z_test = np.ones_like(x_test) * test_height / 300.0
    theta_test = np.ones_like(x_test) * test_theta
    
    x_test = tf.cast(x_test, dtype=data_type)
    y_test = tf.cast(y_test, dtype=data_type)
    z_test = tf.cast(z_test, dtype=data_type)
    theta_test = tf.cast(theta_test, dtype=data_type)
    
    loss_rans, loss_div = compute_PDE_residues(x_test, y_test, z_test, theta_test)
    
    loss_rans = tf.reshape(loss_rans, x_grid.shape)
    loss_div = tf.reshape(loss_div, x_grid.shape)
    
    # plot xy slice for rans
    fig, ax = plt.subplots(figsize=[8,8])
    im = ax.contourf(x_grid, y_grid, loss_rans, extend='both', cmap=plt.cm.RdBu_r, levels=np.linspace(0,20,1000))
    cbar_0 = fig.colorbar(im, ax=ax, orientation="horizontal", ticks=np.linspace(0,20,10, endpoint=True))
    plt.xlabel("x")
    plt.ylabel("y")
    ax.set_title(f'RANS residue on the xy plane, theta={theta}')
    plt.tight_layout(pad=2)
    plt.show()
    
    # plot xy slice for div
    fig, ax = plt.subplots(figsize=[8,8])
    im = ax.contourf(x_grid, y_grid, loss_div, extend='both', cmap=plt.cm.RdBu_r, levels=np.linspace(-10,10,1000))
    cbar_0 = fig.colorbar(im, ax=ax, orientation="horizontal", ticks=np.linspace(-10,10,10, endpoint=True))
    plt.xlabel("x")
    plt.ylabel("y")
    ax.set_title(f'div residue on the xy plane, theta={theta}')
    # plt.gca().invert_xaxis()
    plt.tight_layout(pad=2)
    plt.show()
    
    # plot xy slice for u norm
    test_data = tf.concat([x_test, y_test, z_test, theta_test], axis=1)
    test_data = tf.cast(test_data, dtype=data_type)
    ux_pred = ux_model(test_data) * U
    uy_pred = uy_model(test_data) * U
    uz_pred = uz_model(test_data) * U
    u_norm = np.linalg.norm(tf.concat([ux_pred, uy_pred, uz_pred], axis=1), axis=1)
    u_norm = tf.reshape(u_norm, x_grid.shape)
    
    fig, ax = plt.subplots(figsize=[8,8])
    im = ax.contourf(x_grid, y_grid, u_norm, extend='both', cmap=plt.cm.RdBu_r, levels=np.linspace(0,5,1000))
    cbar_0 = fig.colorbar(im, ax=ax, orientation="horizontal", ticks=np.linspace(0,5,4, endpoint=True))
    
    # plot wind direction
    wind_dir = np.array([np.cos(test_theta), np.sin(test_theta)])
    ax.quiver(0, 0.7, wind_dir[0], wind_dir[1], color="k", scale=10, width=0.001, headwidth=4, headlength=4, headaxislength=3)
    
    plt.xlabel("x")
    plt.ylabel("y")
    ax.set_title(f'|u| on the xy plane, theta={theta}')
    plt.tight_layout(pad=2)
    plt.show()
    

# ---------------------- #

# test on yz plane
U = 4.043564066
resolution = 256
test_x = 500  # unit: meter

theta_range = np.arange(270, 90-15, -15)

for theta in theta_range:
    test_theta = np.deg2rad(theta)
    
    y = np.linspace(-1, 1, resolution)
    z = np.linspace(0, 1, resolution)
    y_grid, z_grid = np.meshgrid(y, z)
    y_test = np.reshape(y_grid, (-1, 1))
    z_test = np.reshape(z_grid, (-1, 1))
    x_test = np.ones_like(y_test) * (test_x - 500) / 300.0
    theta_test = np.ones_like(y_test) * test_theta
    
    x_test = tf.cast(x_test, dtype=data_type)
    y_test = tf.cast(y_test, dtype=data_type)
    z_test = tf.cast(z_test, dtype=data_type)
    theta_test = tf.cast(theta_test, dtype=data_type)
    
    loss_rans, loss_div = compute_PDE_residues(x_test, y_test, z_test, theta_test)
    
    loss_rans = tf.reshape(loss_rans, y_grid.shape)
    loss_div = tf.reshape(loss_div, y_grid.shape)
    
    # plot xy slice for rans
    fig, ax = plt.subplots(figsize=[8,8])
    im = ax.contourf(y_grid, z_grid, loss_rans, extend='both', cmap=plt.cm.RdBu_r, levels=np.linspace(0,20,1000))
    cbar_0 = fig.colorbar(im, ax=ax, orientation="horizontal", ticks=np.linspace(0,20,10, endpoint=True))
    plt.xlabel("y")
    plt.ylabel("z")
    ax.set_title(f'RANS residue on the yz plane, theta={theta}')
    plt.tight_layout(pad=2)
    plt.show()
    
    # plot xy slice for div
    fig, ax = plt.subplots(figsize=[8,8])
    im = ax.contourf(y_grid, z_grid, loss_div, extend='both', cmap=plt.cm.RdBu_r, levels=np.linspace(-10,10,1000))
    cbar_0 = fig.colorbar(im, ax=ax, orientation="horizontal", ticks=np.linspace(-10,10,10, endpoint=True)) 
    plt.xlabel("y")
    plt.ylabel("z")
    ax.set_title(f'div residue on the yz plane, theta={theta}')
    plt.tight_layout(pad=2)
    plt.show()
    
    # plot xy slice for u norm
    test_data = tf.concat([x_test, y_test, z_test, theta_test], axis=1)
    test_data = tf.cast(test_data, dtype=data_type)
    ux_pred = ux_model(test_data) * U
    uy_pred = uy_model(test_data) * U
    uz_pred = uz_model(test_data) * U
    u_norm = np.linalg.norm(tf.concat([ux_pred, uy_pred, uz_pred], axis=1), axis=1)
    u_norm = tf.reshape(u_norm, y_grid.shape)
    
    fig, ax = plt.subplots(figsize=[8,8])
    im = ax.contourf(y_grid, z_grid, u_norm, extend='both', cmap=plt.cm.RdBu_r, levels=np.linspace(0,5,1000))
    cbar_0 = fig.colorbar(im, ax=ax, orientation="horizontal", ticks=np.linspace(0,5,4, endpoint=True))
    
    # plot wind direction
#     wind_dir = np.array([np.cos(test_theta), np.sin(test_theta)])
#     ax.quiver(0, 0.7, wind_dir[0], wind_dir[1], color="k", scale=10, width=0.001, headwidth=4, headlength=4, headaxislength=3)
    
    plt.xlabel("y")
    plt.ylabel("z")
    ax.set_title(f'|u| on the yz plane, theta={theta}')
    plt.tight_layout(pad=2)
    plt.show()

# ---------------------- #

# test on xz plane
U = 4.043564066
resolution = 256
test_y = 500  # unit: meter

theta_range = np.arange(270, 90-15, -15)

for theta in theta_range:
    test_theta = np.deg2rad(theta)
    
    x = np.linspace(-1, 1, resolution)
    z = np.linspace(0, 1, resolution)
    x_grid, z_grid = np.meshgrid(x, z)
    x_test = np.reshape(x_grid, (-1, 1))
    z_test = np.reshape(z_grid, (-1, 1))
    y_test = np.ones_like(x_test) * (test_y - 500) / 300.0
    theta_test = np.ones_like(y_test) * test_theta
    
    x_test = tf.cast(x_test, dtype=data_type)
    y_test = tf.cast(y_test, dtype=data_type)
    z_test = tf.cast(z_test, dtype=data_type)
    theta_test = tf.cast(theta_test, dtype=data_type)
    
    loss_rans, loss_div = compute_PDE_residues(x_test, y_test, z_test, theta_test)
    
    loss_rans = tf.reshape(loss_rans, x_grid.shape)
    loss_div = tf.reshape(loss_div, x_grid.shape)
    
    # plot xz slice for rans
    fig, ax = plt.subplots(figsize=[8,8])
    im = ax.contourf(x_grid, z_grid, loss_rans, extend='both', cmap=plt.cm.RdBu_r, levels=np.linspace(0,20,1000))
    cbar_0 = fig.colorbar(im, ax=ax, orientation="horizontal", ticks=np.linspace(0,20,10, endpoint=True))
    plt.xlabel("x")
    plt.ylabel("z")
    ax.set_title(f'RANS residue on the xz plane, theta={theta}')
    plt.tight_layout(pad=2)
    plt.show()
    
    # plot xz slice for div
    fig, ax = plt.subplots(figsize=[8,8])
    im = ax.contourf(x_grid, z_grid, loss_div, extend='both', cmap=plt.cm.RdBu_r, levels=np.linspace(-10,10,1000))
    cbar_0 = fig.colorbar(im, ax=ax, orientation="horizontal", ticks=np.linspace(-10,10,10, endpoint=True)) 
    plt.xlabel("x")
    plt.ylabel("z")
    ax.set_title(f'div residue on the xz plane, theta={theta}')
    plt.tight_layout(pad=2)
    plt.show()
    
    # plot xz slice for u norm
    test_data = tf.concat([x_test, y_test, z_test, theta_test], axis=1)
    test_data = tf.cast(test_data, dtype=data_type)
    ux_pred = ux_model(test_data) * U
    uy_pred = uy_model(test_data) * U
    uz_pred = uz_model(test_data) * U
    u_norm = np.linalg.norm(tf.concat([ux_pred, uy_pred, uz_pred], axis=1), axis=1)
    u_norm = tf.reshape(u_norm, x_grid.shape)
    
    fig, ax = plt.subplots(figsize=[8,8])
    im = ax.contourf(x_grid, z_grid, u_norm, extend='both', cmap=plt.cm.RdBu_r, levels=np.linspace(0,5,1000))
    cbar_0 = fig.colorbar(im, ax=ax, orientation="horizontal", ticks=np.linspace(0,5,4, endpoint=True))
    
    # plot wind direction
#     wind_dir = np.array([np.cos(test_theta), np.sin(test_theta)])
#     ax.quiver(0, 0.7, wind_dir[0], wind_dir[1], color="k", scale=10, width=0.001, headwidth=4, headlength=4, headaxislength=3)
    
    plt.xlabel("x")
    plt.ylabel("z")
    ax.set_title(f'|u| on the xz plane, theta={theta}')
    plt.tight_layout(pad=2)
    plt.show()

# ---------------------- #

optimizer = tf.keras.optimizers.Adam(2e-5)

@tf.function
def train_step(x_, y_, z_, theta_, train_data_cfd, train_label_cfd):
    
    with tf.GradientTape(persistent=True) as tape2:
        
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(x_)
            tape1.watch(y_)
            tape1.watch(z_)
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(x_)
                tape.watch(y_)
                tape.watch(z_)
                
                # for NN inputs
                points_ = tf.concat([x_, y_, z_, theta_], axis=1)
                
                # forward pass
                ux = ux_model(points_)
                uy = uy_model(points_)
                uz = uz_model(points_)
                p = NN_p(points_)
                nut = NN_nut(points_)
                
            dux_dx = tape.gradient(ux, x_)
            dux_dy = tape.gradient(ux, y_)
            dux_dz = tape.gradient(ux, z_)
            
            duy_dx = tape.gradient(uy, x_)
            duy_dy = tape.gradient(uy, y_)
            duy_dz = tape.gradient(uy, z_)
            
            duz_dx = tape.gradient(uz, x_)
            duz_dy = tape.gradient(uz, y_)
            duz_dz = tape.gradient(uz, z_)
            
            dp_dx = tape.gradient(p, x_)
            dp_dy = tape.gradient(p, y_)
            dp_dz = tape.gradient(p, z_)
            
            dnut_dx = tape.gradient(nut, x_)
            dnut_dy = tape.gradient(nut, y_)
            dnut_dz = tape.gradient(nut, z_)
        
        d2ux_dx2 = tape1.gradient(dux_dx, x_)
        d2ux_dy2 = tape1.gradient(dux_dy, y_)
        d2ux_dz2 = tape1.gradient(dux_dz, z_)
        
        d2uy_dx2 = tape1.gradient(duy_dx, x_)
        d2uy_dy2 = tape1.gradient(duy_dy, y_)
        d2uy_dz2 = tape1.gradient(duy_dz, z_)
        
        d2uz_dx2 = tape1.gradient(duz_dx, x_)
        d2uz_dy2 = tape1.gradient(duz_dy, y_)
        d2uz_dz2 = tape1.gradient(duz_dz, z_)
            
        # term 1
        term_1_x = dux_dx * ux + dux_dy * uy + dux_dz * uz
        term_1_y = duy_dx * ux + duy_dy * uy + duy_dz * uz
        term_1_z = duz_dx * ux + duz_dy * uy + duz_dz * uz
        
        # term 2
        term_2_x = -dp_dx
        term_2_y = -dp_dy
        term_2_z = -dp_dz
        
        # term 3
        term_3_x = nut * (d2ux_dx2 + d2ux_dy2 + d2ux_dz2)
        term_3_y = nut * (d2uy_dx2 + d2uy_dy2 + d2uy_dz2)
        term_3_z = nut * (d2uz_dx2 + d2uz_dy2 + d2uz_dz2)

        # term 4
        term_4_x = dnut_dx * (dux_dx + dux_dx) + dnut_dy * (dux_dy + duy_dx) + dnut_dz * (dux_dz + duz_dx)
        term_4_y = dnut_dx * (duy_dx + dux_dy) + dnut_dy * (duy_dy + duy_dy) + dnut_dz * (duy_dz + duz_dy)
        term_4_z = dnut_dx * (duz_dx + dux_dz) + dnut_dy * (duz_dy + duy_dz) + dnut_dz * (duz_dz + duz_dz)
        
        # RANS
        residue_rans_x = term_1_x - term_2_x - term_3_x - term_4_x
        residue_rans_y = term_1_y - term_2_y - term_3_y - term_4_y
        residue_rans_z = term_1_z - term_2_z - term_3_z - term_4_z
        loss_rans = tf.math.reduce_mean(tf.math.square(residue_rans_x) + tf.math.square(residue_rans_y) + tf.math.square(residue_rans_z))

        # div
        div = dux_dx + duy_dy + duz_dz
        loss_div = tf.math.reduce_mean(tf.math.square(div))
        
        # fitting loss
        ux_1 = ux_model(train_data_cfd)
        uy_1 = uy_model(train_data_cfd)
        uz_1 = uz_model(train_data_cfd)
        p_1 = NN_p(train_data_cfd)
        nut_1 = NN_nut(train_data_cfd)

        flow_attribute_pred = tf.concat([ux_1, uy_1, uz_1, p_1, nut_1], axis=1)
        loss_fit = tf.math.reduce_mean(tf.math.square(flow_attribute_pred - train_label_cfd)) * 5
        
#         loss_fit = tf.math.reduce_mean(
#                                        tf.math.square(ux_1 - tf.reshape(train_label_cfd[:, 0], [-1, 1]))
#                                      + tf.math.square(uy_1 - tf.reshape(train_label_cfd[:, 1], [-1, 1]))
#                                      + tf.math.square(uz_1 - tf.reshape(train_label_cfd[:, 2], [-1, 1]))
#                                      + tf.math.square(p_1 - tf.reshape(train_label_cfd[:, 3], [-1, 1]))
#                                      + tf.math.square(nut_1 - tf.reshape(train_label_cfd[:, 4], [-1, 1]))
#         )

        
        # total loss
        # times 0.1 for loss ralated to equations
        loss = loss_fit + (loss_rans + loss_div) * 1e-3
        
    # apply gradient
    grad_ux = tape2.gradient(loss, NN_ux.trainable_variables)
    grad_uy = tape2.gradient(loss, NN_uy.trainable_variables)
    grad_uz = tape2.gradient(loss, NN_uz.trainable_variables)
    grad_p = tape2.gradient(loss, NN_p.trainable_variables)
    grad_nut = tape2.gradient(loss, NN_nut.trainable_variables)
    
    optimizer.apply_gradients(zip(grad_ux, NN_ux.trainable_variables))
    optimizer.apply_gradients(zip(grad_uy, NN_uy.trainable_variables))
    optimizer.apply_gradients(zip(grad_uz, NN_uz.trainable_variables))
    optimizer.apply_gradients(zip(grad_p, NN_p.trainable_variables))
    optimizer.apply_gradients(zip(grad_nut, NN_nut.trainable_variables))

    del tape
    del tape1
    del tape2
    
    return loss, loss_fit

loss_total_list = []
loss_fit_list = []


num_epochs = 10

start = time.time()
for i in range(num_epochs):
    
    loss_batch_list = []
    loss_fit_part_list = []
    for PDE_pts, (CFD_pts, CFD_labels) in zip(train_dataset_PDE, train_dataset_CFD):
        x_train = tf.reshape(PDE_pts[:, 0], (-1, 1))
        y_train = tf.reshape(PDE_pts[:, 1], (-1, 1))
        z_train = tf.reshape(PDE_pts[:, 2], (-1, 1))
        theta_train = tf.reshape(PDE_pts[:, 3], (-1, 1))
        loss, loss_fit_part = train_step(x_train, y_train, z_train, theta_train, CFD_pts, CFD_labels)
        loss_batch_list.append(loss)
        loss_fit_part_list.append(loss_fit_part)
    loss_epoch = tf.math.reduce_mean(loss_batch_list)
    loss_fit_part_epoch = tf.math.reduce_mean(loss_fit_part_list)
    
    print("epoch: ", i, "loss: ", loss_epoch.numpy(), "loss fit: ", loss_fit_part_epoch.numpy())
    
    loss_total_list.append(loss_epoch)
    loss_fit_list.append(loss_fit_part_epoch)
    

print(f"time taken: {(time.time() - start)/60} minutes")

# ---------------------- #

start = time.time()
for i in range(10):
    
    loss_batch_list = []
    loss_fit_part_list = []
    for PDE_pts, (CFD_pts, CFD_labels) in zip(train_dataset_PDE, train_dataset_CFD):
        x_train = tf.reshape(PDE_pts[:, 0], (-1, 1))
        y_train = tf.reshape(PDE_pts[:, 1], (-1, 1))
        z_train = tf.reshape(PDE_pts[:, 2], (-1, 1))
        theta_train = tf.reshape(PDE_pts[:, 3], (-1, 1))
        loss, loss_fit_part = train_step(x_train, y_train, z_train, theta_train, CFD_pts, CFD_labels)
        loss_batch_list.append(loss)
        loss_fit_part_list.append(loss_fit_part)
    loss_epoch = tf.math.reduce_mean(loss_batch_list)
    loss_fit_part_epoch = tf.math.reduce_mean(loss_fit_part_list)
    
    print("epoch: ", i+10, "loss: ", loss_epoch.numpy(), "loss fit: ", loss_fit_part_epoch.numpy())
    
    loss_total_list.append(loss_epoch)
    loss_fit_list.append(loss_fit_part_epoch)
    

print(f"time taken: {(time.time() - start)/60} minutes")

# ---------------------- #

start = time.time()
for i in range(10):
    
    loss_batch_list = []
    loss_fit_part_list = []
    for PDE_pts, (CFD_pts, CFD_labels) in zip(train_dataset_PDE, train_dataset_CFD):
        x_train = tf.reshape(PDE_pts[:, 0], (-1, 1))
        y_train = tf.reshape(PDE_pts[:, 1], (-1, 1))
        z_train = tf.reshape(PDE_pts[:, 2], (-1, 1))
        theta_train = tf.reshape(PDE_pts[:, 3], (-1, 1))
        loss, loss_fit_part = train_step(x_train, y_train, z_train, theta_train, CFD_pts, CFD_labels)
        loss_batch_list.append(loss)
        loss_fit_part_list.append(loss_fit_part)
    loss_epoch = tf.math.reduce_mean(loss_batch_list)
    loss_fit_part_epoch = tf.math.reduce_mean(loss_fit_part_list)
    
    print("epoch: ", i+20, "loss: ", loss_epoch.numpy(), "loss fit: ", loss_fit_part_epoch.numpy())
    
    loss_total_list.append(loss_epoch)
    loss_fit_list.append(loss_fit_part_epoch)
    

print(f"time taken: {(time.time() - start)/60} minutes")

# ---------------------- #

start = time.time()
for i in range(10):
    
    loss_batch_list = []
    loss_fit_part_list = []
    for PDE_pts, (CFD_pts, CFD_labels) in zip(train_dataset_PDE, train_dataset_CFD):
        x_train = tf.reshape(PDE_pts[:, 0], (-1, 1))
        y_train = tf.reshape(PDE_pts[:, 1], (-1, 1))
        z_train = tf.reshape(PDE_pts[:, 2], (-1, 1))
        theta_train = tf.reshape(PDE_pts[:, 3], (-1, 1))
        loss, loss_fit_part = train_step(x_train, y_train, z_train, theta_train, CFD_pts, CFD_labels)
        loss_batch_list.append(loss)
        loss_fit_part_list.append(loss_fit_part)
    loss_epoch = tf.math.reduce_mean(loss_batch_list)
    loss_fit_part_epoch = tf.math.reduce_mean(loss_fit_part_list)
    
    print("epoch: ", i+30, "loss: ", loss_epoch.numpy(), "loss fit: ", loss_fit_part_epoch.numpy())
    
    loss_total_list.append(loss_epoch)
    loss_fit_list.append(loss_fit_part_epoch)
    

print(f"time taken: {(time.time() - start)/60} minutes")

# ---------------------- #

optimizer = tf.keras.optimizers.Adam(1e-5)

start = time.time()
for i in range(10):
    
    loss_batch_list = []
    loss_fit_part_list = []
    for PDE_pts, (CFD_pts, CFD_labels) in zip(train_dataset_PDE, train_dataset_CFD):
        x_train = tf.reshape(PDE_pts[:, 0], (-1, 1))
        y_train = tf.reshape(PDE_pts[:, 1], (-1, 1))
        z_train = tf.reshape(PDE_pts[:, 2], (-1, 1))
        theta_train = tf.reshape(PDE_pts[:, 3], (-1, 1))
        loss, loss_fit_part = train_step(x_train, y_train, z_train, theta_train, CFD_pts, CFD_labels)
        loss_batch_list.append(loss)
        loss_fit_part_list.append(loss_fit_part)
    loss_epoch = tf.math.reduce_mean(loss_batch_list)
    loss_fit_part_epoch = tf.math.reduce_mean(loss_fit_part_list)
    
    print("epoch: ", i+40, "loss: ", loss_epoch.numpy(), "loss fit: ", loss_fit_part_epoch.numpy())
    
    loss_total_list.append(loss_epoch)
    loss_fit_list.append(loss_fit_part_epoch)
    

print(f"time taken: {(time.time() - start)/60} minutes")

# ---------------------- #


start = time.time()
for i in range(10):
    
    loss_batch_list = []
    loss_fit_part_list = []
    for PDE_pts, (CFD_pts, CFD_labels) in zip(train_dataset_PDE, train_dataset_CFD):
        x_train = tf.reshape(PDE_pts[:, 0], (-1, 1))
        y_train = tf.reshape(PDE_pts[:, 1], (-1, 1))
        z_train = tf.reshape(PDE_pts[:, 2], (-1, 1))
        theta_train = tf.reshape(PDE_pts[:, 3], (-1, 1))
        loss, loss_fit_part = train_step(x_train, y_train, z_train, theta_train, CFD_pts, CFD_labels)
        loss_batch_list.append(loss)
        loss_fit_part_list.append(loss_fit_part)
    loss_epoch = tf.math.reduce_mean(loss_batch_list)
    loss_fit_part_epoch = tf.math.reduce_mean(loss_fit_part_list)
    
    print("epoch: ", i+50, "loss: ", loss_epoch.numpy(), "loss fit: ", loss_fit_part_epoch.numpy())
    
    loss_total_list.append(loss_epoch)
    loss_fit_list.append(loss_fit_part_epoch)
    

print(f"time taken: {(time.time() - start)/60} minutes")

# ---------------------- #

plt.figure()
plt.plot(loss_total_list)
plt.ylabel("loss")
plt.show()

plt.figure()
plt.plot(loss_fit_list)
plt.ylabel("loss fit")
plt.show()

# ---------------------- #

loss_total_list[-1], loss_fit_list[-1]

# ---------------------- #

NN_ux.save("./after_RANS_div_refinement_with_CFD_ux", overwrite=True)
NN_uy.save("./after_RANS_div_refinement_with_CFD_uy", overwrite=True)
NN_uz.save("./after_RANS_div_refinement_with_CFD_uz", overwrite=True)
NN_p.save("./after_RANS_div_refinement_with_CFD_p", overwrite=True)
NN_nut.save("./after_RANS_div_refinement_with_CFD_nut", overwrite=True)

# ---------------------- #

# check after training

# sample on uniform grid and plot the results
# test on xy plane
resolution = 256
test_height = 50  # unit: meter

theta_range = np.arange(270, 90-15, -15)

for theta in theta_range:
    test_theta = np.deg2rad(theta)
    
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    x_grid, y_grid = np.meshgrid(x, y)
    x_test = np.reshape(x_grid, (-1, 1))
    y_test = np.reshape(y_grid, (-1, 1))
    z_test = np.ones_like(x_test) * test_height / 300.0
    theta_test = np.ones_like(x_test) * test_theta
    
    x_test = tf.cast(x_test, dtype=data_type)
    y_test = tf.cast(y_test, dtype=data_type)
    z_test = tf.cast(z_test, dtype=data_type)
    theta_test = tf.cast(theta_test, dtype=data_type)
    
    loss_rans, loss_div = compute_PDE_residues(x_test, y_test, z_test, theta_test)
    
    loss_rans = tf.reshape(loss_rans, x_grid.shape)
    loss_div = tf.reshape(loss_div, x_grid.shape)
    
    # plot xy slice for rans
    fig, ax = plt.subplots(figsize=[8,8])
    im = ax.contourf(x_grid, y_grid, loss_rans, extend='both', cmap=plt.cm.RdBu_r, levels=np.linspace(0,20,1000))
    cbar_0 = fig.colorbar(im, ax=ax, orientation="horizontal", ticks=np.linspace(0,20,10, endpoint=True))
    plt.xlabel("x")
    plt.ylabel("y")
    ax.set_title(f'RANS residue on the xy plane, theta={theta}')
    plt.tight_layout(pad=2)
    plt.show()
    
    # plot xy slice for div
    fig, ax = plt.subplots(figsize=[8,8])
    im = ax.contourf(x_grid, y_grid, loss_div, extend='both', cmap=plt.cm.RdBu_r, levels=np.linspace(-10,10,1000))
    cbar_0 = fig.colorbar(im, ax=ax, orientation="horizontal", ticks=np.linspace(-10,10,10, endpoint=True))
    plt.xlabel("x")
    plt.ylabel("y")
    ax.set_title(f'div residue on the xy plane, theta={theta}')
    # plt.gca().invert_xaxis()
    plt.tight_layout(pad=2)
    plt.show()
    
    # plot xy slice for u norm
    test_data = tf.concat([x_test, y_test, z_test, theta_test], axis=1)
    test_data = tf.cast(test_data, dtype=data_type)
    ux_pred = ux_model(test_data) * U
    uy_pred = uy_model(test_data) * U
    uz_pred = uz_model(test_data) * U
    u_norm = np.linalg.norm(tf.concat([ux_pred, uy_pred, uz_pred], axis=1), axis=1)
    u_norm = tf.reshape(u_norm, x_grid.shape)
    
    fig, ax = plt.subplots(figsize=[8,8])
    im = ax.contourf(x_grid, y_grid, u_norm, extend='both', cmap=plt.cm.RdBu_r, levels=np.linspace(0,5,1000))
    cbar_0 = fig.colorbar(im, ax=ax, orientation="horizontal", ticks=np.linspace(0,5,4, endpoint=True))
    
    # plot wind direction
    wind_dir = np.array([np.cos(test_theta), np.sin(test_theta)])
    ax.quiver(0, 0.7, wind_dir[0], wind_dir[1], color="k", scale=10, width=0.001, headwidth=4, headlength=4, headaxislength=3)
    
    plt.xlabel("x")
    plt.ylabel("y")
    ax.set_title(f'|u| on the xy plane, theta={theta}')
    plt.tight_layout(pad=2)
    plt.show()
    

# ---------------------- #

# test on yz plane
U = 4.043564066
resolution = 256
test_x = 500  # unit: meter

theta_range = np.arange(270, 90-15, -15)

for theta in theta_range:
    test_theta = np.deg2rad(theta)
    
    y = np.linspace(-1, 1, resolution)
    z = np.linspace(0, 1, resolution)
    y_grid, z_grid = np.meshgrid(y, z)
    y_test = np.reshape(y_grid, (-1, 1))
    z_test = np.reshape(z_grid, (-1, 1))
    x_test = np.ones_like(y_test) * (test_x - 500) / 300.0
    theta_test = np.ones_like(y_test) * test_theta
    
    x_test = tf.cast(x_test, dtype=data_type)
    y_test = tf.cast(y_test, dtype=data_type)
    z_test = tf.cast(z_test, dtype=data_type)
    theta_test = tf.cast(theta_test, dtype=data_type)
    
    loss_rans, loss_div = compute_PDE_residues(x_test, y_test, z_test, theta_test)
    
    loss_rans = tf.reshape(loss_rans, y_grid.shape)
    loss_div = tf.reshape(loss_div, y_grid.shape)
    
    # plot xy slice for rans
    fig, ax = plt.subplots(figsize=[8,8])
    im = ax.contourf(y_grid, z_grid, loss_rans, extend='both', cmap=plt.cm.RdBu_r, levels=np.linspace(0,20,1000))
    cbar_0 = fig.colorbar(im, ax=ax, orientation="horizontal", ticks=np.linspace(0,20,10, endpoint=True))
    plt.xlabel("y")
    plt.ylabel("z")
    ax.set_title(f'RANS residue on the yz plane, theta={theta}')
    plt.tight_layout(pad=2)
    plt.show()
    
    # plot xy slice for div
    fig, ax = plt.subplots(figsize=[8,8])
    im = ax.contourf(y_grid, z_grid, loss_div, extend='both', cmap=plt.cm.RdBu_r, levels=np.linspace(-10,10,1000))
    cbar_0 = fig.colorbar(im, ax=ax, orientation="horizontal", ticks=np.linspace(-10,10,10, endpoint=True)) 
    plt.xlabel("y")
    plt.ylabel("z")
    ax.set_title(f'div residue on the yz plane, theta={theta}')
    plt.tight_layout(pad=2)
    plt.show()
    
    # plot xy slice for u norm
    test_data = tf.concat([x_test, y_test, z_test, theta_test], axis=1)
    test_data = tf.cast(test_data, dtype=data_type)
    ux_pred = ux_model(test_data) * U
    uy_pred = uy_model(test_data) * U
    uz_pred = uz_model(test_data) * U
    u_norm = np.linalg.norm(tf.concat([ux_pred, uy_pred, uz_pred], axis=1), axis=1)
    u_norm = tf.reshape(u_norm, y_grid.shape)
    
    fig, ax = plt.subplots(figsize=[8,8])
    im = ax.contourf(y_grid, z_grid, u_norm, extend='both', cmap=plt.cm.RdBu_r, levels=np.linspace(0,5,1000))
    cbar_0 = fig.colorbar(im, ax=ax, orientation="horizontal", ticks=np.linspace(0,5,4, endpoint=True))
    
    # plot wind direction
#     wind_dir = np.array([np.cos(test_theta), np.sin(test_theta)])
#     ax.quiver(0, 0.7, wind_dir[0], wind_dir[1], color="k", scale=10, width=0.001, headwidth=4, headlength=4, headaxislength=3)
    
    plt.xlabel("y")
    plt.ylabel("z")
    ax.set_title(f'|u| on the yz plane, theta={theta}')
    plt.tight_layout(pad=2)
    plt.show()

# ---------------------- #

# test on xz plane
U = 4.043564066
resolution = 256
test_y = 500  # unit: meter

theta_range = np.arange(270, 90-15, -15)

for theta in theta_range:
    test_theta = np.deg2rad(theta)
    
    x = np.linspace(-1, 1, resolution)
    z = np.linspace(0, 1, resolution)
    x_grid, z_grid = np.meshgrid(x, z)
    x_test = np.reshape(x_grid, (-1, 1))
    z_test = np.reshape(z_grid, (-1, 1))
    y_test = np.ones_like(x_test) * (test_y - 500) / 300.0
    theta_test = np.ones_like(y_test) * test_theta
    
    x_test = tf.cast(x_test, dtype=data_type)
    y_test = tf.cast(y_test, dtype=data_type)
    z_test = tf.cast(z_test, dtype=data_type)
    theta_test = tf.cast(theta_test, dtype=data_type)
    
    loss_rans, loss_div = compute_PDE_residues(x_test, y_test, z_test, theta_test)
    
    loss_rans = tf.reshape(loss_rans, x_grid.shape)
    loss_div = tf.reshape(loss_div, x_grid.shape)
    
    # plot xz slice for rans
    fig, ax = plt.subplots(figsize=[8,8])
    im = ax.contourf(x_grid, z_grid, loss_rans, extend='both', cmap=plt.cm.RdBu_r, levels=np.linspace(0,20,1000))
    cbar_0 = fig.colorbar(im, ax=ax, orientation="horizontal", ticks=np.linspace(0,20,10, endpoint=True))
    plt.xlabel("x")
    plt.ylabel("z")
    ax.set_title(f'RANS residue on the xz plane, theta={theta}')
    plt.tight_layout(pad=2)
    plt.show()
    
    # plot xz slice for div
    fig, ax = plt.subplots(figsize=[8,8])
    im = ax.contourf(x_grid, z_grid, loss_div, extend='both', cmap=plt.cm.RdBu_r, levels=np.linspace(-10,10,1000))
    cbar_0 = fig.colorbar(im, ax=ax, orientation="horizontal", ticks=np.linspace(-10,10,10, endpoint=True)) 
    plt.xlabel("x")
    plt.ylabel("z")
    ax.set_title(f'div residue on the xz plane, theta={theta}')
    plt.tight_layout(pad=2)
    plt.show()
    
    # plot xz slice for u norm
    test_data = tf.concat([x_test, y_test, z_test, theta_test], axis=1)
    test_data = tf.cast(test_data, dtype=data_type)
    ux_pred = ux_model(test_data) * U
    uy_pred = uy_model(test_data) * U
    uz_pred = uz_model(test_data) * U
    u_norm = np.linalg.norm(tf.concat([ux_pred, uy_pred, uz_pred], axis=1), axis=1)
    u_norm = tf.reshape(u_norm, x_grid.shape)
    
    fig, ax = plt.subplots(figsize=[8,8])
    im = ax.contourf(x_grid, z_grid, u_norm, extend='both', cmap=plt.cm.RdBu_r, levels=np.linspace(0,5,1000))
    cbar_0 = fig.colorbar(im, ax=ax, orientation="horizontal", ticks=np.linspace(0,5,4, endpoint=True))
    
    # plot wind direction
#     wind_dir = np.array([np.cos(test_theta), np.sin(test_theta)])
#     ax.quiver(0, 0.7, wind_dir[0], wind_dir[1], color="k", scale=10, width=0.001, headwidth=4, headlength=4, headaxislength=3)
    
    plt.xlabel("x")
    plt.ylabel("z")
    ax.set_title(f'|u| on the xz plane, theta={theta}')
    plt.tight_layout(pad=2)
    plt.show()

# ---------------------- #

