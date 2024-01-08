# Comments on this version

# no slip boundary condition is satisfied by construction at both z=0 and z=200
# the mean flow takes the for U(z) = U_ref * ln[(z+z_0)/z_0] / ln((z_ref + z_0)/z_0) st U(0) = 0
# not scaled

# sigma does vanish at the walls
# sigma is now a function of u,v,w

import tensorflow as tf

import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

import time
import itertools

print('GPUs available on the machine:')
# !nvidia-smi -L
print()

gpus = tf.config.list_physical_devices('GPU')
device = gpus[0]
tf.config.set_visible_devices(device, 'GPU')

print('Simulation is running on:')
print(f'device = {device}')
print()

# set data precision
data_type    = tf.float64
data_type_nn = tf.float64
tf.keras.backend.set_floatx('float64')

# specify the number of dataset to be loaded
num_dataset = 1

# scale horizontal directions x,y by L and scale z by h
L_x_min = tf.constant(400, dtype=data_type)
L_x_max = tf.constant(600, dtype=data_type)

L_y_min = tf.constant(400, dtype=data_type)
L_y_max = tf.constant(600, dtype=data_type)

L_z = tf.constant(100, dtype=data_type)

# normalization const
L = tf.constant(100, dtype=data_type)
h = tf.constant(100, dtype=data_type)

# scale separation
eta = h / L

# U, theta inferred from meteorological data
meteo_data  = pd.read_csv(f'Sphere\\meteo.csv')
U_dict      = dict()
Z_dict      = dict()
varphi_dict = dict()
mesh_internal_dict = dict()

# create a dictionary to contain dataset
u_ref_internal_dict     = dict()
p_ref_internal_dict     = dict()
omega_ref_internal_dict = dict()
nut_ref_internal_dict   = dict()

with tf.device('cpu'):
    for i in range(1, num_dataset+1):

        U_dict[i] = tf.cast(meteo_data['Magnitude'][i-1], dtype=data_type)
        Z_dict[i] = tf.cast(meteo_data['Height'][i-1], dtype=data_type)
        varphi_dict[i] = tf.cast(meteo_data['Orientation'][i-1], dtype=data_type)

        data_internal_i = pd.read_csv(f'Sphere\\velocity_internal_small_{i}.csv')

        u_x_i = tf.reshape(tf.cast(data_internal_i['Velocity:0'], dtype=data_type), [-1,1])
        u_y_i = tf.reshape(tf.cast(data_internal_i['Velocity:1'], dtype=data_type), [-1,1])
        u_z_i = tf.reshape(tf.cast(data_internal_i['Velocity:2'], dtype=data_type), [-1,1])

        u_ref_internal_dict[i] = tf.concat([u_x_i, u_y_i, u_z_i], axis=1) 
        
        p_i = tf.reshape(tf.cast(data_internal_i['Pressure'], dtype=data_type), [-1,1])
        p_ref_internal_dict[i] = p_i
        
        nut_i = tf.reshape(tf.cast(data_internal_i['TurbVisc'], dtype=data_type), [-1,1])
        nut_ref_internal_dict[i] = nut_i
        
        # unnormalized vorticity, otherwise it is too large
        omega_x_i = tf.reshape(tf.cast(data_internal_i['Vorticity:0'], dtype=data_type), [-1,1])
        omega_y_i = tf.reshape(tf.cast(data_internal_i['Vorticity:1'], dtype=data_type), [-1,1])
        omega_z_i = tf.reshape(tf.cast(data_internal_i['Vorticity:2'], dtype=data_type), [-1,1])
        
        omega_ref_internal_dict[i] = tf.concat([omega_x_i, omega_y_i, omega_z_i], axis=1) 
        
        x_internal_i = tf.reshape(tf.cast(data_internal_i['Points:0'], dtype=data_type), [-1,1])
        y_internal_i = tf.reshape(tf.cast(data_internal_i['Points:1'], dtype=data_type), [-1,1])
        z_internal_i = tf.reshape(tf.cast(data_internal_i['Points:2'], dtype=data_type), [-1,1])
        
        mesh_internal_dict[i] = tf.concat([x_internal_i, y_internal_i, z_internal_i], axis=1)     

with tf.device('cpu'):
    
    mesh_buildings = pd.read_csv('Sphere\\mesh_sphere.csv')

    x_buildings = tf.reshape(tf.cast( (mesh_buildings['Points:0']/1000 - 500) / L, dtype=data_type), [-1,1])
    y_buildings = tf.reshape(tf.cast( (mesh_buildings['Points:1']/1000 - 500) / L, dtype=data_type), [-1,1])
    z_buildings = tf.reshape(tf.cast( (mesh_buildings['Points:2']/1000 - 0  ) / h, dtype=data_type), [-1,1])
    
    mesh_buildings_surf = pd.read_csv('Sphere\\mesh_sphere_surf.csv')

    x_buildings_surf = tf.reshape(tf.cast( (mesh_buildings_surf['Points:0']/1000 - 500) / L, dtype=data_type), [-1,1])
    y_buildings_surf = tf.reshape(tf.cast( (mesh_buildings_surf['Points:1']/1000 - 500) / L, dtype=data_type), [-1,1])
    z_buildings_surf = tf.reshape(tf.cast( (mesh_buildings_surf['Points:2']/1000 - 0  ) / h, dtype=data_type), [-1,1])

    mesh_internal = mesh_internal_dict[1]

    x_internal = tf.reshape(-1 + 2 * (mesh_internal[:,0] - L_x_min) / (L_x_max - L_x_min), [-1,1])
    y_internal = tf.reshape(-1 + 2 * (mesh_internal[:,1] - L_y_min) / (L_y_max - L_y_min), [-1,1])
    z_internal = tf.reshape(mesh_internal[:,2] / h, [-1,1])
    
# Creating figure
fig  = plt.figure(figsize=[24,8], dpi = 450)

ax_1 = fig.add_subplot(131, projection='3d')
ax_1.scatter(x_buildings, y_buildings, z_buildings, s=0.05, alpha=1)
ax_1.set_title('buildings')

ax_2 = fig.add_subplot(132, projection='3d')
ax_2.scatter(x_buildings_surf, y_buildings_surf, z_buildings_surf, s=0.05, alpha=1)
ax_2.set_title('buildings_surf')

ax_3 = fig.add_subplot(133, projection='3d')
ax_3.scatter(x_internal, y_internal, z_internal, s=0.05, alpha=1)
ax_3.set_title('internal') 
    
plt.show()

import shutil

import logging, os 
logging.disable(logging.WARNING) 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def save_nn_velocity(NN_u_x_dict, NN_u_y_dict, NN_u_z_dict, NN_p_dict):
    
    for i in range(1, num_dataset+1):

        NN_u_x_i  = NN_u_x_dict[i]
        NN_u_y_i  = NN_u_y_dict[i]
        NN_u_z_i  = NN_u_z_dict[i]
        NN_p_i    = NN_p_dict[i]
        
        tf.keras.models.save_model(NN_u_x_i, 
                                   filepath = f'{folder_name_save}/NN_u_x_{i}',
                                   include_optimizer = False, 
                                   overwrite = True)
        
        tf.keras.models.save_model(NN_u_y_i, 
                                   filepath = f'{folder_name_save}/NN_u_y_{i}',
                                   include_optimizer = False, 
                                   overwrite = True)
        
        tf.keras.models.save_model(NN_u_z_i, 
                                   filepath = f'{folder_name_save}/NN_u_z_{i}',
                                   include_optimizer = False, 
                                   overwrite = True)
        
        tf.keras.models.save_model(NN_p_i, 
                                   filepath = f'{folder_name_save}/NN_p_{i}',
                                   include_optimizer = False, 
                                   overwrite = True)

        
def save_nn_sigma(NN_nut, NN_mask):
    
    tf.keras.models.save_model(NN_nut, 
                               filepath = f'{folder_name_save}/NN_nut',
                               include_optimizer = False, 
                               overwrite = True)
    
    tf.keras.models.save_model(NN_mask, 
                               filepath = f'{folder_name_save}/NN_mask',
                               include_optimizer = False, 
                               overwrite = True)
    
def load_nn_potential(folder_name_load):
    
    NN_u_x_dict  = dict()
    NN_u_y_dict  = dict()
    NN_u_z_dict  = dict()
    NN_p_dict    = dict()
    
    for i in range(1, num_dataset+1):
        NN_u_x_dict[i]  = tf.keras.models.load_model(f'{folder_name_load}/NN_u_x_{i}')
        NN_u_y_dict[i]  = tf.keras.models.load_model(f'{folder_name_load}/NN_u_y_{i}')
        NN_u_z_dict[i]  = tf.keras.models.load_model(f'{folder_name_load}/NN_u_z_{i}')
        NN_p_dict[i]    = tf.keras.models.load_model(f'{folder_name_load}/NN_p_{i}')
    
    return NN_u_x_dict, NN_u_y_dict, NN_u_z_dict, NN_p_dict

def load_nn_sigma(folder_name_load):
    
    NN_nut  = tf.keras.models.load_model(f'{folder_name_load}/NN_nut')
    NN_mask = tf.keras.models.load_model(f'{folder_name_load}/NN_mask')
    
    return NN_nut, NN_mask

folder_name_load = 'Surrogate_sphere_uvw_nut_ss_nut_last_trial'
folder_name_save = 'Surrogate_sphere_uvw_nut_ss_nut_last_trial'

def net_velocity():

    """
    map (x,y,z) to phi and psi: linear activation
    """

    input_   = tf.keras.Input(shape=(3,))
    
    dense_in = tf.keras.layers.Dense(64, activation='tanh', use_bias=False)(input_)

    ######### transnet unit ############################################################################################
    dense_11 = tf.keras.layers.Dense(64, activation='tanh', use_bias=True)(dense_in)
    dense_12 = tf.keras.layers.Dense(64, activation='tanh', use_bias=True)(dense_11)
    
    added_1  = tf.keras.layers.Add()([dense_in, dense_12])
    ####################################################################################################################
    
    ######### transnet unit ############################################################################################
    dense_21 = tf.keras.layers.Dense(64, activation='tanh', use_bias=True)(added_1)
    dense_22 = tf.keras.layers.Dense(64, activation='tanh', use_bias=True)(dense_21)
    
    added_2  = tf.keras.layers.Add()([added_1, dense_22])
    ####################################################################################################################
    
    ######### transnet unit ############################################################################################
    dense_31 = tf.keras.layers.Dense(64, activation='tanh', use_bias=True)(added_2)
    dense_32 = tf.keras.layers.Dense(64, activation='tanh', use_bias=True)(dense_31)
    
    added_3  = tf.keras.layers.Add()([added_2, dense_32])
    ####################################################################################################################
    
    ######### transnet unit ############################################################################################
    dense_41 = tf.keras.layers.Dense(64, activation='tanh', use_bias=True)(added_3)
    dense_42 = tf.keras.layers.Dense(64, activation='tanh', use_bias=True)(dense_41)
    
    added_4  = tf.keras.layers.Add()([added_3, dense_42])
    ####################################################################################################################
    
    ######### transnet unit ############################################################################################
    dense_51 = tf.keras.layers.Dense(64, activation='tanh', use_bias=True)(added_4)
    dense_52 = tf.keras.layers.Dense(64, activation='tanh', use_bias=True)(dense_51)
    
    added_5  = tf.keras.layers.Add()([added_4, dense_52])
    ####################################################################################################################
    
    dense_out = tf.keras.layers.Dense(64, activation="linear", use_bias=False)(added_5)
    norm_out  = tf.keras.layers.BatchNormalization()(dense_out)
    tanh_out  = tf.keras.layers.Activation('tanh')(norm_out)
    
    output_ = tf.keras.layers.Dense(1, activation = 'linear', use_bias=False, dtype=data_type_nn)(tanh_out)

    model   = tf.keras.models.Model(inputs=input_, outputs=output_)

    return model

def net_mask():

    """
    map (x,y,z) to phi and psi: linear activation
    """

    input_   = tf.keras.Input(shape=(3,))
    noisy_input_ = tf.keras.layers.GaussianNoise(0.1)(input_)
    
    dense_in = tf.keras.layers.Dense(64, activation='tanh', use_bias=False)(noisy_input_)

    ######### transnet unit ############################################################################################
    dense_11 = tf.keras.layers.Dense(64, activation='tanh', use_bias=True)(dense_in)
    dense_12 = tf.keras.layers.Dense(64, activation='tanh', use_bias=True)(dense_11)
    
    added_1  = tf.keras.layers.Add()([dense_in, dense_12])
    ####################################################################################################################
    
    ######### transnet unit ############################################################################################
    dense_21 = tf.keras.layers.Dense(64, activation='tanh', use_bias=True)(added_1)
    dense_22 = tf.keras.layers.Dense(64, activation='tanh', use_bias=True)(dense_21)
    
    added_2  = tf.keras.layers.Add()([added_1, dense_22])
    ####################################################################################################################
    
    ######### transnet unit ############################################################################################
    dense_31 = tf.keras.layers.Dense(64, activation='tanh', use_bias=True)(added_2)
    dense_32 = tf.keras.layers.Dense(64, activation='tanh', use_bias=True)(dense_31)
    
    added_3  = tf.keras.layers.Add()([added_2, dense_32])
    ####################################################################################################################
    
    ######### transnet unit ############################################################################################
    dense_41 = tf.keras.layers.Dense(64, activation='tanh', use_bias=True)(added_3)
    dense_42 = tf.keras.layers.Dense(64, activation='tanh', use_bias=True)(dense_41)
    
    added_4  = tf.keras.layers.Add()([added_3, dense_42])
    ####################################################################################################################
    
    ######### transnet unit ############################################################################################
    dense_51 = tf.keras.layers.Dense(64, activation='tanh', use_bias=True)(added_4)
    dense_52 = tf.keras.layers.Dense(64, activation='tanh', use_bias=True)(dense_51)
    
    added_5  = tf.keras.layers.Add()([added_4, dense_52])
    ####################################################################################################################
    
    dense_out = tf.keras.layers.Dense(64, activation="linear", use_bias=False)(added_5)
    norm_out  = tf.keras.layers.BatchNormalization()(dense_out)
    tanh_out  = tf.keras.layers.Activation('tanh')(norm_out)
    
    output_ = tf.keras.layers.Dense(1, activation = 'sigmoid', use_bias=True, dtype=data_type_nn)(tanh_out)

    model   = tf.keras.models.Model(inputs=input_, outputs=output_)

    return model

def net_nut():

    """
    map 3d images of shape (-1, 2N+1, 2N+1, 2N+1, 2) to a scalar
    """
    
    input_   = tf.keras.Input(shape=(3,))
    
    dense_in = tf.keras.layers.Dense(64, activation='tanh', use_bias=False)(input_)

    ######### transnet unit ############################################################################################
    dense_11 = tf.keras.layers.Dense(64, activation='tanh', use_bias=True)(dense_in)
    dense_12 = tf.keras.layers.Dense(64, activation='tanh', use_bias=True)(dense_11)
    
    added_1  = tf.keras.layers.Add()([dense_in, dense_12])
    ####################################################################################################################
    
    ######### transnet unit ############################################################################################
    dense_21 = tf.keras.layers.Dense(64, activation='tanh', use_bias=True)(added_1)
    dense_22 = tf.keras.layers.Dense(64, activation='tanh', use_bias=True)(dense_21)
    
    added_2  = tf.keras.layers.Add()([added_1, dense_22])
    ####################################################################################################################
    
    ######### transnet unit ############################################################################################
    dense_31 = tf.keras.layers.Dense(64, activation='tanh', use_bias=True)(added_2)
    dense_32 = tf.keras.layers.Dense(64, activation='tanh', use_bias=True)(dense_31)
    
    added_3  = tf.keras.layers.Add()([added_2, dense_32])
    ####################################################################################################################
    
    ######### transnet unit ############################################################################################
    dense_41 = tf.keras.layers.Dense(64, activation='tanh', use_bias=True)(added_3)
    dense_42 = tf.keras.layers.Dense(64, activation='tanh', use_bias=True)(dense_41)
    
    added_4  = tf.keras.layers.Add()([added_3, dense_42])
    ####################################################################################################################
    
    ######### transnet unit ############################################################################################
    dense_51 = tf.keras.layers.Dense(64, activation='tanh', use_bias=True)(added_4)
    dense_52 = tf.keras.layers.Dense(64, activation='tanh', use_bias=True)(dense_51)
    
    added_5  = tf.keras.layers.Add()([added_4, dense_52])
    ####################################################################################################################
    
    dense_out = tf.keras.layers.Dense(64, activation="linear", use_bias=False)(added_5)
    norm_out  = tf.keras.layers.BatchNormalization()(dense_out)
    tanh_out  = tf.keras.layers.Activation('tanh')(norm_out)
    
    output_ = tf.keras.layers.Dense(1, activation = 'softplus', use_bias=True, dtype=data_type_nn)(tanh_out)

    model   = tf.keras.models.Model(inputs=input_, outputs=output_)

    return model

# initialization #######################################################################################################

z_0_const = tf.constant(1, dtype=data_type)

NN_u_x_dict  = dict()
NN_u_y_dict  = dict()
NN_u_z_dict  = dict()
NN_p_dict    = dict()

# initialization 

# NN_u_x_dict, NN_u_y_dict, NN_u_z_dict, NN_p_dict = load_nn_potential(folder_name_load)
# NN_nut, NN_mask = load_nn_sigma(folder_name_load)

NN_nut = net_nut()
NN_mask = net_mask()

for i in range(1, num_dataset+1):
    NN_u_x_dict[i] = net_velocity()
    NN_u_y_dict[i] = net_velocity()
    NN_u_z_dict[i] = net_velocity()
    NN_p_dict[i]   = net_velocity()

NN_u_x_dict[1].summary()
print()
NN_nut.summary()

def neural_mask(x, y, z):
    
    x_ = tf.cast(tf.reshape(x, [-1,1]), dtype=data_type)
    y_ = tf.cast(tf.reshape(y, [-1,1]), dtype=data_type)
    z_ = tf.cast(tf.reshape(z, [-1,1]), dtype=data_type)

    mask_ = NN_mask(tf.concat([x_, y_, z_], axis=1))
    
    return mask_

def neural_p_tilde(x, y, z, index):
    
    x_ = tf.cast(tf.reshape(x, [-1,1]), dtype=data_type)
    y_ = tf.cast(tf.reshape(y, [-1,1]), dtype=data_type)
    z_ = tf.cast(tf.reshape(z, [-1,1]), dtype=data_type)

    p_ = NN_p_dict[index](tf.concat([x_, y_, z_], axis=1))
    
    return p_

def neural_u_x_tilde(x, y, z, index):
    
    x_ = tf.cast(tf.reshape(x, [-1,1]), dtype=data_type)
    y_ = tf.cast(tf.reshape(y, [-1,1]), dtype=data_type)
    z_ = tf.cast(tf.reshape(z, [-1,1]), dtype=data_type)
    
    g_ = ( tf.cos(np.pi * L * x_ / (L_x_max - L_x_min))**2 * 
           tf.cos(np.pi * L * y_ / (L_y_max - L_y_min))    * 
           tf.sin(np.pi * h * z_ / L_z)                    )

    u_nn_x_ = g_ * NN_u_x_dict[index](tf.concat([x_, y_, z_], axis=1))
    
    return u_nn_x_

def neural_u_y_tilde(x, y, z, index):
    
    x_ = tf.cast(tf.reshape(x, [-1,1]), dtype=data_type)
    y_ = tf.cast(tf.reshape(y, [-1,1]), dtype=data_type)
    z_ = tf.cast(tf.reshape(z, [-1,1]), dtype=data_type)
    
    g_ = ( tf.cos(np.pi * L * x_ / (L_x_max - L_x_min))    * 
           tf.cos(np.pi * L * y_ / (L_y_max - L_y_min))**2 * 
           tf.sin(np.pi * h * z_ / L_z)                    )
    
    u_nn_y_ = g_ * NN_u_y_dict[index](tf.concat([x_, y_, z_], axis=1))
    
    return u_nn_y_

def neural_u_z_tilde(x, y, z, index):
    
    x_ = tf.cast(tf.reshape(x, [-1,1]), dtype=data_type)
    y_ = tf.cast(tf.reshape(y, [-1,1]), dtype=data_type)
    z_ = tf.cast(tf.reshape(z, [-1,1]), dtype=data_type)
    
    g_ = ( tf.cos(np.pi * L * x_ / (L_x_max - L_x_min)) * 
           tf.cos(np.pi * L * y_ / (L_y_max - L_y_min)) * 
           tf.sin(np.pi * h * z_ / L_z)**2              )
    
    u_nn_z_ = g_ * NN_u_z_dict[index](tf.concat([x_, y_, z_], axis=1))
    
    return u_nn_z_

def neural_velocity_tilde(x, y, z, index):
    
    x_ = tf.cast(tf.reshape(x, [-1,1]), dtype=data_type)
    y_ = tf.cast(tf.reshape(y, [-1,1]), dtype=data_type)
    z_ = tf.cast(tf.reshape(z, [-1,1]), dtype=data_type)
    
    u_nn_x = neural_u_x_tilde(x_, y_, z_, index)
    u_nn_y = neural_u_y_tilde(x_, y_, z_, index)
    u_nn_z = neural_u_z_tilde(x_, y_, z_, index)
    
    u_nn = tf.concat([u_nn_x, u_nn_y, u_nn_z], axis=1)
    
    return u_nn

def neural_omega_x_tilde(x, y, z, index):
    
    x_ = tf.cast(tf.reshape(x, [-1,1]), dtype=data_type)
    y_ = tf.cast(tf.reshape(y, [-1,1]), dtype=data_type)
    z_ = tf.cast(tf.reshape(z, [-1,1]), dtype=data_type)
    
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(y_)
        tape.watch(z_)
        
        u_ = neural_velocity(x_, y_ ,z_, 1) 
    
        u_x_ = u_[:,0]
        u_y_ = u_[:,1]
        u_z_ = u_[:,2]
        
    omega_nn_x_ = tape.gradient(u_z_, y_) - tape.gradient(u_y_, z_)
    
    return omega_nn_x_

def neural_omega_y_tilde(x, y, z, index):
    
    x_ = tf.cast(tf.reshape(x, [-1,1]), dtype=data_type)
    y_ = tf.cast(tf.reshape(y, [-1,1]), dtype=data_type)
    z_ = tf.cast(tf.reshape(z, [-1,1]), dtype=data_type)
    
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x_)
        tape.watch(z_)
        
        u_ = neural_velocity(x_, y_ ,z_, 1) 
    
        u_x_ = u_[:,0]
        u_y_ = u_[:,1]
        u_z_ = u_[:,2]

    omega_nn_y_ = tape.gradient(u_x_, z_) - tape.gradient(u_z_, x_)
    
    return omega_nn_y_

def neural_omega_z_tilde(x, y, z, index):
    
    x_ = tf.cast(tf.reshape(x, [-1,1]), dtype=data_type)
    y_ = tf.cast(tf.reshape(y, [-1,1]), dtype=data_type)
    z_ = tf.cast(tf.reshape(z, [-1,1]), dtype=data_type)
    
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x_)
        tape.watch(y_)
        
        u_ = neural_velocity(x_, y_ ,z_, 1) 

        u_x_ = u_[:,0]
        u_y_ = u_[:,1]
        u_z_ = u_[:,2]
        
    omega_nn_z_ = tape.gradient(u_y_, x_) - tape.gradient(u_x_, y_)
    
    return omega_nn_z_

def neural_vorticity_tilde(x, y, z, index):
    
    x_ = tf.cast(tf.reshape(x, [-1,1]), dtype=data_type)
    y_ = tf.cast(tf.reshape(y, [-1,1]), dtype=data_type)
    z_ = tf.cast(tf.reshape(z, [-1,1]), dtype=data_type)
    
    omega_nn_x = neural_omega_x_tilde(x_, y_, z_, index)
    omega_nn_y = neural_omega_y_tilde(x_, y_, z_, index)
    omega_nn_z = neural_omega_z_tilde(x_, y_, z_, index)
    
    omega_nn = tf.concat([omega_nn_x, omega_nn_y, omega_nn_z], axis=1)
    
    return omega_nn

def velocity_mean(x, y, z, index): 
    
    x_ = tf.cast(tf.reshape(x, [-1,1]), dtype=data_type)
    y_ = tf.cast(tf.reshape(y, [-1,1]), dtype=data_type)
    z_ = tf.cast(tf.reshape(z, [-1,1]), dtype=data_type)

    # the roughness length of urban is z_0 ~ 1m
    z_0   = z_0_const / h
    z_ref = Z_dict[index] / h

    cos_ = tf.cast(tf.cos(varphi_dict[index]), dtype=data_type)
    sin_ = tf.cast(tf.sin(varphi_dict[index]), dtype=data_type)

    log_profile = tf.math.log(z_/z_0 + 1) / tf.math.log(z_ref/z_0 + 1)

    U_x = cos_ * log_profile + 0 * (x_**2 + y_**2 + z_**2)
    U_y = sin_ * log_profile + 0 * (x_**2 + y_**2 + z_**2)
    U_z = 0 * (x_**2 + y_**2 + z_**2)
   
    U = tf.cast(tf.concat([U_x, U_y, U_z], axis=1), dtype=data_type)
    
    return U

def neural_velocity(x, y, z, index):
    
    u_tilde = neural_velocity_tilde(x, y, z, index)
    U_mean  = velocity_mean(x, y, z, index)
    
    mask_ = tf.repeat(tf.stop_gradient(neural_mask(x, y, z)), 3, axis=1)
    
    u_ = mask_ * (u_tilde + U_mean)
    
    return u_

def neural_nut_an(x, y, z, index):
    
    """
    Re_t = eta * Uh/nu_t 
    """
    
    x_ = tf.cast(tf.reshape(x, [-1,1]), dtype=data_type)
    y_ = tf.cast(tf.reshape(y, [-1,1]), dtype=data_type)
    z_ = tf.cast(tf.reshape(z, [-1,1]), dtype=data_type)
    
    # the roughness length
    z_0   = z_0_const / h
    z_ref = Z_dict[index] / h
    kappa = tf.constant(0.41, dtype=data_type)
 
    nut_an = kappa**2 * U_dict[index] * h * (z_ + z_0) / tf.math.log(z_ref/z_0 + 1) + 0 * (x_**2 + y_**2 + z_**2)
    
    return nut_an

def Re_inv_cal(x, y, z, index):
    
    """
    Re_t = eta * Uh/nu_t 
    """
    
    x_ = tf.cast(tf.reshape(x, [-1,1]), dtype=data_type)
    y_ = tf.cast(tf.reshape(y, [-1,1]), dtype=data_type)
    z_ = tf.cast(tf.reshape(z, [-1,1]), dtype=data_type)
    
    z_0   = z_0_const / h
    z_ref = Z_dict[index] / h
    kappa = tf.constant(0.41, dtype=data_type)
    
    nut_0 = kappa**2 * (z_ref + z_0) / tf.math.log(z_ref/z_0 + 1)
    
    Re_inv = (1.48*10**(-5) / (eta * U_dict[index] * h)) + nut_0 * z_ * NN_nut(tf.concat([x_, y_, z_], axis=1))
    
    return Re_inv

def neural_nut(x, y, z, index):
    
    """
    Re_t = eta * Uh/nu_t 
    """
    
    x_ = tf.cast(tf.reshape(x, [-1,1]), dtype=data_type)
    y_ = tf.cast(tf.reshape(y, [-1,1]), dtype=data_type)
    z_ = tf.cast(tf.reshape(z, [-1,1]), dtype=data_type)
    
    nut_ = Re_inv_cal(x, y, z, index) * (eta * U_dict[index] * h)
    
    return nut_

## verify that the div u = 0 by construction

x_ = tf.cast(tf.reshape(np.random.uniform(-1, 1, 1024), [-1,1]), dtype=data_type)
y_ = tf.cast(tf.reshape(np.random.uniform(-1, 1, 1024), [-1,1]), dtype=data_type)
z_ = tf.cast(tf.reshape(np.random.uniform( 0, 1, 1024), [-1,1]), dtype=data_type)

with tf.GradientTape(persistent=True) as tape:
    tape.watch(x_)
    tape.watch(y_)
    tape.watch(z_)
    
    u_ = neural_velocity(x_, y_ ,z_, 1) 
    
    u_x_ = u_[:,0]
    u_y_ = u_[:,1]
    u_z_ = u_[:,2]
    
div_u = tf.reduce_mean(tape.gradient(u_x_, x_) + tape.gradient(u_y_, y_) + tape.gradient(u_z_, z_)) 

print(f'div u = {div_u}')

def plot_vert(y_loc, index):
    
    x_ = tf.cast(tf.reshape(np.full(256, 0.0), [-1,1]), dtype=data_type)
    y_ = tf.cast(tf.reshape(np.full(256, y_loc), [-1,1]), dtype=data_type)
    z_ = tf.cast(tf.reshape(np.linspace(0, 1, 256), [-1,1]), dtype=data_type)

    with tf.GradientTape(persistent=True) as tape:

        tape.watch(x_)
        tape.watch(y_)
        tape.watch(z_)

        u_bar = neural_velocity(x_, y_, z_, index)
        
        u_x = tf.reshape(u_bar[:,0], [-1,1])
        u_y = tf.reshape(u_bar[:,1], [-1,1])
        u_z = tf.reshape(u_bar[:,2], [-1,1])
        
        p_ = neural_p_tilde(x_, y_, z_, index)
        omega_z_ = neural_omega_z_tilde(x_, y_, z_, index)

    grad_u_x_x = tape.gradient(u_x, x_)
    grad_u_x_y = tape.gradient(u_x, y_)
    grad_u_x_z = tape.gradient(u_x, z_)

    grad_u_y_x = tape.gradient(u_y, x_)
    grad_u_y_y = tape.gradient(u_y, y_)
    grad_u_y_z = tape.gradient(u_y, z_)

    grad_u_z_x = tape.gradient(u_z, x_)
    grad_u_z_y = tape.gradient(u_z, y_)
    grad_u_z_z = tape.gradient(u_z, z_)

    u_del_u_x = u_x * grad_u_x_x + u_y * grad_u_x_y + u_z * grad_u_x_z
    u_del_u_y = u_x * grad_u_y_x + u_y * grad_u_y_y + u_z * grad_u_y_z
    u_del_u_z = u_x * grad_u_z_x + u_y * grad_u_z_y + u_z * grad_u_z_z
    
    grad_p_x = tape.gradient(p_, x_)
    grad_p_y = tape.gradient(p_, y_)
    grad_p_z = tape.gradient(p_, z_)
    
    grad_omega_x_ = tape.gradient(omega_z_, x_)
    grad_omega_y_ = tape.gradient(omega_z_, y_)
    grad_omega_z_ = tape.gradient(omega_z_, z_)
    
    U = velocity_mean(x_, y_, z_, index)
    u_nn = neural_velocity(x_, y_, z_, index)
    
    nut_   = neural_nut(x_, y_, z_, index)
    nut_an = neural_nut_an(x_, y_, z_, index)

    fig, ax = plt.subplots(1,5, figsize=[20,3], dpi = 250)
    
    ax[0].plot(nut_, z_, label = 'nu_t')
    ax[0].plot(nut_an, z_, label = 'nut_an')
    ax[0].legend(loc='upper right')

    ax[1].plot(grad_u_z_x, z_, label='grad_u_z_x')
    ax[1].plot(grad_u_z_y, z_, label='grad_u_z_y')
    ax[1].plot(grad_u_z_z, z_, label='grad_u_z_z')
    ax[1].legend(loc='upper right')
    
    ax[2].plot(1.23 * U_dict[index]**2 * p_, z_, label='p')
    ax[2].legend(loc='upper right')

    ax[3].plot(U[:,0], z_,  label='U_x')
    ax[3].plot(u_nn[:,0], z_,  label='u_fit_x')
    ax[3].plot(U[:,1], z_,  label='U_y')
    ax[3].plot(u_nn[:,1], z_,  label='u_fit_y')
    ax[3].legend(loc='upper right')
    
    ax[4].plot(neural_mask(x_, y_, z_), z_,  label='mask')
    ax[4].legend(loc='upper right')

    plt.tight_layout()
    plt.show()
    
plot_vert(0,    1)
plot_vert(0.05, 1)

def plt_velocity_xz(index):
    
    # compute neural velocities ######################################################################
    x_ = np.linspace(-1, 1, 256)
    z_ = np.linspace( 0, 1, 256)

    xx_, zz_ = np.meshgrid(x_, z_)
    yy_ = np.full(np.shape(xx_), 0)

    velocities_ = neural_velocity(xx_, yy_, zz_, index)

    u_x_ = U_dict[index] * tf.reshape(velocities_[:,0], np.shape(xx_))
    u_y_ = U_dict[index] * tf.reshape(velocities_[:,1], np.shape(yy_))
    u_z_ = U_dict[index] * tf.reshape(velocities_[:,2], np.shape(zz_))

    u_abs_ = tf.sqrt(u_x_**2 + u_y_**2 + u_z_**2)

    # find buildings ###################################################################################
    theta_ = tf.cast(tf.reshape(np.linspace(0, 2*np.pi, 128), [-1,1]), dtype=data_type)
    r_     = tf.constant(5, dtype=data_type) 

    x_sphere = (tf.cos(theta_) * r_) / L
    z_sphere = (50  + (tf.sin(theta_) * r_)) / h

    # plot results #####################################################################################

    fig, ax = plt.subplots(1,4, figsize=[16,4], dpi = 250)

    im_0 = ax[0].contourf(xx_, zz_, u_x_, extend='both', cmap = plt.cm.RdBu_r, levels = 128)
    cbar_0 = fig.colorbar(im_0, ax=ax[0], orientation="horizontal")
    cbar_0.ax.locator_params(nbins=5)
    ax[0].scatter(x_sphere, z_sphere, s=1, color="black")
    ax[0].set_xlabel = ('x')
    ax[0].set_ylabel = ('z')
    ax[0].set_title(f'u_x on the xz plane', y=1.05)

    im_1 = ax[1].contourf(xx_, zz_, u_y_, extend='both', cmap = plt.cm.RdBu_r, levels = 128)
    cbar_1 = fig.colorbar(im_1, ax=ax[1], orientation="horizontal")
    cbar_1.ax.locator_params(nbins=5)
    ax[1].scatter(x_sphere, z_sphere, s=1, color="black")
    ax[1].set_xlabel = ('x')
    ax[1].set_ylabel = ('z')
    ax[1].set_title(f'u_y on the xz plane', y=1.05)

    im_2 = ax[2].contourf(xx_, zz_, u_z_, extend='both', cmap = plt.cm.RdBu_r, levels = 128)
    cbar_2 = fig.colorbar(im_2, ax=ax[2], orientation="horizontal")
    cbar_2.ax.locator_params(nbins=5)
    ax[2].scatter(x_sphere, z_sphere, s=1, color="black")
    ax[2].set_xlabel = ('x')
    ax[2].set_ylabel = ('z')
    ax[2].set_title(f'u_z on the xz plane', y=1.05)

    im_3 = ax[3].contourf(xx_, zz_, u_abs_, extend='both', cmap = plt.cm.RdBu_r, levels = 128)
    cbar_2 = fig.colorbar(im_3, ax=ax[3], orientation="horizontal")
    cbar_2.ax.locator_params(nbins=5)
    ax[3].scatter(x_sphere, z_sphere, s=1, color="black")
    ax[3].set_xlabel = ('x')
    ax[3].set_ylabel = ('z')
    ax[3].set_title(f'|u| on the xz plane', y=1.05)

    plt.tight_layout(pad = 2)
    plt.show()

plt_velocity_xz(1)

def plt_velocity_yz(index):

    # compute neural velocities ######################################################################
    y_ = np.linspace(-1, 1, 256)
    z_ = np.linspace( 0, 1, 256)

    yy_, zz_ = np.meshgrid(y_, z_)
    xx_ = np.full(np.shape(zz_), 0)

    velocities_ = neural_velocity(xx_, yy_, zz_, index)

    u_x_ = U_dict[index] * tf.reshape(velocities_[:,0], np.shape(xx_))
    u_y_ = U_dict[index] * tf.reshape(velocities_[:,1], np.shape(yy_))
    u_z_ = U_dict[index] * tf.reshape(velocities_[:,2], np.shape(zz_))

    u_abs_ = tf.sqrt(u_x_**2 + u_y_**2 + u_z_**2)

    # find buildings ###################################################################################
    theta_ = tf.cast(tf.reshape(np.linspace(0, 2*np.pi, 128), [-1,1]), dtype=data_type)
    r_     = tf.constant(5, dtype=data_type) 

    y_sphere = (tf.cos(theta_) * r_) / L
    z_sphere = (50  + (tf.sin(theta_) * r_)) / h

    # plot results #####################################################################################

    fig, ax = plt.subplots(1,4, figsize=[16,4], dpi = 250)

    im_0 = ax[0].contourf(yy_, zz_, u_x_, extend='both', cmap = plt.cm.RdBu_r, levels = 128)
    cbar_0 = fig.colorbar(im_0, ax=ax[0], orientation="horizontal")
    cbar_0.ax.locator_params(nbins=5)
    ax[0].scatter(y_sphere, z_sphere, s=1, color="black")
    ax[0].set_xlabel = ('x')
    ax[0].set_ylabel = ('z')
    ax[0].set_title(f'u_x on the yz plane', y=1.05)

    im_1 = ax[1].contourf(yy_, zz_, u_y_, extend='both', cmap = plt.cm.RdBu_r, levels = 128)
    cbar_1 = fig.colorbar(im_1, ax=ax[1], orientation="horizontal")
    cbar_1.ax.locator_params(nbins=5)
    ax[1].scatter(y_sphere, z_sphere, s=1, color="black")
    ax[1].set_xlabel = ('x')
    ax[1].set_ylabel = ('z')
    ax[1].set_title(f'u_y on the yz plane', y=1.05)

    im_2 = ax[2].contourf(yy_, zz_, u_z_, extend='both', cmap = plt.cm.RdBu_r, levels = 128)
    cbar_2 = fig.colorbar(im_2, ax=ax[2], orientation="horizontal")
    cbar_2.ax.locator_params(nbins=5)
    ax[2].scatter(y_sphere, z_sphere, s=1, color="black")
    ax[2].set_xlabel = ('x')
    ax[2].set_ylabel = ('z')
    ax[2].set_title(f'u_z on the yz plane', y=1.05)

    im_3 = ax[3].contourf(yy_, zz_, u_abs_, extend='both', cmap = plt.cm.RdBu_r, levels = 128)
    cbar_2 = fig.colorbar(im_3, ax=ax[3], orientation="horizontal")
    cbar_2.ax.locator_params(nbins=5)
    ax[3].scatter(y_sphere, z_sphere, s=1, color="black")
    ax[3].set_xlabel = ('x')
    ax[3].set_ylabel = ('z')
    ax[3].set_title(f'|u| on the yz plane', y=1.05)

    plt.tight_layout(pad = 2)
    plt.show()

plt_velocity_yz(1)

def plt_velocity_xy(index):

    # compute neural velocities ######################################################################
    x_ = np.linspace(-1, 1, 256)
    y_ = np.linspace(-1, 1, 256)

    xx_, yy_ = np.meshgrid(x_, y_)
    zz_ = np.full(np.shape(xx_), 0.5)

    velocities_ = neural_velocity(xx_, yy_, zz_, index)

    u_x_ = U_dict[index] * tf.reshape(velocities_[:,0], np.shape(xx_))
    u_y_ = U_dict[index] * tf.reshape(velocities_[:,1], np.shape(yy_))
    u_z_ = U_dict[index] * tf.reshape(velocities_[:,2], np.shape(zz_))

    u_abs_ = tf.sqrt(u_x_**2 + u_y_**2 + u_z_**2)

    # find buildings ###################################################################################
    theta_ = tf.cast(tf.reshape(np.linspace(0, 2*np.pi, 128), [-1,1]), dtype=data_type)
    r_     = tf.constant(5, dtype=data_type) 

    x_sphere = (tf.cos(theta_) * r_) / L
    y_sphere = (tf.sin(theta_) * r_) / L

    # plot results #####################################################################################

    fig, ax = plt.subplots(1,4, figsize=[16,4], dpi = 250)

    im_0 = ax[0].contourf(xx_, yy_, u_x_, extend='both', cmap = plt.cm.RdBu_r, levels = 128)
    cbar_0 = fig.colorbar(im_0, ax=ax[0], orientation="horizontal")
    cbar_0.ax.locator_params(nbins=5)
    ax[0].scatter(x_sphere, y_sphere, s=1, color="black")
    ax[0].set_xlabel = ('x')
    ax[0].set_ylabel = ('y')
    ax[0].set_title(f'u_x on the xy plane', y=1.05)

    im_1 = ax[1].contourf(xx_, yy_, u_y_, extend='both', cmap = plt.cm.RdBu_r, levels = 128)
    cbar_1 = fig.colorbar(im_1, ax=ax[1], orientation="horizontal")
    cbar_1.ax.locator_params(nbins=5)
    ax[1].scatter(x_sphere, y_sphere, s=1, color="black")
    ax[1].set_xlabel = ('x')
    ax[1].set_ylabel = ('y')
    ax[1].set_title(f'u_y on the xy plane', y=1.05)

    im_2 = ax[2].contourf(xx_, yy_, u_z_, extend='both', cmap = plt.cm.RdBu_r, levels = 128)
    cbar_2 = fig.colorbar(im_2, ax=ax[2], orientation="horizontal")
    cbar_2.ax.locator_params(nbins=5)
    ax[2].scatter(x_sphere, y_sphere, s=1, color="black")
    ax[2].set_xlabel = ('x')
    ax[2].set_ylabel = ('y')
    ax[2].set_title(f'u_z on the xy plane', y=1.05)

    im_3 = ax[3].contourf(xx_, yy_, u_abs_, extend='both', cmap = plt.cm.RdBu_r, levels = 128)
    cbar_2 = fig.colorbar(im_3, ax=ax[3], orientation="horizontal")
    cbar_2.ax.locator_params(nbins=5)
    ax[3].scatter(x_sphere, y_sphere, s=1, color="black")
    ax[3].set_xlabel = ('x')
    ax[3].set_ylabel = ('y')
    ax[3].set_title(f'|u| on the xy plane', y=1.05)

    plt.tight_layout(pad = 2)
    plt.show()

plt_velocity_xy(1)

def plt_vorticity_xz(index):
    
    # compute neural velocities ######################################################################
    x_ = np.linspace(-1, 1, 64)
    z_ = np.linspace( 0, 1, 64)

    xx_, zz_ = np.meshgrid(x_, z_)
    yy_ = np.full(np.shape(xx_), 0)

    omega_ = neural_vorticity_tilde(xx_, yy_, zz_, index)

    omega_x_ = (U_dict[index]/L) * tf.reshape(omega_[:,0], np.shape(xx_))
    omega_y_ = (U_dict[index]/L) * tf.reshape(omega_[:,1], np.shape(yy_))
    omega_z_ = (U_dict[index]/L) * tf.reshape(omega_[:,2], np.shape(zz_))

    omega_abs_ = tf.sqrt(omega_x_**2 + omega_y_**2 + omega_z_**2)

    # find buildings ###################################################################################
    theta_ = tf.cast(tf.reshape(np.linspace(0, 2*np.pi, 128), [-1,1]), dtype=data_type)
    r_     = tf.constant(5, dtype=data_type) 

    x_sphere = (tf.cos(theta_) * r_) / L
    z_sphere = (50  + (tf.sin(theta_) * r_)) / h

    # plot results #####################################################################################

    fig, ax = plt.subplots(1,4, figsize=[16,4], dpi = 250)

    im_0 = ax[0].contourf(xx_, zz_, omega_x_, extend='both', cmap = plt.cm.RdBu_r, levels = 128)
    cbar_0 = fig.colorbar(im_0, ax=ax[0], orientation="horizontal")
    cbar_0.ax.locator_params(nbins=5)
    ax[0].scatter(x_sphere, z_sphere, s=1, color="black")
    ax[0].set_xlabel = ('x')
    ax[0].set_ylabel = ('z')
    ax[0].set_title(f'omega_x on the xz plane', y=1.05)

    im_1 = ax[1].contourf(xx_, zz_, omega_y_, extend='both', cmap = plt.cm.RdBu_r, levels = 128)
    cbar_1 = fig.colorbar(im_1, ax=ax[1], orientation="horizontal")
    cbar_1.ax.locator_params(nbins=5)
    ax[1].scatter(x_sphere, z_sphere, s=1, color="black")
    ax[1].set_xlabel = ('x')
    ax[1].set_ylabel = ('z')
    ax[1].set_title(f'omega_y on the xz plane', y=1.05)

    im_2 = ax[2].contourf(xx_, zz_, omega_z_, extend='both', cmap = plt.cm.RdBu_r, levels = 128)
    cbar_2 = fig.colorbar(im_2, ax=ax[2], orientation="horizontal")
    cbar_2.ax.locator_params(nbins=5)
    ax[2].scatter(x_sphere, z_sphere, s=1, color="black")
    ax[2].set_xlabel = ('x')
    ax[2].set_ylabel = ('z')
    ax[2].set_title(f'omega_z on the xz plane', y=1.05)

    im_3 = ax[3].contourf(xx_, zz_, omega_abs_, extend='both', cmap = plt.cm.RdBu_r, levels = 128)
    cbar_2 = fig.colorbar(im_3, ax=ax[3], orientation="horizontal")
    cbar_2.ax.locator_params(nbins=5)
    ax[3].scatter(x_sphere, z_sphere, s=1, color="black")
    ax[3].set_xlabel = ('x')
    ax[3].set_ylabel = ('z')
    ax[3].set_title(f'|omega| on the xz plane', y=1.05)

    plt.tight_layout(pad = 2)
    plt.show()

plt_vorticity_xz(1)

def plt_vorticity_yz(index):

    # compute neural velocities ######################################################################
    y_ = np.linspace(-1, 1, 64)
    z_ = np.linspace( 0, 1, 64)

    yy_, zz_ = np.meshgrid(y_, z_)
    xx_ = np.full(np.shape(zz_), 0)

    omega_ = neural_vorticity_tilde(xx_, yy_, zz_, index)

    omega_x_ = (U_dict[index]/L) * tf.reshape(omega_[:,0], np.shape(xx_))
    omega_y_ = (U_dict[index]/L) * tf.reshape(omega_[:,1], np.shape(yy_))
    omega_z_ = (U_dict[index]/L) * tf.reshape(omega_[:,2], np.shape(zz_))

    omega_abs_ = tf.sqrt(omega_x_**2 + omega_y_**2 + omega_z_**2)

    # find buildings ###################################################################################
    theta_ = tf.cast(tf.reshape(np.linspace(0, 2*np.pi, 128), [-1,1]), dtype=data_type)
    r_     = tf.constant(5, dtype=data_type) 

    y_sphere = (tf.cos(theta_) * r_) / L
    z_sphere = (50  + (tf.sin(theta_) * r_)) / h

    # plot results #####################################################################################

    fig, ax = plt.subplots(1,4, figsize=[16,4], dpi = 250)

    im_0 = ax[0].contourf(yy_, zz_, omega_x_, extend='both', cmap = plt.cm.RdBu_r, levels = 128)
    cbar_0 = fig.colorbar(im_0, ax=ax[0], orientation="horizontal")
    cbar_0.ax.locator_params(nbins=5)
    ax[0].scatter(y_sphere, z_sphere, s=1, color="black")
    ax[0].set_xlabel = ('x')
    ax[0].set_ylabel = ('z')
    ax[0].set_title(f'omega_x on the yz plane', y=1.05)

    im_1 = ax[1].contourf(yy_, zz_, omega_y_, extend='both', cmap = plt.cm.RdBu_r, levels = 128)
    cbar_1 = fig.colorbar(im_1, ax=ax[1], orientation="horizontal")
    cbar_1.ax.locator_params(nbins=5)
    ax[1].scatter(y_sphere, z_sphere, s=1, color="black")
    ax[1].set_xlabel = ('x')
    ax[1].set_ylabel = ('z')
    ax[1].set_title(f'omega_y on the yz plane', y=1.05)

    im_2 = ax[2].contourf(yy_, zz_, omega_z_, extend='both', cmap = plt.cm.RdBu_r, levels = 128)
    cbar_2 = fig.colorbar(im_2, ax=ax[2], orientation="horizontal")
    cbar_2.ax.locator_params(nbins=5)
    ax[2].scatter(y_sphere, z_sphere, s=1, color="black")
    ax[2].set_xlabel = ('x')
    ax[2].set_ylabel = ('z')
    ax[2].set_title(f'omega_z on the yz plane', y=1.05)

    im_3 = ax[3].contourf(yy_, zz_, omega_abs_, extend='both', cmap = plt.cm.RdBu_r, levels = 128)
    cbar_2 = fig.colorbar(im_3, ax=ax[3], orientation="horizontal")
    cbar_2.ax.locator_params(nbins=5)
    ax[3].scatter(y_sphere, z_sphere, s=1, color="black")
    ax[3].set_xlabel = ('x')
    ax[3].set_ylabel = ('z')
    ax[3].set_title(f'|omega| on the yz plane', y=1.05)

    plt.tight_layout(pad = 2)
    plt.show()

plt_vorticity_yz(1)

def plt_vorticity_xy(index):

    # compute neural velocities ######################################################################
    x_ = np.linspace(-1, 1, 64)
    y_ = np.linspace(-1, 1, 64)

    xx_, yy_ = np.meshgrid(x_, y_)
    zz_ = np.full(np.shape(xx_), 0.5)

    omega_ = neural_vorticity_tilde(xx_, yy_, zz_, index)

    omega_x_ = (U_dict[index]/L) * tf.reshape(omega_[:,0], np.shape(xx_))
    omega_y_ = (U_dict[index]/L) * tf.reshape(omega_[:,1], np.shape(yy_))
    omega_z_ = (U_dict[index]/L) * tf.reshape(omega_[:,2], np.shape(zz_))

    omega_abs_ = tf.sqrt(omega_x_**2 + omega_y_**2 + omega_z_**2)

    # find buildings ###################################################################################
    theta_ = tf.cast(tf.reshape(np.linspace(0, 2*np.pi, 128), [-1,1]), dtype=data_type)
    r_     = tf.constant(5, dtype=data_type) 

    x_sphere = (tf.cos(theta_) * r_) / L
    y_sphere = (tf.sin(theta_) * r_) / L

    # plot results #####################################################################################

    fig, ax = plt.subplots(1,4, figsize=[16,4], dpi = 250)

    im_0 = ax[0].contourf(xx_, yy_, omega_x_, extend='both', cmap = plt.cm.RdBu_r, levels = 128)
    cbar_0 = fig.colorbar(im_0, ax=ax[0], orientation="horizontal")
    cbar_0.ax.locator_params(nbins=5)
    ax[0].scatter(x_sphere, y_sphere, s=1, color="black")
    ax[0].set_xlabel = ('x')
    ax[0].set_ylabel = ('y')
    ax[0].set_title(f'omega_x on the xy plane', y=1.05)

    im_1 = ax[1].contourf(xx_, yy_, omega_y_, extend='both', cmap = plt.cm.RdBu_r, levels = 128)
    cbar_1 = fig.colorbar(im_1, ax=ax[1], orientation="horizontal")
    cbar_1.ax.locator_params(nbins=5)
    ax[1].scatter(x_sphere, y_sphere, s=1, color="black")
    ax[1].set_xlabel = ('x')
    ax[1].set_ylabel = ('y')
    ax[1].set_title(f'omega_y on the xy plane', y=1.05)

    im_2 = ax[2].contourf(xx_, yy_, omega_z_, extend='both', cmap = plt.cm.RdBu_r, levels = 128)
    cbar_2 = fig.colorbar(im_2, ax=ax[2], orientation="horizontal")
    cbar_2.ax.locator_params(nbins=5)
    ax[2].scatter(x_sphere, y_sphere, s=1, color="black")
    ax[2].set_xlabel = ('x')
    ax[2].set_ylabel = ('y')
    ax[2].set_title(f'omega_z on the xy plane', y=1.05)

    im_3 = ax[3].contourf(xx_, yy_, omega_abs_, extend='both', cmap = plt.cm.RdBu_r, levels = 128)
    cbar_2 = fig.colorbar(im_3, ax=ax[3], orientation="horizontal")
    cbar_2.ax.locator_params(nbins=5)
    ax[3].scatter(x_sphere, y_sphere, s=1, color="black")
    ax[3].set_xlabel = ('x')
    ax[3].set_ylabel = ('y')
    ax[3].set_title(f'|omega| on the xy plane', y=1.05)

    plt.tight_layout(pad = 2)
    plt.show()

plt_vorticity_xy(1)

def plot_eqn(y_loc, index):
    
    num_internal = 256
    
    x_ = tf.cast(tf.reshape(np.full(num_internal, 0), [-1,1]), dtype=data_type)
    y_ = tf.cast(tf.reshape(np.full(num_internal, y_loc), [-1,1]), dtype=data_type)
    z_ = tf.cast(tf.reshape(np.linspace(0, 1, num_internal), [-1,1]), dtype=data_type)

    with tf.GradientTape(persistent=True) as tape_:

        tape_.watch(x_)
        tape_.watch(y_)
        tape_.watch(z_)

        with tf.GradientTape(persistent=True) as tape:

            tape.watch(x_)
            tape.watch(y_)
            tape.watch(z_)

            u_bar = neural_velocity(x_, y_, z_, index)

            p_ = neural_p_tilde(x_, y_, z_, index)

            u_x_ = tf.reshape(u_bar[:,0], [-1,1])
            u_y_ = tf.reshape(u_bar[:,1], [-1,1])
            u_z_ = tf.reshape(u_bar[:,2], [-1,1])

        grad_u_x_x = tape.gradient(u_x_, x_)
        grad_u_x_y = tape.gradient(u_x_, y_)
        grad_u_x_z = tape.gradient(u_x_, z_)

        grad_u_y_x = tape.gradient(u_y_, x_)
        grad_u_y_y = tape.gradient(u_y_, y_)
        grad_u_y_z = tape.gradient(u_y_, z_)

        grad_u_z_x = tape.gradient(u_z_, x_)
        grad_u_z_y = tape.gradient(u_z_, y_)
        grad_u_z_z = tape.gradient(u_z_, z_)

        nut_ = Re_inv_cal(x_, y_, z_, index)

    grad_u_x_xx = tape_.gradient(grad_u_x_x, x_)
    grad_u_x_yy = tape_.gradient(grad_u_x_y, y_)
    grad_u_x_zz = tape_.gradient(grad_u_x_z, z_)

    grad_u_y_xx = tape_.gradient(grad_u_y_x, x_)
    grad_u_y_yy = tape_.gradient(grad_u_y_y, y_)
    grad_u_y_zz = tape_.gradient(grad_u_y_z, z_)

    grad_u_z_xx = tape_.gradient(grad_u_z_x, x_)
    grad_u_z_yy = tape_.gradient(grad_u_z_y, y_)
    grad_u_z_zz = tape_.gradient(grad_u_z_z, z_)

    mask_ = neural_mask(x_, y_, z_)

    grad_p_x = mask_ * tape.gradient(p_, x_)
    grad_p_y = mask_ * tape.gradient(p_, y_)
    grad_p_z = mask_ * tape.gradient(p_, z_)

    grad_nut_x = tape_.gradient(nut_, x_)
    grad_nut_y = tape_.gradient(nut_, y_)
    grad_nut_z = tape_.gradient(nut_, z_)

    u_del_u_x = u_x_ * grad_u_x_x + u_y_ * grad_u_x_y + u_z_ * grad_u_x_z
    u_del_u_y = u_x_ * grad_u_y_x + u_y_ * grad_u_y_y + u_z_ * grad_u_y_z
    u_del_u_z = u_x_ * grad_u_z_x + u_y_ * grad_u_z_y + u_z_ * grad_u_z_z

    term_nut_x_x = grad_nut_x * (grad_u_x_x + grad_u_x_x) 
    term_nut_x_y = grad_nut_y * (grad_u_x_y + grad_u_y_x)  
    term_nut_x_z = grad_nut_z * (grad_u_x_z + grad_u_z_x) 

    term_nut_y_x = grad_nut_x * (grad_u_y_x + grad_u_x_y)  
    term_nut_y_y = grad_nut_y * (grad_u_y_y + grad_u_y_y)  
    term_nut_y_z = grad_nut_z * (grad_u_y_z + grad_u_z_y)

    term_nut_z_x = grad_nut_x * (grad_u_z_x + grad_u_x_z)  
    term_nut_z_y = grad_nut_y * (grad_u_z_y + grad_u_y_z)  
    term_nut_z_z = grad_nut_z * (grad_u_z_z + grad_u_z_z) 


    Laplacian_x = (grad_u_x_xx + grad_u_x_yy + grad_u_x_zz)
    Laplacian_y = (grad_u_y_xx + grad_u_y_yy + grad_u_y_zz)
    Laplacian_z = (grad_u_z_xx + grad_u_z_yy + grad_u_z_zz)

    viscous_x = ( nut_ * Laplacian_x   + 
                  term_nut_x_x  + 
                  term_nut_x_y  + 
                  term_nut_x_z  )

    viscous_y = ( nut_ * Laplacian_y   + 
                  term_nut_y_x  + 
                  term_nut_y_y  + 
                  term_nut_y_z  )

    viscous_z = ( nut_ * Laplacian_z   + 
                  term_nut_z_x  + 
                  term_nut_z_y  + 
                  term_nut_z_z  )

    Eq_x = viscous_x - grad_p_x - u_del_u_x
    Eq_y = viscous_y - grad_p_y - u_del_u_y
    Eq_z = viscous_z - grad_p_z - u_del_u_z

    fig, ax = plt.subplots(1,5, figsize=[15,6], dpi=250)

    ax[0].plot(Eq_x,  z_,  label='rest x')
    ax[0].plot(Eq_y,  z_,  label='rest y')
    ax[0].plot(Eq_z,  z_,  label='rest z')
    ax[0].legend(loc = 'upper right')

    ax[1].plot(nut_ * Laplacian_x,  z_,  label='Laplacian x')
    ax[1].plot(nut_ * Laplacian_y,  z_,  label='Laplacian y')
    ax[1].plot(nut_ * Laplacian_z,  z_,  label='Laplacian z')
    ax[1].legend(loc = 'upper right')

    ax[2].plot(Laplacian_x,  z_,  label='Laplacian x')
    ax[2].plot(Laplacian_y,  z_,  label='Laplacian y')
    ax[2].plot(Laplacian_z,  z_,  label='Laplacian z')
    ax[2].legend(loc = 'upper right')

    ax[3].plot(grad_p_x, z_, label='grad_p_x')
    ax[3].plot(grad_p_y, z_, label='grad_p_y')
    ax[3].plot(grad_p_z, z_, label='grad_p_z')
    ax[3].legend(loc = 'upper right')

    ax[4].plot(u_del_u_x, z_, label='u_del_u_x')
    ax[4].plot(u_del_u_y, z_, label='u_del_u_y')
    ax[4].plot(u_del_u_z, z_, label='u_del_u_z')
    ax[4].legend(loc = 'upper right')

    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(1,5, figsize=[15,6], dpi=250)

    ax[0].plot(grad_nut_x, z_, label='grad_nut_x')
    ax[0].plot(grad_nut_y, z_, label='grad_nut_y')
    ax[0].plot(grad_nut_z, z_, label='grad_nut_z')
    ax[0].legend(loc='upper right')

    ax[1].plot((term_nut_x_x + term_nut_x_y + term_nut_x_z),  z_,  label='term_nut_x')
    ax[1].plot((term_nut_y_x + term_nut_y_y + term_nut_y_z),  z_,  label='term_nut_y')
    ax[1].plot((term_nut_z_x + term_nut_z_y + term_nut_z_z),  z_,  label='term_nut_z')
    ax[1].legend(loc = 'upper right')

    ax[2].plot(grad_u_x_x,  z_,  label='grad_u_x_x')
    ax[2].plot(grad_u_x_y,  z_,  label='grad_u_x_y')
    ax[2].plot(grad_u_x_z,  z_,  label='grad_u_x_z')
    ax[2].legend(loc = 'upper right')

    ax[3].plot(grad_u_y_x,  z_,  label='grad_u_y_x')
    ax[3].plot(grad_u_y_y,  z_,  label='grad_u_y_y')
    ax[3].plot(grad_u_y_z,  z_,  label='grad_u_y_z')
    ax[3].legend(loc = 'upper right')

    ax[4].plot(grad_u_z_x,  z_,  label='grad_u_z_x')
    ax[4].plot(grad_u_z_y,  z_,  label='grad_u_z_y')
    ax[4].plot(grad_u_z_z,  z_,  label='grad_u_z_z')
    ax[4].legend(loc = 'upper right')

    plt.tight_layout()
    plt.show()

plot_eqn(y_loc=0,    index=1)
plot_eqn(y_loc=-0.2, index=1)

def plt_dev_xz(index):
    
    # deviation_internal varphi=0 ###################################################################################
    inds_internal_0, _ = np.where(tf.abs(y_internal) < 0.25/h)

    x_internal_0 = tf.reshape(tf.gather(x_internal, inds_internal_0), [-1,])
    y_internal_0 = tf.reshape(tf.gather(y_internal, inds_internal_0), [-1,])
    z_internal_0 = tf.reshape(tf.gather(z_internal, inds_internal_0), [-1,])

    velocities_internal_0  = U_dict[index] * neural_velocity(x_internal_0, y_internal_0, z_internal_0, index)
    vorticities_internal_0 = (U_dict[index]/L) * neural_vorticity_tilde(x_internal_0, y_internal_0, z_internal_0, index)

    dev_u_x = (velocities_internal_0[:,0] - tf.gather(u_ref_internal_dict[index], inds_internal_0)[:,0])
    dev_u_y = (velocities_internal_0[:,1] - tf.gather(u_ref_internal_dict[index], inds_internal_0)[:,1])
    dev_u_z = (velocities_internal_0[:,2] - tf.gather(u_ref_internal_dict[index], inds_internal_0)[:,2])

    dev_omega_x = vorticities_internal_0[:,0] - tf.gather(omega_ref_internal_dict[index], inds_internal_0)[:,0]
    dev_omega_y = vorticities_internal_0[:,1] - tf.gather(omega_ref_internal_dict[index], inds_internal_0)[:,1]
    dev_omega_z = vorticities_internal_0[:,2] - tf.gather(omega_ref_internal_dict[index], inds_internal_0)[:,2]
                              
    ##################################################################################################################
    fig, ax = plt.subplots(2,3, figsize=[16,6], dpi = 450)

    im_0 = ax[0][0].scatter(x_internal_0, z_internal_0, s=15, c=dev_u_x, cmap = plt.cm.RdBu_r, alpha=0.85)
    plt.colorbar(im_0, ax=ax[0][0], pad=0.02)
    ax[0][0].set_title('deviation u_x,  xz plane', y=1.05)

    im_1 = ax[0][1].scatter(x_internal_0, z_internal_0, s=15, c=dev_u_y, cmap = plt.cm.RdBu_r, alpha=0.85)
    plt.colorbar(im_1, ax=ax[0][1], pad=0.02)
    ax[0][1].set_title('deviation u_y,  xz plane', y=1.05)

    im_2 = ax[0][2].scatter(x_internal_0, z_internal_0, s=15, c=dev_u_z, cmap = plt.cm.RdBu_r, alpha=0.85)
    plt.colorbar(im_2, ax=ax[0][2], pad=0.02)
    ax[0][2].set_title('deviation u_z,  xz plane', y=1.05)

    im_3 = ax[1][0].scatter(x_internal_0, z_internal_0, s=15, c=dev_omega_x, cmap = plt.cm.RdBu_r, alpha=0.85)
    plt.colorbar(im_3, ax=ax[1][0], pad=0.02)
    ax[1][0].set_title('deviation omega_x,  xz plane', y=1.05)

    im_4 = ax[1][1].scatter(x_internal_0, z_internal_0, s=15, c=dev_omega_y, cmap = plt.cm.RdBu_r, alpha=0.85)
    plt.colorbar(im_4, ax=ax[1][1], pad=0.02)
    ax[1][1].set_title('deviation omega_y,  xz plane', y=1.05)

    im_5 = ax[1][2].scatter(x_internal_0, z_internal_0, s=15, c=dev_omega_z, cmap = plt.cm.RdBu_r, alpha=0.85)
    plt.colorbar(im_5, ax=ax[1][2], pad=0.02)
    ax[1][2].set_title('deviation oemga_z,  xz plane', y=1.05)

    plt.tight_layout(pad=1)
    plt.show()
    
plt_dev_xz(1)

def plt_dev_yz(index):
    
    # deviation_internal varphi=0 ###################################################################################
    inds_internal_0, _ = np.where(tf.abs(x_internal) < 0.25/h)

    x_internal_0 = tf.reshape(tf.gather(x_internal, inds_internal_0), [-1,])
    y_internal_0 = tf.reshape(tf.gather(y_internal, inds_internal_0), [-1,])
    z_internal_0 = tf.reshape(tf.gather(z_internal, inds_internal_0), [-1,])

    velocities_internal_0  = U_dict[index] * neural_velocity(x_internal_0, y_internal_0, z_internal_0, index)
    vorticities_internal_0 = (U_dict[index]/L) * neural_vorticity_tilde(x_internal_0, y_internal_0, z_internal_0, index)

    dev_u_x = (velocities_internal_0[:,0] - tf.gather(u_ref_internal_dict[index], inds_internal_0)[:,0])
    dev_u_y = (velocities_internal_0[:,1] - tf.gather(u_ref_internal_dict[index], inds_internal_0)[:,1])
    dev_u_z = (velocities_internal_0[:,2] - tf.gather(u_ref_internal_dict[index], inds_internal_0)[:,2])

    dev_omega_x = vorticities_internal_0[:,0] - tf.gather(omega_ref_internal_dict[index], inds_internal_0)[:,0]
    dev_omega_y = vorticities_internal_0[:,1] - tf.gather(omega_ref_internal_dict[index], inds_internal_0)[:,1]
    dev_omega_z = vorticities_internal_0[:,2] - tf.gather(omega_ref_internal_dict[index], inds_internal_0)[:,2]
                              
    ##################################################################################################################
    fig, ax = plt.subplots(2,3, figsize=[16,6], dpi = 450)

    im_0 = ax[0][0].scatter(y_internal_0, z_internal_0, s=15, c=dev_u_x, cmap = plt.cm.RdBu_r, alpha=0.85)
    plt.colorbar(im_0, ax=ax[0][0], pad=0.02)
    ax[0][0].set_title('deviation u_x,  yz plane', y=1.05)

    im_1 = ax[0][1].scatter(y_internal_0, z_internal_0, s=15, c=dev_u_y, cmap = plt.cm.RdBu_r, alpha=0.85)
    plt.colorbar(im_1, ax=ax[0][1], pad=0.02)
    ax[0][1].set_title('deviation u_y,  yz plane', y=1.05)

    im_2 = ax[0][2].scatter(y_internal_0, z_internal_0, s=15, c=dev_u_z, cmap = plt.cm.RdBu_r, alpha=0.85)
    plt.colorbar(im_2, ax=ax[0][2], pad=0.02)
    ax[0][2].set_title('deviation u_z,  yz plane', y=1.05)

    im_3 = ax[1][0].scatter(y_internal_0, z_internal_0, s=15, c=dev_omega_x, cmap = plt.cm.RdBu_r, alpha=0.85)
    plt.colorbar(im_3, ax=ax[1][0], pad=0.02)
    ax[1][0].set_title('deviation omega_x,  yz plane', y=1.05)

    im_4 = ax[1][1].scatter(y_internal_0, z_internal_0, s=15, c=dev_omega_y, cmap = plt.cm.RdBu_r, alpha=0.85)
    plt.colorbar(im_4, ax=ax[1][1], pad=0.02)
    ax[1][1].set_title('deviation omega_y,  yz plane', y=1.05)

    im_5 = ax[1][2].scatter(y_internal_0, z_internal_0, s=15, c=dev_omega_z, cmap = plt.cm.RdBu_r, alpha=0.85)
    plt.colorbar(im_5, ax=ax[1][2], pad=0.02)
    ax[1][2].set_title('deviation oemga_z,  yz plane', y=1.05)

    plt.tight_layout(pad=1)
    plt.show()
    
plt_dev_yz(1)

def plt_dev_xy(index):
    
    # deviation_internal varphi=0 ###################################################################################
    inds_internal_0, _ = np.where(tf.abs(z_internal-0.5) < 0.25/h)

    x_internal_0 = tf.reshape(tf.gather(x_internal, inds_internal_0), [-1,])
    y_internal_0 = tf.reshape(tf.gather(y_internal, inds_internal_0), [-1,])
    z_internal_0 = tf.reshape(tf.gather(z_internal, inds_internal_0), [-1,])

    velocities_internal_0  = U_dict[index] * neural_velocity(x_internal_0, y_internal_0, z_internal_0, index)
    vorticities_internal_0 = (U_dict[index]/L) * neural_vorticity_tilde(x_internal_0, y_internal_0, z_internal_0, index)

    dev_u_x = (velocities_internal_0[:,0] - tf.gather(u_ref_internal_dict[index], inds_internal_0)[:,0])
    dev_u_y = (velocities_internal_0[:,1] - tf.gather(u_ref_internal_dict[index], inds_internal_0)[:,1])
    dev_u_z = (velocities_internal_0[:,2] - tf.gather(u_ref_internal_dict[index], inds_internal_0)[:,2])

    dev_omega_x = vorticities_internal_0[:,0] - tf.gather(omega_ref_internal_dict[index], inds_internal_0)[:,0]
    dev_omega_y = vorticities_internal_0[:,1] - tf.gather(omega_ref_internal_dict[index], inds_internal_0)[:,1]
    dev_omega_z = vorticities_internal_0[:,2] - tf.gather(omega_ref_internal_dict[index], inds_internal_0)[:,2]
                              
    ##################################################################################################################
    fig, ax = plt.subplots(2,3, figsize=[16,6], dpi = 450)

    im_0 = ax[0][0].scatter(x_internal_0, y_internal_0, s=15, c=dev_u_x, cmap = plt.cm.RdBu_r, alpha=0.85)
    plt.colorbar(im_0, ax=ax[0][0], pad=0.02)
    ax[0][0].set_title('deviation u_x,  xy plane', y=1.05)

    im_1 = ax[0][1].scatter(x_internal_0, y_internal_0, s=15, c=dev_u_y, cmap = plt.cm.RdBu_r, alpha=0.85)
    plt.colorbar(im_1, ax=ax[0][1], pad=0.02)
    ax[0][1].set_title('deviation u_y,  xy plane', y=1.05)

    im_2 = ax[0][2].scatter(x_internal_0, y_internal_0, s=15, c=dev_u_z, cmap = plt.cm.RdBu_r, alpha=0.85)
    plt.colorbar(im_2, ax=ax[0][2], pad=0.02)
    ax[0][2].set_title('deviation u_z,  xy plane', y=1.05)

    im_3 = ax[1][0].scatter(x_internal_0, y_internal_0, s=15, c=dev_omega_x, cmap = plt.cm.RdBu_r, alpha=0.85)
    plt.colorbar(im_3, ax=ax[1][0], pad=0.02)
    ax[1][0].set_title('deviation omega_x,  xy plane', y=1.05)

    im_4 = ax[1][1].scatter(x_internal_0, y_internal_0, s=15, c=dev_omega_y, cmap = plt.cm.RdBu_r, alpha=0.85)
    plt.colorbar(im_4, ax=ax[1][1], pad=0.02)
    ax[1][1].set_title('deviation omega_y,  xy plane', y=1.05)

    im_5 = ax[1][2].scatter(x_internal_0, y_internal_0, s=15, c=dev_omega_z, cmap = plt.cm.RdBu_r, alpha=0.85)
    plt.colorbar(im_5, ax=ax[1][2], pad=0.02)
    ax[1][2].set_title('deviation oemga_z,  xy plane', y=1.05)

    plt.tight_layout(pad=1)
    plt.show()
    
plt_dev_xy(1)

def plt_dev(loss_fit_u_list, loss_fit_p_list, loss_fit_nut_list, loss_eqn_x_list, loss_eqn_y_list, loss_eqn_z_list, index):
    
    # deviation_internal varphi=0 ###################################################################################
    inds_internal_0, _ = np.where(tf.abs(x_internal) < 0.25/h)

    x_internal_0 = tf.reshape(tf.gather(x_internal, inds_internal_0), [-1,])
    y_internal_0 = tf.reshape(tf.gather(y_internal, inds_internal_0), [-1,])
    z_internal_0 = tf.reshape(tf.gather(z_internal, inds_internal_0), [-1,])

    velocities_internal_0  = U_dict[index] * neural_velocity(x_internal_0, y_internal_0, z_internal_0, index)
    vorticities_internal_0 = (U_dict[index]/L) * neural_vorticity_tilde(x_internal_0, y_internal_0, z_internal_0, index)

    dev_internal_velocity  = tf.sort(tf.sqrt(tf.reduce_sum(tf.square(velocities_internal_0 - tf.gather(u_ref_internal_dict[index], inds_internal_0)), axis=1, keepdims=True)), axis=0)
    dev_internal_vorticity = tf.sort(tf.sqrt(tf.reduce_sum(tf.square(vorticities_internal_0 - tf.gather(omega_ref_internal_dict[index], inds_internal_0)), axis=1, keepdims=True)), axis=0)

    ##################################################################################################################
    
    # deviation building varphi=0 ####################################################################################
    inds_buildings_0, _ = np.where(tf.abs(x_buildings) < 0.25/h)
    
    x_buildings_0 = tf.reshape(tf.gather(x_buildings, inds_buildings_0), [-1,])
    y_buildings_0 = tf.reshape(tf.gather(y_buildings, inds_buildings_0), [-1,])
    z_buildings_0 = tf.reshape(tf.gather(z_buildings, inds_buildings_0), [-1,])
    
    velocities_buildings_0  = U_dict[index] * neural_velocity(x_buildings_0, y_buildings_0, z_buildings_0, index)
    vorticities_buildings_0 = (U_dict[index]/L) * neural_vorticity_tilde(x_buildings_0, y_buildings_0, z_buildings_0, index)
    
    dev_buildings_velocity  = U_dict[index] * tf.sort(tf.sqrt(tf.reduce_sum(tf.square(velocities_buildings_0), axis=1, keepdims=True)), axis=0)
    dev_buildings_vorticity = tf.sort(tf.sqrt(tf.reduce_sum(tf.square(vorticities_buildings_0), axis=1, keepdims=True)), axis=0)
    ##################################################################################################################
                              
    ##################################################################################################################
    fig, ax = plt.subplots(2,4, figsize=[16,6], dpi = 450)

    im_0 = ax[0][0].scatter(y_internal_0, z_internal_0, s=15, c=dev_internal_velocity, cmap = plt.cm.RdBu_r, alpha=0.85)
    plt.colorbar(im_0, ax=ax[0][0], pad=0.02)
    ax[0][0].set_title('RMSE velocity,  yz plane', y=1.05)

    im_1 = ax[0][1].scatter(y_internal_0, z_internal_0, s=15, c=dev_internal_vorticity, cmap = plt.cm.RdBu_r, alpha=0.85)
    plt.colorbar(im_1, ax=ax[0][1], pad=0.02)
    ax[0][1].set_title('RMSE vorticity, yz plane', y=1.05)

    ax[0][2].semilogy(loss_fit_u_list,  label='loss_fit_u')
    ax[0][2].legend(loc='best')
    
    ax[0][3].semilogy(loss_fit_p_list,  label='loss_fit_p')
    ax[0][3].legend(loc='best')
    
    im_2 = ax[1][0].scatter(y_buildings_0, z_buildings_0, s=15, c=dev_buildings_velocity, cmap = plt.cm.RdBu_r, alpha=0.85)
    plt.colorbar(im_2, ax=ax[1][0], pad=0.02)
    ax[1][0].set_title('RMSE velocity, yz plane', y=1.05)
    
    im_3 = ax[1][1].scatter(y_buildings_0, z_buildings_0, s=15, c=dev_buildings_vorticity, cmap = plt.cm.RdBu_r, alpha=0.85)
    plt.colorbar(im_3, ax=ax[1][1], pad=0.02)
    ax[1][1].set_title('RMSE velocity, yz plane', y=1.05)
    
    ax[1][2].semilogy(loss_fit_nut_list,  label='loss_fit_nut')
    ax[1][2].legend(loc='best')
    
    ax[1][3].semilogy(loss_eqn_x_list,  label='loss_eqn_x')
    ax[1][3].semilogy(loss_eqn_y_list,  label='loss_eqn_y')
    ax[1][3].semilogy(loss_eqn_z_list,  label='loss_eqn_z')
    ax[1][3].legend(loc='best')

    plt.tight_layout(pad=1)
    plt.show()
    
plt_dev([], [], [], [], [], [], 1)

def loss_fit_u_cal(index):
        
    # down sample dataset #################################################
    num_internal  = 16**3

    inds_internal = tf.range(len(x_internal))
    inds_internal_ds = tf.random.shuffle(inds_internal)[:num_internal]

    x_internal_ds = tf.gather(x_internal, inds_internal_ds)
    y_internal_ds = tf.gather(y_internal, inds_internal_ds)
    z_internal_ds = tf.gather(z_internal, inds_internal_ds)
    # compute velocities ###########################################################
    u_ref_internal_ds = tf.gather(u_ref_internal_dict[index], inds_internal_ds)
    u_nn_internal_ds  = U_dict[index] * neural_velocity(x_internal_ds, y_internal_ds, z_internal_ds, index)
    # compute losses ###############################################################

    weight_fit_u = tf.clip_by_value(tf.reduce_mean(tf.abs(u_ref_internal_ds), axis=0) / tf.abs(u_ref_internal_ds), clip_value_min = 0.1, clip_value_max=10)

    loss_fit_u = tf.reduce_mean(weight_fit_u * tf.square(u_nn_internal_ds - u_ref_internal_ds))
    
    return loss_fit_u

time_0 = time.time()
loss_fit_u = loss_fit_u_cal(1)
time_1 = time.time()

print(f'computation time = {time_1 - time_0}')
print(f'loss_fit_u = {loss_fit_u}')

def loss_fit_nut_cal(index):
    
    # down sample dataset #################################################
    num_internal  = 16**3

    inds_internal = tf.range(len(x_internal))
    inds_internal_ds = tf.random.shuffle(inds_internal)[:num_internal]

    x_internal_ds = tf.gather(x_internal, inds_internal_ds)
    y_internal_ds = tf.gather(y_internal, inds_internal_ds)
    z_internal_ds = tf.gather(z_internal, inds_internal_ds)
    # compute velocities ###########################################################
    nut_ref_internal_ds  = tf.gather(nut_ref_internal_dict[index], inds_internal_ds)
    nut_nn_internal_ds   = neural_nut(x_internal_ds, y_internal_ds, z_internal_ds, index)
    # compute losses ###############################################################

    weight_fit_nut = tf.clip_by_value(tf.reduce_mean(tf.abs(nut_ref_internal_ds)) / tf.abs(nut_ref_internal_ds), clip_value_min = 0.1, clip_value_max=10)
    loss_fit_nut = tf.reduce_mean(weight_fit_nut * tf.square(nut_ref_internal_ds - nut_nn_internal_ds))
    
    return loss_fit_nut

time_0 = time.time()
loss_fit_nut = loss_fit_nut_cal(1)
time_1 = time.time()

print(f'computation time = {time_1 - time_0}')
print(f'loss_fit_nut = {loss_fit_nut}')

def loss_fit_p_cal(index):
    
    # down sample dataset #################################################
    num_internal  = 16**3

    inds_internal = tf.range(len(x_internal))
    inds_internal_ds = tf.random.shuffle(inds_internal)[:num_internal]

    x_internal_ds = tf.gather(x_internal, inds_internal_ds)
    y_internal_ds = tf.gather(y_internal, inds_internal_ds)
    z_internal_ds = tf.gather(z_internal, inds_internal_ds)
    # compute velocities ###########################################################
    p_ref_internal_ds = tf.gather(p_ref_internal_dict[index], inds_internal_ds)

    p_nn_internal_ds  = 1.23 * U_dict[index]**2 * neural_p_tilde(x_internal_ds, y_internal_ds, z_internal_ds, index)
    # compute losses ###############################################################

    weight_fit_p = tf.clip_by_value(tf.reduce_mean(tf.abs(p_ref_internal_ds)) / tf.abs(p_ref_internal_ds), clip_value_min = 0.1, clip_value_max=10)
    loss_fit_p = tf.reduce_mean(weight_fit_p * tf.square(p_nn_internal_ds - p_ref_internal_ds))
    
    return loss_fit_p

time_0 = time.time()
loss_fit_p = loss_fit_p_cal(1)
time_1 = time.time()

print(f'computation time = {time_1 - time_0}')
print(f'loss_fit_p = {loss_fit_p}')

def loss_fit_mask_cal(index):
    
    # down sample dataset #################################################
    num_buildings =  1024
    
    inds_buildings = tf.range(len(x_buildings))
    inds_buildings_ds = tf.random.shuffle(inds_buildings)[:num_buildings]
    
    x_buildings_ds = tf.gather(x_buildings, inds_buildings_ds)
    y_buildings_ds = tf.gather(y_buildings, inds_buildings_ds)
    z_buildings_ds = tf.gather(z_buildings, inds_buildings_ds)

    num_internal  = 2048 * 2

    inds_internal = tf.range(len(x_internal))
    inds_internal_ds = tf.random.shuffle(inds_internal)[:num_internal]

    x_internal_ds = tf.gather(x_internal, inds_internal_ds)
    y_internal_ds = tf.gather(y_internal, inds_internal_ds)
    z_internal_ds = tf.gather(z_internal, inds_internal_ds)

    # compute losses ###############################################################
    
    loss_mask_bulk = tf.reduce_mean(tf.square(neural_mask(x_internal_ds,  y_internal_ds,  z_internal_ds) - 1))
    loss_mask_insd = tf.reduce_mean(tf.square(neural_mask(x_buildings_ds, y_buildings_ds, z_buildings_ds) - 0))

    loss_fit_mask = loss_mask_bulk + loss_mask_insd
    
    return loss_fit_mask

time_0 = time.time()
loss_fit_mask = loss_fit_mask_cal(1)
time_1 = time.time()

print(f'computation time = {time_1 - time_0}')
print(f'loss_fit_mask = {loss_fit_mask}')

def loss_bcs_cal(index):
    
    num_buildings =  2048
    
    inds_buildings = tf.range(len(x_buildings))
    inds_buildings_ds = tf.random.shuffle(inds_buildings)[:num_buildings]
    
    x_buildings_surf_ds = tf.gather(x_buildings_surf, inds_buildings_ds)
    y_buildings_surf_ds = tf.gather(y_buildings_surf, inds_buildings_ds)
    z_buildings_surf_ds = tf.gather(z_buildings_surf, inds_buildings_ds)
    
    u_nn_buildings_surf_ds   = neural_velocity(x_buildings_surf_ds, y_buildings_surf_ds, z_buildings_surf_ds, index)   
    nut_nn_buildings_surf_ds = Re_inv_cal(x_buildings_surf_ds, y_buildings_surf_ds, z_buildings_surf_ds, index)
    
    weight_bcs_u   = tf.stop_gradient(tf.clip_by_value(tf.reduce_mean(tf.abs(u_nn_buildings_surf_ds)) / tf.abs(u_nn_buildings_surf_ds), clip_value_min = 0.1, clip_value_max=10))
    weight_bcs_nut = tf.stop_gradient(tf.clip_by_value(tf.reduce_mean(tf.abs(nut_nn_buildings_surf_ds)) / tf.abs(nut_nn_buildings_surf_ds), clip_value_min = 0.1, clip_value_max=10))
    
    loss_fit_bc = tf.reduce_mean(weight_bcs_u * tf.square(u_nn_buildings_surf_ds)) + tf.reduce_mean(weight_bcs_nut * tf.square(nut_nn_buildings_surf_ds))

    return loss_fit_bc

time_0 = time.time()
loss_bcs = loss_bcs_cal(1)
time_1 = time.time()

print(f'computation time = {time_1 - time_0}')
print(f'loss_bcs = {loss_bcs}')

def loss_div_cal(index):
    
    """
    the time difference between computing grad_z^2 and the full Laplacian is 0.01s per iteration
    """
    
    # down sample dataset #################################################   
    num_internal  = 1024

    x_1 = tf.random.uniform(shape=[num_internal, 1], minval=-1, maxval=1, dtype=data_type)
    y_1 = tf.random.uniform(shape=[num_internal, 1], minval=-1, maxval=1, dtype=data_type)
    z_1 = tf.random.uniform(shape=[num_internal, 1], minval= 0, maxval=1, dtype=data_type)
    
    x_2 = tf.random.uniform(shape=[num_internal, 1], minval=-0.1, maxval=0.1, dtype=data_type)
    y_2 = tf.random.uniform(shape=[num_internal, 1], minval=-0.1, maxval=0.1, dtype=data_type)
    z_2 = tf.random.uniform(shape=[num_internal, 1], minval= 0.4, maxval=0.6, dtype=data_type)
    
    x_3 = tf.random.uniform(shape=[num_internal, 1], minval=-0.2, maxval=0.2, dtype=data_type)
    y_3 = tf.random.uniform(shape=[num_internal, 1], minval=-0.2, maxval=0.2, dtype=data_type)
    z_3 = tf.random.uniform(shape=[num_internal, 1], minval= 0.3, maxval=0.7, dtype=data_type)

    x_4 = tf.random.uniform(shape=[num_internal, 1], minval=-1, maxval=1, dtype=data_type)
    y_4 = tf.random.uniform(shape=[num_internal, 1], minval=-1, maxval=1, dtype=data_type)
    z_4 = tf.random.uniform(shape=[num_internal, 1], minval= 0., maxval=0.1, dtype=data_type)

    x_rand = tf.concat([x_1, x_2, x_3, x_4], axis=0)
    y_rand = tf.concat([y_1, y_2, y_3, y_4], axis=0)
    z_rand = tf.concat([z_1, z_2, z_3, z_4], axis=0)
    
    # collecting points outside of the buildings #############################################
    mask_ = tf.stop_gradient(neural_mask(x_rand, y_rand, z_rand))
    inds_is_out = tf.where(mask_ > 0.5)[:,0]

    x_ = tf.gather(x_rand, inds_is_out)
    y_ = tf.gather(y_rand, inds_is_out)
    z_ = tf.gather(z_rand, inds_is_out)
    
    with tf.GradientTape(persistent=True) as tape:
        
        tape.watch(x_)
        tape.watch(y_)
        tape.watch(z_)
        
        u_x = neural_u_x_tilde(x_, y_, z_, index)
        u_y = neural_u_y_tilde(x_, y_, z_, index)
        u_z = neural_u_z_tilde(x_, y_, z_, index)
    
    div_u = (tape.gradient(u_x, x_) + tape.gradient(u_y, y_) + tape.gradient(u_z, z_))
    
    weight_div = tf.stop_gradient(tf.clip_by_value(tf.reduce_mean(tf.abs(div_u)) / tf.abs(div_u), clip_value_min = 0.1, clip_value_max=10))
    loss_div = tf.reduce_mean(weight_div * tf.square(div_u)) 
    
    return loss_div

time_0 = time.time()
loss_div = loss_div_cal(1)
time_1 = time.time()

print(f'computation time = {(time_1 - time_0)}')
print(f'loss_div = {loss_div}')

def loss_eqn_cal(index):
    
    """
    the time difference between computing grad_z^2 and the full Laplacian is 0.01s per iteration
    """
    
    # down sample dataset #################################################
    num_internal  = 1024

    x_1 = tf.random.uniform(shape=[num_internal, 1], minval=-1, maxval=1, dtype=data_type)
    y_1 = tf.random.uniform(shape=[num_internal, 1], minval=-1, maxval=1, dtype=data_type)
    z_1 = tf.random.uniform(shape=[num_internal, 1], minval= 0, maxval=1, dtype=data_type)

    x_2 = tf.random.uniform(shape=[num_internal, 1], minval=-0.1, maxval=0.1, dtype=data_type)
    y_2 = tf.random.uniform(shape=[num_internal, 1], minval=-0.1, maxval=0.1, dtype=data_type)
    z_2 = tf.random.uniform(shape=[num_internal, 1], minval= 0.4, maxval=0.6, dtype=data_type)

    x_3 = tf.random.uniform(shape=[num_internal, 1], minval=-0.2, maxval=0.2, dtype=data_type)
    y_3 = tf.random.uniform(shape=[num_internal, 1], minval=-0.2, maxval=0.2, dtype=data_type)
    z_3 = tf.random.uniform(shape=[num_internal, 1], minval= 0.3, maxval=0.7, dtype=data_type)

    x_4 = tf.random.uniform(shape=[num_internal, 1], minval=-1, maxval=1, dtype=data_type)
    y_4 = tf.random.uniform(shape=[num_internal, 1], minval=-1, maxval=1, dtype=data_type)
    z_4 = tf.random.uniform(shape=[num_internal, 1], minval= 0., maxval=0.1, dtype=data_type)

    x_all = tf.concat([x_1, x_2, x_3, x_4], axis=0)
    y_all = tf.concat([y_1, y_2, y_3, y_4], axis=0)
    z_all = tf.concat([z_1, z_2, z_3, z_4], axis=0)

    # collecting points outside of the buildings #############################################
    mask_ = tf.stop_gradient(neural_mask(x_all, y_all, z_all))
    inds_is_out = tf.where(mask_ > 0.1)[:,0]

    x_ = tf.gather(x_all, inds_is_out)
    y_ = tf.gather(y_all, inds_is_out)
    z_ = tf.gather(z_all, inds_is_out)
    
    with tf.GradientTape(persistent=True) as tape_:

        tape_.watch(x_)
        tape_.watch(y_)
        tape_.watch(z_)

        with tf.GradientTape(persistent=True) as tape:

            tape.watch(x_)
            tape.watch(y_)
            tape.watch(z_)

            u_bar = neural_velocity(x_, y_, z_, index)

            p_ = neural_p_tilde(x_, y_, z_, index)

            u_x_ = tf.reshape(u_bar[:,0], [-1,1])
            u_y_ = tf.reshape(u_bar[:,1], [-1,1])
            u_z_ = tf.reshape(u_bar[:,2], [-1,1])

        grad_u_x_x = tape.gradient(u_x_, x_)
        grad_u_x_y = tape.gradient(u_x_, y_)
        grad_u_x_z = tape.gradient(u_x_, z_)

        grad_u_y_x = tape.gradient(u_y_, x_)
        grad_u_y_y = tape.gradient(u_y_, y_)
        grad_u_y_z = tape.gradient(u_y_, z_)

        grad_u_z_x = tape.gradient(u_z_, x_)
        grad_u_z_y = tape.gradient(u_z_, y_)
        grad_u_z_z = tape.gradient(u_z_, z_)

        nut_ = Re_inv_cal(x_, y_, z_, index)

    grad_u_x_xx = tape_.gradient(grad_u_x_x, x_)
    grad_u_x_yy = tape_.gradient(grad_u_x_y, y_)
    grad_u_x_zz = tape_.gradient(grad_u_x_z, z_)

    grad_u_y_xx = tape_.gradient(grad_u_y_x, x_)
    grad_u_y_yy = tape_.gradient(grad_u_y_y, y_)
    grad_u_y_zz = tape_.gradient(grad_u_y_z, z_)

    grad_u_z_xx = tape_.gradient(grad_u_z_x, x_)
    grad_u_z_yy = tape_.gradient(grad_u_z_y, y_)
    grad_u_z_zz = tape_.gradient(grad_u_z_z, z_)
    
    grad_p_x = tape.gradient(p_, x_)
    grad_p_y = tape.gradient(p_, y_)
    grad_p_z = tape.gradient(p_, z_)

    grad_nut_x = tape_.gradient(nut_, x_)
    grad_nut_y = tape_.gradient(nut_, y_)
    grad_nut_z = tape_.gradient(nut_, z_)

    u_del_u_x = u_x_ * grad_u_x_x + u_y_ * grad_u_x_y + u_z_ * grad_u_x_z
    u_del_u_y = u_x_ * grad_u_y_x + u_y_ * grad_u_y_y + u_z_ * grad_u_y_z
    u_del_u_z = u_x_ * grad_u_z_x + u_y_ * grad_u_z_y + u_z_ * grad_u_z_z

    viscous_x = ( nut_ * (grad_u_x_xx + grad_u_x_yy + grad_u_x_zz) + 
                  grad_nut_x * (grad_u_x_x + grad_u_x_x)           + 
                  grad_nut_y * (grad_u_x_y + grad_u_y_x)           + 
                  grad_nut_z * (grad_u_x_z + grad_u_z_x)           )

    viscous_y = ( nut_ * (grad_u_y_xx + grad_u_y_yy + grad_u_y_zz) + 
                  grad_nut_x * (grad_u_y_x + grad_u_x_y)           + 
                  grad_nut_y * (grad_u_y_y + grad_u_y_y)           + 
                  grad_nut_z * (grad_u_y_z + grad_u_z_y)           )

    viscous_z = ( nut_ * (grad_u_z_xx + grad_u_z_yy + grad_u_z_zz) + 
                  grad_nut_x * (grad_u_z_x + grad_u_x_z)           + 
                  grad_nut_y * (grad_u_z_y + grad_u_y_z)           + 
                  grad_nut_z * (grad_u_z_z + grad_u_z_z)           )

    Eq_x = (viscous_x - grad_p_x - u_del_u_x) 
    Eq_y = (viscous_y - grad_p_y - u_del_u_y)
    Eq_z = (viscous_z - grad_p_z - u_del_u_z) 

    weight_eqn_x = tf.stop_gradient(tf.clip_by_value(tf.reduce_mean(tf.abs(Eq_x)) / tf.abs(Eq_x), clip_value_min = 0.1, clip_value_max=10))
    weight_eqn_y = tf.stop_gradient(tf.clip_by_value(tf.reduce_mean(tf.abs(Eq_y)) / tf.abs(Eq_y), clip_value_min = 0.1, clip_value_max=10))
    weight_eqn_z = tf.stop_gradient(tf.clip_by_value(tf.reduce_mean(tf.abs(Eq_z)) / tf.abs(Eq_z), clip_value_min = 0.1, clip_value_max=10))
 
    loss_eqn_x = tf.reduce_mean(weight_eqn_x * tf.square(Eq_x))
    loss_eqn_y = tf.reduce_mean(weight_eqn_y * tf.square(Eq_y)) 
    loss_eqn_z = tf.reduce_mean(weight_eqn_z * tf.square(Eq_z)) 
    
    return loss_eqn_x, loss_eqn_y, loss_eqn_z

time_0 = time.time()
loss_eqn_x, loss_eqn_y, loss_eqn_z = loss_eqn_cal(1)
time_1 = time.time()

print(f'computation time = {(time_1 - time_0)}')
print(f'loss_eqn_x = {loss_eqn_x}, loss_eqn_y = {loss_eqn_y}, loss_eqn_z = {loss_eqn_z},')

optimizer_NN_mask = tf.keras.optimizers.Adam(learning_rate= 3*10**-4)

@tf.function
def train_mask(index):
     
    with tf.GradientTape() as tape:
        
        loss_mask = loss_fit_mask_cal(index)
    
    grad_mask = tape.gradient(loss_mask, NN_mask.trainable_variables)
    optimizer_NN_mask.apply_gradients( zip(grad_mask, NN_mask.trainable_variables) )
    
    return loss_mask

time_0 = time.time()
loss_mask = train_mask(1)
time_1 = time.time()

print(f'computation time = {(time_1 - time_0)}')
print(f'loss_mask = {loss_mask}')

time_0 = time.time()
for i in range(50000):
    loss_mask = train_mask(1)
    if loss_mask < 0.05:
        optimizer_NN_mask.lr.assign(3*10**-5)
    if loss_mask < 0.01:
        break
time_1 = time.time()

print(f'computation time = {(time_1 - time_0)/50000}')
print(f'loss_mask = {loss_mask}')

optimizer_NN_u_x_dict = dict()
optimizer_NN_u_y_dict = dict()
optimizer_NN_u_z_dict = dict()
optimizer_NN_p_dict   = dict()

optimizer_NN_nut  = tf.keras.optimizers.Adam(learning_rate = 10**-3)
optimizer_NN_mask = tf.keras.optimizers.Adam(learning_rate = 10**-5)

for index in range(1, num_dataset+1):
    optimizer_NN_u_x_dict[index] = tf.keras.optimizers.Adam(learning_rate = 10**-3)
    optimizer_NN_u_y_dict[index] = tf.keras.optimizers.Adam(learning_rate = 10**-3)
    optimizer_NN_u_z_dict[index] = tf.keras.optimizers.Adam(learning_rate = 10**-3)
    optimizer_NN_p_dict[index]   = tf.keras.optimizers.Adam(learning_rate = 10**-3)

lambda_eqn = tf.constant(1, dtype=data_type)

@tf.function
def train_step(lambda_eqn):
    
    loss_fit_u_list  = []
    loss_fit_p_list  = []
    loss_fit_nut_list  = []
    loss_bcs_list  = []
    loss_div_list  = []
    loss_eqn_x_list  = []
    loss_eqn_y_list  = []
    loss_eqn_z_list  = []
    
    for index in range(1, num_dataset+1):
        
        with tf.GradientTape(persistent=True) as tape:

            loss_fit_u    = loss_fit_u_cal(index)
            loss_fit_p    = loss_fit_p_cal(index)
            loss_fit_nut  = loss_fit_nut_cal(index)
            loss_mask = loss_fit_mask_cal(index)
            loss_div  = loss_div_cal(index)
            loss_bcs  = loss_bcs_cal(index)
            loss_mask = loss_fit_mask_cal(index)
            
            loss_eqn_x, loss_eqn_y, loss_eqn_z  = loss_eqn_cal(index)
            
            loss_eqn = loss_eqn_x + loss_eqn_y + loss_eqn_z

            loss_ = loss_fit_u + loss_fit_p + loss_bcs + loss_fit_nut + loss_div + lambda_eqn * loss_eqn + loss_mask    

        grad_u_x  = tape.gradient(loss_, NN_u_x_dict[index].trainable_variables)
        grad_u_y  = tape.gradient(loss_, NN_u_y_dict[index].trainable_variables)
        grad_u_z  = tape.gradient(loss_, NN_u_z_dict[index].trainable_variables)
        grad_p    = tape.gradient(loss_, NN_p_dict[index].trainable_variables)
        grad_nut  = tape.gradient(loss_, NN_nut.trainable_variables)
        grad_mask = tape.gradient(loss_, NN_mask.trainable_variables)
        
        grad_u_x_clip  = [tf.clip_by_norm(grads, 0.1) for grads in grad_u_x]
        grad_u_y_clip  = [tf.clip_by_norm(grads, 0.1) for grads in grad_u_y]
        grad_u_z_clip  = [tf.clip_by_norm(grads, 0.1) for grads in grad_u_z]
        grad_p_clip    = [tf.clip_by_norm(grads, 0.1) for grads in grad_p]
        grad_nut_clip  = [tf.clip_by_norm(grads, 0.1) for grads in grad_nut]
        grad_mask_clip  = [tf.clip_by_norm(grads, 0.1) for grads in grad_mask]

        # perform gradient descent
        optimizer_NN_u_x_dict[index].apply_gradients( zip(grad_u_x_clip, NN_u_x_dict[index].trainable_variables) )
        optimizer_NN_u_y_dict[index].apply_gradients( zip(grad_u_y_clip, NN_u_y_dict[index].trainable_variables) )
        optimizer_NN_u_z_dict[index].apply_gradients( zip(grad_u_z_clip, NN_u_z_dict[index].trainable_variables) )
        optimizer_NN_p_dict[index].apply_gradients( zip(grad_p_clip, NN_p_dict[index].trainable_variables) )
        optimizer_NN_nut.apply_gradients( zip(grad_nut_clip, NN_nut.trainable_variables) )
        optimizer_NN_mask.apply_gradients( zip(grad_mask_clip, NN_mask.trainable_variables) )
        
        loss_fit_u_list.append(loss_fit_u)
        loss_fit_p_list.append(loss_fit_p)
        loss_fit_nut_list.append(loss_fit_nut)
        loss_bcs_list.append(loss_bcs)
        loss_div_list.append(loss_div)
        loss_eqn_x_list.append(loss_eqn_x)
        loss_eqn_y_list.append(loss_eqn_y)
        loss_eqn_z_list.append(loss_eqn_z)

    loss_fit_u_   = tf.reduce_mean(loss_fit_u_list)
    loss_fit_p_   = tf.reduce_mean(loss_fit_p_list)
    loss_fit_nut_ = tf.reduce_mean(loss_fit_nut_list)
    loss_div_   = tf.reduce_mean(loss_div_list)
    loss_bcs_   = tf.reduce_mean(loss_bcs_list)
    loss_eqn_x_ = tf.reduce_mean(loss_eqn_x_list)
    loss_eqn_y_ = tf.reduce_mean(loss_eqn_y_list)
    loss_eqn_z_ = tf.reduce_mean(loss_eqn_z_list)
    
    return loss_fit_u_, loss_fit_p_, loss_fit_nut_, loss_bcs_, loss_div_, loss_eqn_x_, loss_eqn_y_, loss_eqn_z_

time_0 = time.time()
loss_fit_u, loss_fit_p, loss_fit_nut, loss_bcs, loss_div, loss_eqn_x, loss_eqn_y, loss_eqn_z = train_step(lambda_eqn)
time_1 = time.time()

print(f'computation time = {time_1 - time_0}')
print(f'loss_fit_u = {loss_fit_u}, loss_fit_p = {loss_fit_p}, loss_fit_nut = {loss_fit_nut}, loss_bcs = {loss_bcs}, loss_div = {loss_div}, loss_eqn_x = {loss_eqn_x}, loss_eqn_y = {loss_eqn_y},  loss_eqn_z = {loss_eqn_z}')
print()

time_0 = time.time()
for i in range(100):
    loss_fit_u, loss_fit_p, loss_fit_nut, loss_bcs, loss_div, loss_eqn_x, loss_eqn_y, loss_eqn_z = train_step(lambda_eqn)
time_1 = time.time()

print(f'computation time = {(time_1 - time_0)/100}')
print(f'loss_fit_u = {loss_fit_u}, loss_fit_p = {loss_fit_p}, loss_fit_nut = {loss_fit_nut}, loss_bcs = {loss_bcs}, loss_div = {loss_div}, loss_eqn_x = {loss_eqn_x}, loss_eqn_y = {loss_eqn_y},  loss_eqn_z = {loss_eqn_z}')
print()

time_0 = time.time()

index_vis = 1

learning_rate = 3*10**-4

optimizer_NN_mask.lr.assign(learning_rate)
optimizer_NN_nut.lr.assign(learning_rate/4)

for index in range(1, num_dataset+1):
    optimizer_NN_u_x_dict[index].lr.assign(learning_rate)
    optimizer_NN_u_y_dict[index].lr.assign(learning_rate)
    optimizer_NN_u_z_dict[index].lr.assign(learning_rate)
    optimizer_NN_p_dict[index].lr.assign(learning_rate)

loss_fit_u_list  = []
loss_fit_p_list  = []
loss_fit_nut_list  = []
loss_bcs_list  = []
loss_div_list  = []
loss_eqn_x_list  = []
loss_eqn_y_list  = []
loss_eqn_z_list  = []

lambda_eqn = tf.constant(1, dtype=data_type)

for iter_ in range(15001):
    
    loss_fit_u, loss_fit_p, loss_fit_nut, loss_bcs, loss_div, loss_eqn_x, loss_eqn_y, loss_eqn_z = train_step(lambda_eqn)
    
    loss_fit_u_list.append(loss_fit_u)
    loss_fit_p_list.append(loss_fit_p)
    loss_fit_nut_list.append(loss_fit_nut)
    loss_bcs_list.append(loss_bcs)
    loss_div_list.append(loss_div)
    loss_eqn_x_list.append(loss_eqn_x)
    loss_eqn_y_list.append(loss_eqn_y)
    loss_eqn_z_list.append(loss_eqn_z)
    
    if iter_ % 250 ==0:
        print(f'iter = {iter_}, lambda_eqn = {lambda_eqn}: loss_fit_u = {loss_fit_u_list[-1]}, loss_fit_p = {loss_fit_p_list[-1]}, loss_fit_nut = {loss_fit_nut_list[-1]}, loss_bcs = {loss_bcs_list[-1]}, loss_div = {loss_div_list[-1]}, loss_eqn_x = {loss_eqn_x_list[-1]}, loss_eqn_y = {loss_eqn_y_list[-1]}, loss_eqn_z = {loss_eqn_z_list[-1]}')
    
    if iter_ % 10000 == 0 and iter_ != 0 and lambda_eqn < 1:
            lambda_eqn *= 1.2
    
    if iter_ % 2000 == 0 and iter_ != 0 and learning_rate > 10**-6:
        
        learning_rate /= 1.5
        
        optimizer_NN_mask.lr.assign(learning_rate/16)
        optimizer_NN_nut.lr.assign(learning_rate/4)

        for index in range(1, num_dataset+1):
            optimizer_NN_u_x_dict[index].lr.assign(learning_rate)
            optimizer_NN_u_y_dict[index].lr.assign(learning_rate)
            optimizer_NN_u_z_dict[index].lr.assign(learning_rate)
            optimizer_NN_p_dict[index].lr.assign(learning_rate/16)

    if iter_ % 1000 == 0:
        
        plt_velocity_xz(1)
        plt_velocity_yz(1)
        plt_velocity_xy(1)
#         plt_vorticity_xz(1)
#         plt_vorticity_yz(1)
#         plt_vorticity_xy(1)
        plot_eqn(y_loc=0, index=1)
        plot_vert(0, 1)
        plot_eqn(y_loc=-0.2, index=1)
        plot_vert(-0.2, 1)
#         plt_dev_xz(1)
#         plt_dev_yz(1)
#         plt_dev_xy(1)
        plt_dev(loss_fit_u_list, loss_fit_p_list, loss_fit_nut_list, loss_eqn_x_list, loss_eqn_y_list, loss_eqn_z_list, index_vis)
        
time_1 = time.time()

print(f'Total time for training = {time_1- time_0}')

plt_velocity_xz(1)
plt_velocity_yz(1)
plt_velocity_xy(1)
plot_eqn(y_loc=0, index=1)
plot_vert(0, 1)
plot_eqn(y_loc=-0.2, index=1)
plot_vert(-0.2, 1)
plt_dev(loss_fit_u_list, loss_fit_p_list, loss_fit_nut_list, loss_eqn_x_list, loss_eqn_y_list, loss_eqn_z_list, index_vis)

save_nn_velocity(NN_u_x_dict, NN_u_y_dict, NN_u_z_dict, NN_p_dict)
save_nn_sigma(NN_nut, NN_mask)

index = 1

# compute neural velocities ######################################################################
y_ = np.linspace(-1, 1, 256)
z_ = np.linspace( 0, 1, 256)

yy_, zz_ = np.meshgrid(y_, z_)
xx_ = np.full(np.shape(zz_), 0)

velocities_ = neural_velocity(xx_, yy_, zz_, index)

u_x_ = U_dict[index] * tf.reshape(velocities_[:,0], np.shape(xx_))
u_y_ = U_dict[index] * tf.reshape(velocities_[:,1], np.shape(yy_))
u_z_ = U_dict[index] * tf.reshape(velocities_[:,2], np.shape(zz_))

u_abs_ = tf.sqrt(u_x_**2 + u_y_**2 + u_z_**2)

# find buildings ###################################################################################
theta_ = tf.cast(tf.reshape(np.linspace(0, 2*np.pi, 128), [-1,1]), dtype=data_type)
r_     = tf.constant(5, dtype=data_type) 

y_sphere = (tf.cos(theta_) * r_) / L
z_sphere = (50  + (tf.sin(theta_) * r_)) / h

# plot results #####################################################################################

fig, ax = plt.subplots(1,1, figsize=[11,8], dpi = 250)

im_3 = ax.contourf(yy_, zz_, u_abs_, extend='both', cmap = plt.cm.RdBu_r, levels = np.linspace(0,4,128))
cbar_0 = fig.colorbar(im_3, ax=ax, orientation="horizontal", ticks=np.linspace(0,4,4, endpoint=True))
ax.scatter(y_sphere, z_sphere, s=1, color="black")
ax.set_xlabel = ('x')
ax.set_ylabel = ('z')
ax.set_title(f'|u| on the yz plane', y=1.05)

# plt.gca().invert_xaxis()
plt.tight_layout(pad = 2)
plt.show()
