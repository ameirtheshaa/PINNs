from definitions import *
from PINN import *

def compute_data_loss(model, X, y):
    predictions = model(X)
    loss_per_sample = tf.keras.losses.MSE(y, predictions)
    loss = tf.reduce_mean(loss_per_sample)  # Reduce to a scalar value
    return loss