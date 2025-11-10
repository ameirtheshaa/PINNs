from definitions import *

def weighting(data_loss, total_averaged_boundary_loss, cont_loss, momentum_loss):
	data_boundary_loss = data_loss + total_averaged_boundary_loss
	all_physics_loss = momentum_loss + cont_loss
	
	weight_data_boundary = data_boundary_loss/(data_boundary_loss+all_physics_loss)
	weight_physics = all_physics_loss/(data_boundary_loss+all_physics_loss)

	total_loss = weight_data_boundary*all_physics_loss + weight_physics*data_boundary_loss

	return total_loss