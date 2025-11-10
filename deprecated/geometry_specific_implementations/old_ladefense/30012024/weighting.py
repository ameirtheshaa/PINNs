from definitions import *

def adaptive_weighting(epoch, max_epochs, initial_weight, final_weight):
    # Linearly interpolate the weight between the initial and final values based on the current epoch
    weight = initial_weight + (final_weight - initial_weight) * (epoch / max_epochs)
    return weight


def adaptive_loss(config, loss_dict, epoch, max_epochs):
    # Extract individual loss components from the dictionary
    data_loss = loss_dict.get('data_loss', 0)
    inlet_loss = loss_dict.get('inlet_loss', 0)
    no_slip_loss = loss_dict.get('no_slip_loss', 0)
    cont_loss = loss_dict.get('cont_loss', 0)
    momentum_loss = loss_dict.get('momentum_loss', 0)

    # List of individual loss components
    loss_components = [data_loss, inlet_loss, no_slip_loss, cont_loss, momentum_loss]

    initial_weight_data = config["loss_components"]["adaptive_weighting_initial_weight"]
    final_weight_data = config["loss_components"]["adaptive_weighting_final_weight"]

    weight_data = adaptive_weighting(epoch, max_epochs, initial_weight_data, final_weight_data)

    # Check for active physical loss components (non-zero)
    active_physical_losses = [loss for loss in loss_components[1:] if loss != 0]
    num_active_physical = len(active_physical_losses)

    # Calculate the weight for each physical loss component
    weight_physical = (1 - weight_data) / num_active_physical if num_active_physical > 0 else 0

    # Calculate total loss with adaptive weighting
    total_loss = weight_data * data_loss if data_loss != 0 else 0
    for physical_loss in active_physical_losses:
        total_loss += weight_physical * physical_loss

    return total_loss

def gradient_magnitude_scaling(loss_gradients, scaling_factor=1.0, epsilon=1e-6):
    inverse_magnitudes = []
    for grad in loss_gradients:
        if grad is not None and torch.norm(grad) > epsilon:  # Check if gradient is not zero
            inverse_magnitude = scaling_factor / (torch.norm(grad) + epsilon)
        else:
            inverse_magnitude = 0  # Assign no weight to zero gradients
        inverse_magnitudes.append(inverse_magnitude)

    total = sum(inverse_magnitudes)
    normalized_weights = [w / total if total > epsilon else 0 for w in inverse_magnitudes]

    return normalized_weights

def gradient_magnitude_gradients(loss_components, model, optimizer):
    loss_gradients = []
    for loss in loss_components:
        if loss != 0:  # Check if the loss component is active (non-zero)
            optimizer.zero_grad()  # Zero out gradients before the backward pass
            loss.backward(retain_graph=True)  # Compute gradients for the active loss component
            loss_grad = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])
            loss_gradients.append(loss_grad)
        else:
            loss_grad = torch.zeros_like(next(model.parameters()).view(-1))
            loss_gradients.append(loss_grad)

    return loss_gradients

def gradient_magnitude_loss(loss_dict, model, optimizer):
    """
    Calculates the total loss using gradient magnitude scaling.
    The function first computes the gradients of each loss component.
    Then, it scales the loss components based on the magnitude of their gradients.
    """

    # Extract individual loss components from the dictionary
    data_loss = loss_dict.get('data_loss', 0)
    inlet_loss = loss_dict.get('inlet_loss', 0)
    no_slip_loss = loss_dict.get('no_slip_loss', 0)
    cont_loss = loss_dict.get('cont_loss', 0)
    momentum_loss = loss_dict.get('momentum_loss', 0)

    # List of individual loss components
    loss_components = [data_loss, inlet_loss, no_slip_loss, cont_loss, momentum_loss]

    loss_gradients = gradient_magnitude_gradients(loss_components, model, optimizer=optimizer)

    # Get the adaptive weights based on gradient magnitudes
    weights = gradient_magnitude_scaling(loss_gradients)

    # Initialize total loss
    total_loss = 0

    # Calculate the total weighted loss
    for weight, loss in zip(weights, loss_components):
        if weight > 1e-6:  # Check if the weight is significant
            total_loss += weight * loss

    return total_loss

def weighting(config, loss_components, epoch, max_epochs, model, optimizer):
    if config['loss_components']['use_weighting']:
        if config['loss_components']['weighting_scheme'] == 'adaptive_weighting':
            return adaptive_loss(config, loss_components, epoch, max_epochs)
        elif config['loss_components']['weighting_scheme'] == 'gradient_magnitude':
            return gradient_magnitude_loss(loss_components, model, optimizer=optimizer)
        else:
            raise ValueError("Unknown weighting scheme")
    else:
        return sum(loss_components.values())