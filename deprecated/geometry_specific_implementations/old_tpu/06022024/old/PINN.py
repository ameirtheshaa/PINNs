from definitions import *
from training import *
from plotting import *

class PINN(Model):
    def __init__(self, input_params, output_params, hidden_layers, neurons_per_layer, activation, use_batch_norm, dropout_rate):
        super(PINN, self).__init__()

        self.input_params = input_params
        self.output_params = output_params
        self.hidden_layers = hidden_layers
        self.neurons_per_layer = neurons_per_layer
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate

        # Build model
        self.build_model()

    def build_model(self):
        self.layers_list = []

        # Input layer
        self.layers_list.append(Dense(self.neurons_per_layer[0], activation=self.activation, input_shape=(len(self.input_params),)))

        # Hidden layers
        for i in range(1, self.hidden_layers):
            if self.use_batch_norm:
                self.layers_list.append(BatchNormalization())
            if self.dropout_rate > 0:
                self.layers_list.append(Dropout(self.dropout_rate))
            self.layers_list.append(Dense(self.neurons_per_layer[i], activation=self.activation))

        # Output layer
        self.layers_list.append(Dense(len(self.output_params)))

    def call(self, inputs):
        x = inputs
        for layer in self.layers_list:
            x = layer(x)
        return x

    def get_config(self):
        config = super(PINN, self).get_config()
        config.update({
            "input_params": self.input_params,
            "output_params": self.output_params,
            "hidden_layers": self.hidden_layers,
            "neurons_per_layer": self.neurons_per_layer,
            "activation": self.activation,
            "use_batch_norm": self.use_batch_norm,
            "dropout_rate": self.dropout_rate
        })
        return config

    def compute_data_loss(self, X, y):
        predictions = self(X)
        loss = tf.keras.losses.MeanSquaredError()(y, predictions)
        return loss

    def compute_boundary_loss(self, X_boundary, y_boundary_known, output_params):

        def find_velocity_indices(output_params):
            velocity_labels = ['Velocity:0', 'Velocity:1', 'Velocity:2']
            velocity_indices = [output_params.index(label) for label in velocity_labels]
            return velocity_indices

        velocity_indices = find_velocity_indices(output_params)
        start_idx, end_idx = min(velocity_indices), max(velocity_indices) + 1

        boundary_predictions = self(X_boundary)
        velocities = boundary_predictions[:, start_idx:end_idx]
        
        criterion = tf.keras.losses.MeanSquaredError()
        inlet_loss = criterion(velocities, y_boundary_known)
        
        return inlet_loss

    def extract_parameters(self, X, input_params):
        extracted_params = []
        for param in input_params:
            if param == 'Points:0':
                extracted_params.append(X[:, 0:1])
            elif param == 'Points:1':
                extracted_params.append(X[:, 1:2])
            elif param == 'Points:2':
                extracted_params.append(X[:, 2:3])
            elif param == 'cos(WindAngle)':
                extracted_params.append(X[:, 3:4])
            elif param == 'sin(WindAngle)':
                extracted_params.append(X[:, 4:5])
            # Add more conditions here for other potential parameters
            else:
                raise ValueError(f"Unknown parameter: {param}")

        return extracted_params

    def extract_output_parameters(self, Y, output_params):
        extracted_output_params = []
        for param in output_params:
            if param == 'Pressure':
                extracted_output_params.append(Y[:, 0:1])
            elif param == 'Velocity:0':
                extracted_output_params.append(Y[:, 1:2])
            elif param == 'Velocity:1':
                extracted_output_params.append(Y[:, 2:3])
            elif param == 'Velocity:2':
                extracted_output_params.append(Y[:, 3:4])
            elif param == 'TurbVisc':
                extracted_output_params.append(Y[:, 4:5])
            # Add more conditions here for other potential output parameters
            else:
                raise ValueError(f"Unknown output parameter: {param}")

        return extracted_output_params

    def continuity(self, cos_wind_angle, sin_wind_angle, x, y, z, output_params):
        # Stack the input data
        input_data = tf.concat([x, y, z, cos_wind_angle, sin_wind_angle], axis=1)

        with tf.GradientTape(persistent=True) as tape:
            # Ensure x, y, and z are being watched by the gradient tape
            tape.watch([x, y, z])

            output = self(input_data)  # Model output
            extracted_outputs = self.extract_output_parameters(output, output_params)

            # Create a dictionary to hold the extracted outputs
            output_dict = {param: value for param, value in zip(output_params, extracted_outputs)}

            # Extract velocity components
            u = output_dict.get('Velocity:0')
            v = output_dict.get('Velocity:1')
            w = output_dict.get('Velocity:2')

            # Compute gradients
            u_x = tape.gradient(u, x)
            v_y = tape.gradient(v, y)
            w_z = tape.gradient(w, z)

        # Continuity equation
        cont = u_x + v_y + w_z

        return cont

    def RANS(self, cos_wind_angle, sin_wind_angle, x, y, z, rho, nu, output_params):
        # Stack the input data
        input_data = tf.concat([x, y, z, cos_wind_angle, sin_wind_angle], axis=1)

        with tf.GradientTape(persistent=True) as tape:
            # Ensure x, y, and z are being watched by the gradient tape
            tape.watch([x, y, z])

            output = self(input_data)  # Model output
            extracted_outputs = self.extract_output_parameters(output, output_params)

            # Create a dictionary to hold the extracted outputs
            output_dict = {param: value for param, value in zip(output_params, extracted_outputs)}

            # Access the outputs as needed
            p = output_dict.get('Pressure')
            u = output_dict.get('Velocity:0')
            v = output_dict.get('Velocity:1')
            w = output_dict.get('Velocity:2')
            nu_t = output_dict.get('TurbVisc')

            nu_eff = nu_t + nu

            # Compute gradients
            u_x = tape.gradient(u, x)
            u_y = tape.gradient(u, y)
            u_z = tape.gradient(u, z)

            v_x = tape.gradient(v, x)
            v_y = tape.gradient(v, y)
            v_z = tape.gradient(v, z)

            w_x = tape.gradient(w, x)
            w_y = tape.gradient(w, y)
            w_z = tape.gradient(w, z)

            # Compute second order gradients
            u_xx = tape.gradient(u_x, x)
            u_yy = tape.gradient(u_y, y)
            u_zz = tape.gradient(u_z, z)

            v_xx = tape.gradient(v_x, x)
            v_yy = tape.gradient(v_y, y)
            v_zz = tape.gradient(v_z, z)

            w_xx = tape.gradient(w_x, x)
            w_yy = tape.gradient(w_y, y)
            w_zz = tape.gradient(w_z, z)

            # Compute mixed derivatives
            u_xy = tape.gradient(u_x, y)
            u_xz = tape.gradient(u_x, z)
            v_xy = tape.gradient(v_x, y)
            v_yz = tape.gradient(v_y, z)
            w_xz = tape.gradient(w_x, z)
            w_yz = tape.gradient(w_y, z)

            # Compute the gradients of p
            p_x = tape.gradient(p, x)
            p_y = tape.gradient(p, y)
            p_z = tape.gradient(p, z)

        # RANS equations
        f = (u * u_x + v * u_y + w * u_z - (1 / rho) * p_x + nu_eff * (2 * u_xx + u_yy + v_xy + u_zz + w_xz))
        g = (u * v_x + v * v_y + w * v_z - (1 / rho) * p_y + nu_eff * (v_xx + u_xy + 2 * v_yy + v_zz + w_yz))
        h = (u * w_x + v * w_y + w * w_z - (1 / rho) * p_z + nu_eff * (w_xx + u_xz + w_yy + v_yz + 2 * w_zz))

        return f, g, h

    def compute_physics_momentum_loss(self, X, input_params, output_params, rho, nu):
        extracted_inputs = self.extract_parameters(X, input_params)
        input_dict = {param: value for param, value in zip(input_params, extracted_inputs)}

        # Access the inputs using keys
        x = input_dict.get('Points:0')
        y = input_dict.get('Points:1')
        z = input_dict.get('Points:2')
        cos_wind_angle = input_dict.get('cos(WindAngle)')
        sin_wind_angle = input_dict.get('sin(WindAngle)')

        f, g, h = self.RANS(cos_wind_angle, sin_wind_angle, x, y, z, rho, nu, output_params)

        # Calculate loss for each component
        mse = tf.keras.losses.MeanSquaredError()
        loss_f = mse(f, tf.zeros_like(f))
        loss_g = mse(g, tf.zeros_like(g))
        loss_h = mse(h, tf.zeros_like(h))

        # Total momentum loss
        loss_physics_momentum = loss_f + loss_g + loss_h

        return loss_physics_momentum

    def compute_physics_cont_loss(self, X, input_params, output_params):
        extracted_inputs = self.extract_parameters(X, input_params)
        input_dict = {param: value for param, value in zip(input_params, extracted_inputs)}

        # Access the inputs using keys
        x = input_dict.get('Points:0')
        y = input_dict.get('Points:1')
        z = input_dict.get('Points:2')
        cos_wind_angle = input_dict.get('cos(WindAngle)')
        sin_wind_angle = input_dict.get('sin(WindAngle)')

        cont = self.continuity(cos_wind_angle, sin_wind_angle, x, y, z, output_params)

        # Calculate continuity loss
        mse = tf.keras.losses.MeanSquaredError()
        loss_cont = mse(cont, tf.zeros_like(cont))

        return loss_cont