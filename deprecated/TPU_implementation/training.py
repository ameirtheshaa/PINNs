from definitions import *
from PINN import *
from training_definitions import *

def train_model(config, model, model_file_path, log_folder, strategy=None):
    start_time = time.time()
    print(f'starting to train, time: {(time.time() - start_time):.2f} seconds')

    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, labels_train_tensor, X_train_tensor_skipped, y_train_tensor_skipped, X_test_tensor_skipped, y_test_tensor_skipped, feature_scaler, target_scaler = load_data(config)

    model_file_path = Path(model_file_path)
    chosen_optimizer_key = config["chosen_optimizer"]
    use_epoch = config["training"]["use_epochs"]
    epochs = config["training"]["num_epochs"]
    wind_angles = config["training"]["angles_to_train"]
    sma_window_size = 50
    sma_threshold = config["training"]["loss_diff_threshold"]
    consecutive_sma_threshold = config["training"]["consecutive_count_threshold"]
    recent_losses = collections.deque(maxlen=sma_window_size)
    consecutive_sma_count = 0
    early_stop = False  # Flag to break out of both loops
    current_losses = []

    chosen_optimizer_key = config["chosen_optimizer"]
    optimizer_config = config[chosen_optimizer_key]
    optimizer = get_optimizer(model, chosen_optimizer_key, optimizer_config)
    print(f'using {chosen_optimizer_key}, time: {(time.time() - start_time):.2f} seconds')

    if strategy:
        with strategy.scope():
            model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
    else:
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])

    if config["training"]["use_batches"]:
        optimal_batch_size = config["training"]["batch_size"]
        print(f'starting batch training with batch size = {optimal_batch_size}, time: {(time.time() - start_time):.2f} seconds')
    else:
        optimal_batch_size = len(X_train_tensor)
        print(f'starting training with full size = {optimal_batch_size}, time: {(time.time() - start_time):.2f} seconds')

    train_loader = create_dataset(X_train_tensor, y_train_tensor, labels_train_tensor, optimal_batch_size)

    if os.path.exists(f'{model_file_path}.json'):
        print(f"continuing from last checkpoint at {model_file_path}..., time: {(time.time() - start_time):.2f} seconds")
        try:
            if strategy:
                with strategy.scope():
                    model, epoch, training_completed = load_model(strategy, model, model_file_path, optimizer)
                    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
            else:
                model, epoch, training_completed = load_model(strategy, model, model_file_path, optimizer)
                model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
        except RuntimeError as e:
            print(f"Error loading checkpoint: {e}")
        if training_completed:
            print(f"Training has already been completed., time: {(time.time() - start_time):.2f} seconds")
            return model
        else:
            start_epoch = epoch
            print(f"continuing from last checkpoint... starting from epoch = {start_epoch}, time: {(time.time() - start_time):.2f} seconds")
    else:
        start_epoch = 1

    @tf.function
    def train_step(X_batch, y_batch):
        with tf.GradientTape() as tape:
            predictions = model(X_batch, training=True)
            total_loss, losses = calculate_total_loss(model, config, X_batch, y_batch)
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return total_loss

    def distributed_train_step(X_batch, y_batch):
        per_replica_losses = strategy.run(train_step, args=(X_batch, y_batch))
        reduced_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
        return reduced_loss

    for epoch in itertools.count(start=start_epoch):  # infinite loop, will only break if the stopping condition is met.
        if early_stop or (use_epoch and epoch > epochs):
            training_completed = True
            save_model(strategy, model, model_file_path, epoch, current_loss, training_completed, optimizer, start_time)
            break
        else:
            training_completed = False
        print_lines = 0
        for Xy_batch in train_loader:
            X_batch = Xy_batch[0]
            y_batch = Xy_batch[1]
            labels = Xy_batch[2]

            if strategy:
                with strategy.scope():
                    current_loss = distributed_train_step(X_batch, y_batch)
            else:
                current_loss = train_step(X_batch, y_batch)
            recent_losses.append(current_loss)
            if len(recent_losses) == sma_window_size:
                sma = sum(recent_losses) / sma_window_size
                if sma < sma_threshold:
                    consecutive_sma_count += 1
                    if consecutive_sma_count >= consecutive_sma_threshold:
                        print(f"SMA of loss below {sma_threshold} for {consecutive_sma_threshold} consecutive epochs at epoch {epoch}. Stopping training..., time: {(time.time() - start_time):.2f} seconds")
                        early_stop = True
                        break
                    else:
                        consecutive_sma_count = 0
                else:
                    consecutive_sma_count = 0

            if epoch % 5 == 0:
                if print_lines < 1:
                    current_elapsed_time = time.time() - start_time
                    current_elapsed_time_hours = current_elapsed_time / 3600
                    print(f'Epoch [{epoch}/{epochs if use_epoch else "infinity"}], Loss: {current_loss}; - {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
                    save_to_csv(epoch, epochs, use_epoch, current_loss, current_elapsed_time_hours, file_path=os.path.join(log_folder, 'info.csv'))
                    save_model(strategy, model, model_file_path, epoch, current_loss, training_completed, optimizer, start_time)
                    save_evaluation_results(strategy, config, model, model_file_path, log_folder, X_test_tensor, y_test_tensor, epoch, epochs, use_epoch, save_name='info_test.csv')
                    save_evaluation_results(strategy, config, model, model_file_path, log_folder, X_test_tensor_skipped, y_test_tensor_skipped, epoch, epochs, use_epoch, save_name='info_skipped.csv')
                    print_lines += 1

    end_time = time.time()
    total_elapsed_time = end_time - start_time
    total_elapsed_time_hours = total_elapsed_time / 3600
    save_model(strategy, model, model_file_path, epoch, current_loss, training_completed, optimizer, start_time)
    print(f'Training completed in {total_elapsed_time_hours:.2f} hours')

    return model