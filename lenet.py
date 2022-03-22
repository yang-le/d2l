import tensorflow as tf
import d2l


def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation='sigmoid', padding='same', input_shape=(28, 28, 1)),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5, activation='sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='sigmoid'),
        tf.keras.layers.Dense(84, activation='sigmoid'),
        tf.keras.layers.Dense(10)
    ])


if __name__ == "__main__":
    # tf.debugging.set_log_device_placement(True)
    tf.compat.v1.enable_eager_execution()

    X = tf.random.uniform((1, 28, 28, 1))
    # for layer in net().layers:
    #     X = layer(X)
    #     print(layer.__class__.__name__, 'output shape: \t', X.shape)

    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

    lr, num_epochs = 0.9, 10
    d2l.train_on_device(net, train_iter, test_iter, num_epochs, lr, d2l.try_dml())
