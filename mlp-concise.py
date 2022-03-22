import tensorflow as tf
import d2l


if __name__ == "__main__":
    net = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    batch_size = 256
    lr = 0.1
    num_epochs = 10

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    trainer = tf.keras.optimizers.SGD(learning_rate=lr)

    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    d2l.train(net, train_iter, test_iter, loss, num_epochs, trainer)
