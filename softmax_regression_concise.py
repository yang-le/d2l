import tensorflow as tf
import d2l

if __name__ == "__main__":
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    net = tf.keras.models.Sequential()
    net.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    weight_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
    net.add(tf.keras.layers.Dense(10, kernel_initializer=weight_initializer))

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    trainer = tf.keras.optimizers.SGD(learning_rate=.1)

    num_epochs = 10
    d2l.train(net, train_iter, test_iter, loss, num_epochs, trainer)
