import tensorflow as tf
import d2l


def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    # 在本情况中，所有元素都被丢弃
    if dropout == 1:
        return tf.zeros_like(X)
    # 在本情况中，所有元素都被保留
    if dropout == 0:
        return X
    mask = tf.random.uniform(shape=tf.shape(X), minval=0, maxval=1) < 1 - dropout
    return tf.cast(mask, dtype=tf.float32) * X / (1.0 - dropout)


class Net(tf.keras.Model):
    def __init__(self, num_outputs, num_hiddens1, num_hiddens2):
        super().__init__()
        self.input_layer = tf.keras.layers.Flatten()
        self.hidden1 = tf.keras.layers.Dense(num_hiddens1, activation='relu')
        self.hidden2 = tf.keras.layers.Dense(num_hiddens2, activation='relu')
        self.output_layer = tf.keras.layers.Dense(num_outputs)

    def call(self, inputs, training=None):
        x = self.input_layer(inputs)
        x = self.hidden1(x)
        # 只有在训练模型时才使用dropout
        if training:
            # 在第一个全连接层之后添加一个dropout层
            x = dropout_layer(x, dropout1)
        x = self.hidden2(x)
        if training:
            # 在第二个全连接层之后添加一个dropout层
            x = dropout_layer(x, dropout2)
        x = self.output_layer(x)
        return x


if __name__ == "__main__":
    # X = tf.reshape(tf.range(16, dtype=tf.float32), (2, 8))
    # print(X)
    # print(dropout_layer(X, 0.))
    # print(dropout_layer(X, 0.5))
    # print(dropout_layer(X, 1.))

    num_outputs, num_hiddens1, num_hiddens2 = 10, 256, 256
    dropout1, dropout2 = 0.2, 0.5

    net = Net(num_outputs, num_hiddens1, num_hiddens2)

    num_epochs, lr, batch_size = 10, 0.5, 256
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    trainer = tf.keras.optimizers.SGD(learning_rate=lr)
    d2l.train(net, train_iter, test_iter, loss, num_epochs, trainer)

    net = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_hiddens1, activation=tf.nn.relu),
        # 在第一个全连接层之后添加一个dropout层
        tf.keras.layers.Dropout(dropout1),
        tf.keras.layers.Dense(num_hiddens2, activation=tf.nn.relu),
        # 在第二个全连接层之后添加一个dropout层
        tf.keras.layers.Dropout(dropout2),
        tf.keras.layers.Dense(num_outputs)
    ])
    trainer = tf.keras.optimizers.SGD(learning_rate=lr)
    d2l.train(net, train_iter, test_iter, loss, num_epochs, trainer)
