import tensorflow as tf
import d2l


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels):
        super(ConvBlock, self).__init__()
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv = tf.keras.layers.Conv2D(filters=num_channels, kernel_size=(3, 3), padding='same')
        self.listLayers = [self.bn, self.relu, self.conv]

    def call(self, x):
        y = x
        for layer in self.listLayers:
            y = layer(y)
        y = tf.keras.layers.concatenate([x, y], axis=-1)
        return y


class DenseBlock(tf.keras.layers.Layer):
    def __init__(self, num_convs, num_channels):
        super(DenseBlock, self).__init__()
        self.listLayers = []
        for _ in range(num_convs):
            self.listLayers.append(ConvBlock(num_channels))

    def call(self, x):
        for layer in self.listLayers:
            x = layer(x)
        return x


class TransitionBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels, **kwargs):
        super(TransitionBlock, self).__init__(**kwargs)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv = tf.keras.layers.Conv2D(num_channels, kernel_size=1)
        self.avg_pool = tf.keras.layers.AvgPool2D(pool_size=2, strides=2)

    def call(self, x):
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.conv(x)
        return self.avg_pool(x)


def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, 7, strides=2, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
        DenseBlock(4, 32),
        TransitionBlock(96),
        DenseBlock(4, 32),
        TransitionBlock(112),
        DenseBlock(4, 32),
        TransitionBlock(120),
        DenseBlock(4, 32),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.GlobalAvgPool2D(),
        tf.keras.layers.Dense(units=10)
    ])


if __name__ == "__main__":
    tf.compat.v1.enable_eager_execution()
    # X = tf.random.uniform(shape=(1, 224, 224, 1))
    # for layer in net().layers:
    #     X = layer(X)
    #     print(layer.__class__.__name__,'output shape:\t', X.shape)

    lr, num_epochs, batch_size = 0.1, 10, 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
    strategy = tf.distribute.OneDeviceStrategy('/device:DML:0')
    # strategy = tf.distribute.MirroredStrategy(['/device:DML:0', '/device:DML:1'])
    d2l.train_on_device(net, train_iter, test_iter, num_epochs, lr, strategy, 'DML')
