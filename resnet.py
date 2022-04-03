import tensorflow as tf
import d2l


class ResnetBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels, num_residuals, first_block=False, **kwargs):
        super(ResnetBlock, self).__init__(**kwargs)
        self.residual_layers = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                self.residual_layers.append(d2l.Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                self.residual_layers.append(d2l.Residual(num_channels))

    def call(self, X):
        for layer in self.residual_layers:
            X = layer(X)
        return X


# 回想之前我们定义一个函数，以便用它在tf.distribute.MirroredStrategy的范围，
# 来利用各种计算资源，例如gpu。
def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, 7, strides=2, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
        ResnetBlock(64, 2, first_block=True),
        ResnetBlock(128, 2),
        ResnetBlock(256, 2),
        ResnetBlock(512, 2),
        tf.keras.layers.GlobalAvgPool2D(),
        tf.keras.layers.Dense(units=10)
    ])


if __name__ == "__main__":
    tf.compat.v1.enable_eager_execution()
    # X = tf.random.uniform(shape=(1, 224, 224, 1))
    # for layer in net().layers:
    #     X = layer(X)
    #     print(layer.__class__.__name__,'output shape:\t', X.shape)

    lr, num_epochs, batch_size = 0.05, 10, 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
    strategy = tf.distribute.OneDeviceStrategy('/device:DML:0')
    # strategy = tf.distribute.MirroredStrategy(['/device:DML:0', '/device:DML:1'])
    d2l.train_on_device(net, train_iter, test_iter, num_epochs, lr, strategy, 'DML')
