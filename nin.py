import tensorflow as tf
import d2l


def nin_block(num_channels, kernel_size, strides, padding):
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(num_channels, kernel_size, strides=strides, padding=padding, activation='relu'),
        tf.keras.layers.Conv2D(num_channels, kernel_size=1, activation='relu'),
        tf.keras.layers.Conv2D(num_channels, kernel_size=1, activation='relu')
    ])


def net():
    return tf.keras.models.Sequential([
        nin_block(96, kernel_size=11, strides=4, padding='valid'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        nin_block(256, kernel_size=5, strides=1, padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        nin_block(384, kernel_size=3, strides=1, padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        tf.keras.layers.Dropout(0.5),
        # 标签类别数是10
        nin_block(10, kernel_size=3, strides=1, padding='same'),
        tf.keras.layers.GlobalAvgPool2D(),
        tf.keras.layers.Reshape((1, 1, 10)),
        # 将四维的输出转成二维的输出，其形状为(批量大小,10)
        tf.keras.layers.Flatten()
    ])


if __name__ == "__main__":
    tf.compat.v1.enable_eager_execution()
    # X = tf.random.uniform((1, 224, 224, 1))
    # for layer in net().layers:
    #     X = layer(X)
    #     print(layer.__class__.__name__,'output shape:\t', X.shape)

    lr, num_epochs, batch_size = 0.1, 10, 128
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
    strategy = tf.distribute.OneDeviceStrategy('/device:DML:0')
    # strategy = tf.distribute.MirroredStrategy(['/device:DML:0', '/device:DML:1'])
    d2l.train_on_device(net, train_iter, test_iter, num_epochs, lr, strategy, 'DML')
