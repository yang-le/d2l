import tensorflow as tf
import d2l


def net():
    return tf.keras.models.Sequential([
        # 这里，我们使用一个11*11的更大窗口来捕捉对象。
        # 同时，步幅为4，以减少输出的高度和宽度。
        # 另外，输出通道的数目远大于LeNet
        tf.keras.layers.Conv2D(filters=96, kernel_size=11, strides=4, activation='relu', input_shape=(224, 224, 1)),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
        tf.keras.layers.Conv2D(filters=256, kernel_size=5, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        # 使用三个连续的卷积层和较小的卷积窗口。
        # 除了最后的卷积层，输出通道的数量进一步增加。
        # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
        tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        tf.keras.layers.Flatten(),
        # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过拟合
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        # 最后是输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
        tf.keras.layers.Dense(10)
    ])


if __name__ == "__main__":
    tf.compat.v1.enable_eager_execution()

    # X = tf.random.uniform((1, 224, 224, 1))
    # for layer in net().layers:
    #     X = layer(X)
    #     print(layer.__class__.__name__, 'output shape:\t', X.shape)

    batch_size = 128
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

    lr, num_epochs = 0.01, 10
    strategy = tf.distribute.OneDeviceStrategy('/device:DML:0')
    # strategy = tf.distribute.MirroredStrategy(['/device:DML:0', '/device:DML:1'])
    d2l.train_on_device(net, train_iter, test_iter, num_epochs, lr, strategy, 'DML')
