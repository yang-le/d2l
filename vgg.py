import tensorflow as tf
import d2l


def vgg_block(num_convs, num_channels):
    blk = tf.keras.models.Sequential()
    for _ in range(num_convs):
        blk.add(tf.keras.layers.Conv2D(num_channels, kernel_size=3, padding='same', activation='relu'))
    blk.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    return blk


def vgg(conv_arch):
    net = tf.keras.models.Sequential()
    # 卷积层部分
    for (num_convs, num_channels) in conv_arch:
        net.add(vgg_block(num_convs, num_channels))
    # 全连接层部分
    net.add(tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10)
    ]))
    return net


if __name__ == "__main__":
    tf.compat.v1.enable_eager_execution()

    conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
    # net = vgg(conv_arch)

    # X = tf.random.uniform((1, 224, 224, 1))
    # for blk in net.layers:
    #     X = blk(X)
    #     print(blk.__class__.__name__,'output shape:\t', X.shape)

    ratio = 4
    small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]

    # 回想一下，这必须是一个将被放入“d2l.train_ch6()”的函数，为了利用我们现有的CPU/GPU设备，这样模型构建/编译需要在strategy.scope()中
    def net():
        return vgg(small_conv_arch)

    lr, num_epochs, batch_size = 0.05, 10, 128
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
    strategy = tf.distribute.OneDeviceStrategy('/device:DML:0')
    # strategy = tf.distribute.MirroredStrategy(['/device:DML:0', '/device:DML:1'])
    d2l.train_on_device(net, train_iter, test_iter, num_epochs, lr, strategy, 'DML')
