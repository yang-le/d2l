import tensorflow as tf
import d2l


class Inception(tf.keras.Model):
    # c1--c4是每条路径的输出通道数
    def __init__(self, c1, c2, c3, c4):
        super().__init__()
        # 线路1，单1x1卷积层
        self.p1_1 = tf.keras.layers.Conv2D(c1, 1, activation='relu')
        # 线路2，1x1卷积层后接3x3卷积层
        self.p2_1 = tf.keras.layers.Conv2D(c2[0], 1, activation='relu')
        self.p2_2 = tf.keras.layers.Conv2D(c2[1], 3, padding='same', activation='relu')
        # 线路3，1x1卷积层后接5x5卷积层
        self.p3_1 = tf.keras.layers.Conv2D(c3[0], 1, activation='relu')
        self.p3_2 = tf.keras.layers.Conv2D(c3[1], 5, padding='same', activation='relu')
        # 线路4，3x3最大汇聚层后接1x1卷积层
        self.p4_1 = tf.keras.layers.MaxPool2D(3, 1, padding='same')
        self.p4_2 = tf.keras.layers.Conv2D(c4, 1, activation='relu')

    def call(self, x):
        p1 = self.p1_1(x)
        p2 = self.p2_2(self.p2_1(x))
        p3 = self.p3_2(self.p3_1(x))
        p4 = self.p4_2(self.p4_1(x))
        # 在通道维度上连结输出
        return tf.keras.layers.Concatenate()([p1, p2, p3, p4])


def b1():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, 7, strides=2, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')
    ])


def b2():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, 1, activation='relu'),
        tf.keras.layers.Conv2D(192, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')
    ])


def b3():
    return tf.keras.models.Sequential([
        Inception(64, (96, 128), (16, 32), 32),
        Inception(128, (128, 192), (32, 96), 64),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')
    ])


def b4():
    return tf.keras.models.Sequential([
        Inception(192, (96, 208), (16, 48), 64),
        Inception(160, (112, 224), (24, 64), 64),
        Inception(128, (128, 256), (24, 64), 64),
        Inception(112, (144, 288), (32, 64), 64),
        Inception(256, (160, 320), (32, 128), 128),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')
    ])


def b5():
    return tf.keras.models.Sequential([
        Inception(256, (160, 320), (32, 128), 128),
        Inception(384, (192, 384), (48, 128), 128),
        tf.keras.layers.GlobalAvgPool2D(),
        tf.keras.layers.Flatten()
    ])


# “net”必须是一个将被传递给“d2l.train()”的函数。
# 为了利用我们现有的CPU/GPU设备，这样模型构建/编译需要在“strategy.scope()”
def net():
    return tf.keras.models.Sequential([
        b1(), b2(), b3(), b4(), b5(),
        tf.keras.layers.Dense(10)
    ])


if __name__ == "__main__":
    tf.compat.v1.enable_eager_execution()
    # X = tf.random.uniform(shape=(1, 96, 96, 1))
    # for layer in net().layers:
    #     X = layer(X)
    #     print(layer.__class__.__name__, 'output shape:\t', X.shape)

    lr, num_epochs, batch_size = 0.1, 10, 128
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
    strategy = tf.distribute.OneDeviceStrategy('/device:DML:0')
    # strategy = tf.distribute.MirroredStrategy(['/device:DML:0', '/device:DML:1'])
    d2l.train_on_device(net, train_iter, test_iter, num_epochs, lr, strategy, 'DML')
