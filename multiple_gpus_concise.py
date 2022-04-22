import tensorflow as tf
import d2l


def net():
    return d2l.resnet18(10)


if __name__ == "__main__":
    tf.compat.v1.enable_eager_execution()

    # num_epochs, batch_size, lr = 10, 256, 0.1
    # train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    # strategy = tf.distribute.OneDeviceStrategy('/device:DML:0')
    # d2l.train_on_device(net, train_iter, test_iter, num_epochs, lr, strategy, 'dml')

    num_epochs, batch_size, lr = 10, 512, 0.2
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    strategy = tf.distribute.MirroredStrategy(['/device:DML:0', '/device:DML:1'])
    d2l.train_on_device(net, train_iter, test_iter, num_epochs, lr, strategy, 'dml')
