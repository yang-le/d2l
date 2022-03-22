import random
import tensorflow as tf
import d2l


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = tf.constant(indices[i:min(i + batch_size, num_examples)])
        yield tf.gather(features, j), tf.gather(labels, j)


# def squared_loss(y_hat, y):
#     """均方损失"""
#     return (y_hat - y) ** 2 / 2


if __name__ == "__main__":
    number_examples = 1000
    batch_size = 10
    lr = 0.03  # 学习率
    num_epochs = 3

    true_w = tf.constant([2, -3.4])
    true_b = 4.2
    features, labels = d2l.synthetic_data(true_w, true_b, number_examples)
    print('features:', features[0], '\nlabel:', labels[0])

    for X, y in data_iter(batch_size, features, labels):
        print(X, '\n', y)
        break

    d2l.set_figsize()
    d2l.plt.scatter(features[:, (1)].numpy(), labels.numpy(), 1)
    d2l.plt.show()

    w = tf.Variable(tf.random.normal(shape=(2, 1), mean=0, stddev=0.01), trainable=True)
    # w = tf.Variable(tf.zeros(shape=(2, 1)), trainable=True)
    b = tf.Variable(tf.zeros(1), trainable=True)

    net = d2l.linreg
    loss = d2l.squared_loss

    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            with tf.GradientTape() as g:
                l = loss(net(X, w, b), y)  # X和y的小批量损失
            # 计算l关于[w,b]的梯度
            dw, db = g.gradient(l, [w, b])
            # 使用参数的梯度更新参数
            d2l.sgd([w, b], [dw, db], lr, batch_size)
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(tf.reduce_mean(train_l)):f}')

    print(f'w的估计误差: {true_w - tf.reshape(w, true_w.shape)}')
    print(f'b的估计误差: {true_b - b}')
