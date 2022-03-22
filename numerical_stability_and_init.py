import tensorflow as tf
import d2l


if __name__ == "__main__":
    x = tf.Variable(tf.range(-8.0, 8.0, 0.1))
    with tf.GradientTape() as t:
        y = tf.nn.sigmoid(x)
    d2l.plot(x.numpy(), [y.numpy(), t.gradient(y, x).numpy()], legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
    d2l.plt.show()

    M = tf.random.normal((4, 4))
    print('一个矩阵 \n', M)
    for i in range(100):
        M = tf.matmul(M, tf.random.normal((4, 4)))

    print('乘以100个矩阵后\n', M.numpy())
