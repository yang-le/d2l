import math
import numpy as np
import tensorflow as tf
import d2l


def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    return p * np.exp(-0.5 / sigma**2 * (x - mu)**2)


if __name__ == "__main__":
    n = 10000
    a = tf.ones(n)
    b = tf.ones(n)

    c = tf.Variable(tf.zeros(n))
    timer = d2l.Timer()
    for i in range(n):
        c[i].assign(a[i] + b[i])
    print(f'{timer.stop():.5f} sec')

    timer.start()
    d = a + b
    print(f'{timer.stop():.5f} sec')

    # 再次使用numpy进行可视化
    x = np.arange(-7, 7, 0.01)

    # 均值和标准差对
    params = [(0, 1), (0, 2), (3, 1)]
    d2l.plot(x, [normal(x, mu, sigma) for mu, sigma in params], xlabel='x',
             ylabel='p(x)', figsize=(4.5, 2.5),
             legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])
    d2l.plt.show()
