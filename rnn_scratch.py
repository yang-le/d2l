import math
import tensorflow as tf
import d2l


def get_params(vocab_size, num_hiddens):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return tf.random.normal(shape=shape, stddev=0.01, mean=0, dtype=tf.float32)

    # 隐藏层参数
    W_xh = tf.Variable(normal((num_inputs, num_hiddens)), dtype=tf.float32)
    W_hh = tf.Variable(normal((num_hiddens, num_hiddens)), dtype=tf.float32)
    b_h = tf.Variable(tf.zeros(num_hiddens), dtype=tf.float32)
    # 输出层参数
    W_hq = tf.Variable(normal((num_hiddens, num_outputs)), dtype=tf.float32)
    b_q = tf.Variable(tf.zeros(num_outputs), dtype=tf.float32)
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    return params


def init_rnn_state(batch_size, num_hiddens):
    return (tf.zeros((batch_size, num_hiddens)),)


def rnn(inputs, state, params):
    # inputs的形状：(时间步数量，批量大小，词表大小)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # X的形状：(批量大小，词表大小)
    for X in inputs:
        X = tf.reshape(X, [-1, W_xh.shape[0]])
        H = tf.tanh(tf.matmul(X, W_xh) + tf.matmul(H, W_hh) + b_h)
        Y = tf.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return tf.concat(outputs, axis=0), (H,)


class RNNModelScratch:
    """从零开始实现的循环神经网络模型"""
    def __init__(self, vocab_size, num_hiddens, init_state, forward_fn, get_params):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.init_state, self.forward_fn = init_state, forward_fn
        self.trainable_variables = get_params(vocab_size, num_hiddens)

    def __call__(self, X, state):
        X = tf.one_hot(tf.transpose(X), self.vocab_size)
        X = tf.cast(X, tf.float32)
        return self.forward_fn(X, state, self.trainable_variables)

    def begin_state(self, batch_size, *args, **kwargs):
        return self.init_state(batch_size, self.num_hiddens)


if __name__ == "__main__":
    tf.compat.v1.enable_eager_execution()

    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

    train_random_iter, vocab_random_iter = d2l.load_data_time_machine(batch_size, num_steps, use_random_iter=True)

    # tf.one_hot(tf.constant([0, 2]), len(vocab))

    X = tf.reshape(tf.range(10), (2, 5))
    # tf.one_hot(tf.transpose(X), 28).shape

    # 定义tensorflow训练策略
    strategy = tf.distribute.OneDeviceStrategy('/device:DML:0')

    num_hiddens = 512
    with strategy.scope():
        net = RNNModelScratch(len(vocab), num_hiddens, init_rnn_state, rnn, get_params)
    state = net.begin_state(X.shape[0])
    Y, new_state = net(X, state)
    print(Y.shape, len(new_state), new_state[0].shape)

    print(d2l.predict_rnn('time traveller ', 10, net, vocab))

    num_epochs, lr = 500, 1
    d2l.train_rnn(net, train_iter, vocab, lr, num_epochs, strategy)
