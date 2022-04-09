import tensorflow as tf
import d2l


def get_params(vocab_size, num_hiddens):
    num_inputs = num_ouputs = vocab_size

    def normal(shape):
        return tf.random.normal(shape=shape, stddev=0.01, mean=0, dtype=tf.float32)

    def get_param():
        return (
            tf.Variable(normal((num_inputs, num_hiddens)), dtype=tf.float32),
            tf.Variable(normal((num_hiddens, num_hiddens)), dtype=tf.float32),
            tf.Variable(tf.zeros(num_hiddens), dtype=tf.float32)
        )

    W_xz, W_hz, b_z = get_param()   # 更新门参数
    W_xr, W_hr, b_r = get_param()   # 重置门参数
    W_xh, W_hh, b_h = get_param()   # 候选隐状态参数
    # 输出层参数
    W_hq = tf.Variable(normal((num_hiddens, num_ouputs)), dtype=tf.float32)
    b_q = tf.Variable(tf.zeros(num_ouputs), dtype=tf.float32)
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    return params


def init_gru_state(batch_size, num_hiddens):
    return (tf.zeros((batch_size, num_hiddens)),)


def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        X = tf.reshape(X, [-1, W_xh.shape[0]])
        Z = tf.sigmoid(tf.matmul(X, W_xz) + tf.matmul(H, W_hz) + b_z)
        R = tf.sigmoid(tf.matmul(X, W_xr) + tf.matmul(H, W_hr) + b_r)
        H_tilda = tf.tanh(tf.matmul(X, W_xh) + tf.matmul(R * H, W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = tf.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return tf.concat(outputs, axis=0), (H,)


if __name__ == "__main__":
    tf.compat.v1.enable_eager_execution()

    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

    vocab_size, num_hiddens = len(vocab), 256
    num_epochs, lr = 500, 1

    # 定义训练策略
    strategy = tf.distribute.OneDeviceStrategy('/device:DML:0')
    # with strategy.scope():
    #     model = d2l.RNNModelScratch(vocab_size, num_hiddens, init_gru_state, gru, get_params)

    gru_layer = tf.keras.layers.GRU(num_hiddens, time_major=True, return_sequences=True, return_state=True)

    with strategy.scope():
        model = d2l.RNNModel(gru_layer, vocab_size=vocab_size)

    d2l.train_rnn(model, train_iter, vocab, lr, num_epochs, strategy)
