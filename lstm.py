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

    W_xi, W_hi, b_i = get_param()  # 输入门参数
    W_xf, W_hf, b_f = get_param()  # 遗忘门参数
    W_xo, W_ho, b_o = get_param()  # 输出门参数
    W_xc, W_hc, b_c = get_param()  # 候选记忆元参数
    # 输出层参数
    W_hq = tf.Variable(normal((num_hiddens, num_ouputs)), dtype=tf.float32)
    b_q = tf.Variable(tf.zeros(num_ouputs), dtype=tf.float32)
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q]
    return params


def init_lstm_state(batch_size, num_hiddens):
    return (
        tf.zeros((batch_size, num_hiddens)),
        tf.zeros((batch_size, num_hiddens))
    )


def lstm(inputs, state, params):
    W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q = params
    (H, C) = state
    outputs = []
    for X in inputs:
        X = tf.reshape(X, [-1, W_xi.shape[0]])
        I = tf.sigmoid(tf.matmul(X, W_xi) + tf.matmul(H, W_hi) + b_i)
        F = tf.sigmoid(tf.matmul(X, W_xf) + tf.matmul(H, W_hf) + b_f)
        O = tf.sigmoid(tf.matmul(X, W_xo) + tf.matmul(H, W_ho) + b_o)
        C_tilda = tf.tanh(tf.matmul(X, W_xc) + tf.matmul(H, W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * tf.tanh(C)
        Y = tf.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return tf.concat(outputs, axis=0), (H, C)


if __name__ == "__main__":
    tf.compat.v1.enable_eager_execution()

    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

    vocab_size, num_hiddens = len(vocab), 256
    num_epochs, lr = 500, 1

    # 定义训练策略
    strategy = tf.distribute.OneDeviceStrategy('/device:DML:0')
    # with strategy.scope():
    #     model = d2l.RNNModelScratch(vocab_size, num_hiddens, init_lstm_state, lstm, get_params)

    lstm_layer = tf.keras.layers.LSTM(num_hiddens, time_major=True, return_sequences=True, return_state=True)

    with strategy.scope():
        model = d2l.RNNModel(lstm_layer, vocab_size=vocab_size)

    d2l.train_rnn(model, train_iter, vocab, lr, num_epochs, strategy)
