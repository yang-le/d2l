import tensorflow as tf
import d2l


if __name__ == "__main__":
    tf.compat.v1.enable_eager_execution()

    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

    vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
    num_epochs, lr = 500, 2

    # 定义训练策略
    strategy = tf.distribute.OneDeviceStrategy('/device:DML:0')

    lstm_layer = tf.keras.layers.RNN(tf.keras.layers.StackedRNNCells([
        tf.keras.layers.LSTMCell(num_hiddens) for _ in range(num_layers)
    ]), time_major=True, return_sequences=True, return_state=True)

    with strategy.scope():
        model = d2l.RNNModel(lstm_layer, vocab_size=vocab_size)

    d2l.train_rnn(model, train_iter, vocab, lr, num_epochs, strategy)
