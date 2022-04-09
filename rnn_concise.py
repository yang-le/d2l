import tensorflow as tf
import d2l

if __name__ == "__main__":
    tf.compat.v1.enable_eager_execution()

    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

    num_hiddens = 256
    rnn_layer = tf.keras.layers.SimpleRNN(num_hiddens, time_major=True, return_sequences=True, return_state=True)

    strategy = tf.distribute.OneDeviceStrategy('/device:DML:0')
    with strategy.scope():
        net = d2l.RNNModel(rnn_layer, vocab_size=len(vocab))

    print(d2l.predict_rnn('time traveller', 10, net, vocab))

    num_epochs, lr = 500, 1
    d2l.train_rnn(net, train_iter, vocab, lr, num_epochs, strategy)
