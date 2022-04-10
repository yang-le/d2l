import tensorflow as tf
import d2l

if __name__ == "__main__":
    tf.compat.v1.enable_eager_execution()

    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0
    batch_size, num_steps = 64, 10
    lr, num_epochs = 0.005, 300

    train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
    encoder = d2l.Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = d2l.Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)

    # 定义训练策略
    strategy = tf.distribute.OneDeviceStrategy('/device:DML:0')

    with strategy.scope():
        net = d2l.EncoderDecoder(encoder, decoder)

    try:
        net.load_weights('nmt.params')
    except Exception:
        ...

    d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, strategy, 'DML:0')
    net.save_weights('nmt.params')
    d2l.plt.show()

    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    for eng, fra in zip(engs, fras):
        translation, attention_weight_seq = d2l.predict_seq2seq(net, eng, src_vocab, tgt_vocab, num_steps)
        print(f'{eng} => {translation}, bleu {d2l.bleu(translation, fra, k=2):.3f}')
