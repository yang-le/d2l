import tensorflow as tf
import d2l

if __name__ == "__main__":
    tf.compat.v1.enable_eager_execution()

    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0
    batch_size, num_steps = 64, 10
    lr, num_epochs = 0.005, 250

    train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
    encoder = d2l.Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = d2l.Seq2SeqAttentionDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)

    # 定义训练策略
    strategy = tf.distribute.OneDeviceStrategy('/device:DML:0')

    with strategy.scope():
        net = d2l.EncoderDecoder(encoder, decoder)

    try:
        net.load_weights('bahdanau.params')
    except Exception:
        ...

    d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, strategy, 'DML:0')
    net.save_weights('bahdanau.params')
    d2l.plt.show()

    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    for eng, fra in zip(engs, fras):
        translation, dec_attention_weight_seq = d2l.predict_seq2seq(net, eng, src_vocab, tgt_vocab, num_steps, True)
        print(f'{eng} => {translation}, bleu {d2l.bleu(translation, fra, k=2):.3f}')

    attention_weights = tf.reshape(tf.concat([step[0][0][0] for step in dec_attention_weight_seq], 0),
                                   (1, 1, -1, num_steps))
    # 加上一个包含序列结束词元
    d2l.show_heatmaps(attention_weights[:, :, :, :len(engs[-1].split()) + 1], xlabel='Key posistions',
                      ylabel='Query posistions')
