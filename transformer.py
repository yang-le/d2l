import tensorflow as tf
import numpy as np
import pandas as pd
import d2l

if __name__ == "__main__":
    tf.compat.v1.enable_eager_execution()

    num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
    lr, num_epochs = 0.005, 200
    ffn_num_hiddens, num_heads = 64, 4
    key_size, query_size, value_size = 32, 32, 32
    norm_shape = [2]

    train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
    encoder = d2l.TransformerEncoder(len(src_vocab), key_size, query_size, value_size, num_hiddens, norm_shape,
                                     ffn_num_hiddens, num_heads, num_layers, dropout)
    decoder = d2l.TransformerDecoder(len(tgt_vocab), key_size, query_size, value_size, num_hiddens, norm_shape,
                                     ffn_num_hiddens, num_heads, num_layers, dropout)

    # 定义训练策略
    strategy = tf.distribute.OneDeviceStrategy('/device:DML:0')

    with strategy.scope():
        net = d2l.EncoderDecoder(encoder, decoder)

    try:
        net.load_weights('transformer.params')
    except Exception:
        ...

    d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, strategy, 'DML:0')
    net.save_weights('transformer.params')
    d2l.plt.show()

    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    for eng, fra in zip(engs, fras):
        translation, dec_attention_weight_seq = d2l.predict_seq2seq(net, eng, src_vocab, tgt_vocab, num_steps, True)
        print(f'{eng} => {translation}, bleu {d2l.bleu(translation, fra, k=2):.3f}')

    enc_attention_weights = tf.reshape(tf.concat(net.encoder.attention_weights, 0),
                                       (num_layers, num_heads, -1, num_steps))

    d2l.show_heatmaps(enc_attention_weights, xlabel='Key positions', ylabel='Query positions',
                      titles=['Head %d' % i for i in range(1, 5)], figsize=(7, 3.5))

    dec_attention_weights_2d = [head[0] for step in dec_attention_weight_seq
                                for attn in step for blk in attn for head in blk]
    dec_attention_weights_filled = tf.convert_to_tensor(
        np.asarray(pd.DataFrame(dec_attention_weights_2d).fillna(0.0).values).astype(np.float32))
    dec_attention_weights = tf.reshape(dec_attention_weights_filled, shape=(-1, 2, num_layers, num_heads, num_steps))
    dec_self_attention_weights, dec_inter_attention_weights = tf.transpose(dec_attention_weights, perm=(1, 2, 3, 0, 4))

    d2l.show_heatmaps(dec_self_attention_weights[:, :, :, :len(translation.split()) + 1], xlabel='Key positions',
                      ylabel='Query positions', titles=['Head %d' % i for i in range(1, 5)], figsize=(7, 3.5))

    d2l.show_heatmaps(dec_inter_attention_weights, xlabel='Key positions', ylabel='Query positions',
                      titles=['Head %d' % i for i in range(1, 5)], figsize=(7, 3.5))
