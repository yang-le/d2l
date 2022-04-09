import tensorflow as tf
import d2l


def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist):
    """绘制列表长度对的直方图"""
    d2l.set_figsize()
    _, _, patches = d2l.plt.hist([[len(l) for l in xlist], [len(l) for l in ylist]])
    d2l.plt.xlabel(xlabel)
    d2l.plt.ylabel(ylabel)
    for patch in patches[1].patches:
        patch.set_hatch('/')
    d2l.plt.legend(legend)
    d2l.plt.show()


if __name__ == "__main__":
    tf.compat.v1.enable_eager_execution()

    raw_text = d2l.read_data_nmt()
    text = d2l.preprocess_nmt(raw_text)
    source, target = d2l.tokenize_nmt(text)
    show_list_len_pair_hist(['source', 'target'], '# tokens per sequence', 'count', source, target)

    train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size=2, num_steps=8)
    for X, X_valid_len, Y, Y_valid_len in train_iter:
        print('X:', tf.cast(X, tf.int32))
        print('X的有效长度:', X_valid_len)
        print('Y:', tf.cast(Y, tf.int32))
        print('Y的有效长度:', Y_valid_len)
        break
