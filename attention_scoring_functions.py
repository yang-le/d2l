import tensorflow as tf
import d2l


if __name__ == "__main__":
    tf.compat.v1.enable_eager_execution()

    queries, keys = tf.random.normal(shape=(2, 1, 20)), tf.ones((2, 10, 2))
    # values的小批量，两个值矩阵是相同的
    values = tf.repeat(tf.reshape(tf.range(40, dtype=tf.float32), shape=(1, 10, 4)), repeats=2, axis=0)
    valid_lens = tf.constant([2, 6])

    attention = d2l.AdditiveAttention(key_size=2, query_size=20, num_hiddens=8, dropout=0.1)
    attention(queries, keys, values, valid_lens, training=False)

    d2l.show_heatmaps(tf.reshape(attention.attention_weights, (1, 1, 2, 10)), xlabel='Keys', ylabel='Queries')

    queries = tf.random.normal(shape=(2, 1, 2))
    attention = d2l.DotProductAttention(dropout=0.5)
    attention(queries, keys, values, valid_lens, training=False)

    d2l.show_heatmaps(tf.reshape(attention.attention_weights, (1, 1, 2, 10)), xlabel='Keys', ylabel='Queries')
