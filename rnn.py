import tensorflow as tf
import d2l

X, W_xh = tf.random.normal((3, 1), 0, 1), tf.random.normal((1, 4), 0, 1)
H, W_hh = tf.random.normal((3, 4), 0, 1), tf.random.normal((4, 4), 0, 1)
print(tf.matmul(X, W_xh) + tf.matmul(H, W_hh))
print(tf.matmul(tf.concat((X, H), 1), tf.concat((W_xh, W_hh), 0)))
