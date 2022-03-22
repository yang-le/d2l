import tensorflow as tf
import d2l


if __name__ == "__main__":
    true_w = tf.constant([2, -3.4])
    true_b = 4.2
    features, labels = d2l.synthetic_data(true_w, true_b, 1000)

    batch_size = 10
    data_iter = d2l.load_array((features, labels), batch_size)
    print(next(iter(data_iter)))

    # keras是TensorFlow的高级API
    initializer = tf.initializers.RandomNormal(stddev=0.01)
    net = tf.keras.Sequential()
    net.add(tf.keras.layers.Dense(1, kernel_initializer=initializer))

    loss = tf.keras.losses.MeanSquaredError()
    # loss = tf.keras.losses.Huber()
    trainer = tf.keras.optimizers.SGD(learning_rate=0.03)

    num_epochs = 3
    for epoch in range(num_epochs):
        for X, y in data_iter:
            with tf.GradientTape() as tape:
                l = loss(net(X, training=True), y)
            grads = tape.gradient(l, net.trainable_variables)
            trainer.apply_gradients(zip(grads, net.trainable_variables))
        l = loss(net(features), labels)
        print(f'epoch {epoch + 1}, loss {l:f}')

    w = net.get_weights()[0]
    print('w的估计误差：', true_w - tf.reshape(w, true_w.shape))
    b = net.get_weights()[1]
    print('b的估计误差：', true_b - b)
