import tensorflow as tf
import d2l


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    d2l.plt.rcParams['font.sans-serif'] = ['SimHei']
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.imshow(img.numpy())
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    d2l.plt.show()
    return axes


if __name__ == "__main__":
    d2l.use_svg_display()

    mnist_train, mnist_test = tf.keras.datasets.fashion_mnist.load_data()

    X = tf.constant(mnist_train[0][:18])
    y = tf.constant(mnist_train[1][:18])
    show_images(X, 2, 9, titles=d2l.get_fashion_mnist_labels(y))

    batch_size = 256
    train_iter = tf.data.Dataset.from_tensor_slices(mnist_train).batch(batch_size).shuffle(len(mnist_train[0]))

    timer = d2l.Timer()
    for X, y in train_iter:
        continue
    print(f'{timer.stop():.2f} sec')

    train_iter, test_iter = d2l.load_data_fashion_mnist(32, resize=64)
    for X, y in train_iter:
        print(X.shape, X.dtype, y.shape, y.dtype)
        break
