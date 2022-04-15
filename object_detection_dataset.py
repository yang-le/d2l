import d2l


if __name__ == "__main__":
    batch_size, edge_size = 32, 256
    train_iter, _ = d2l.load_data_bananas(batch_size)
    batch = next(iter(train_iter))
    print(batch[0].shape, batch[1].shape)

    imgs = (batch[0][0:10].permute(0, 2, 3, 1)) / 255
    axes = d2l.show_images(imgs, 2, 5, scale=2, show=False)
    for ax, label in zip(axes, batch[1][0:10]):
        d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
    d2l.plt.show()
