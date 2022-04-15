import torch
import d2l

if __name__ == '__main__':
    voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
    train_features, train_labels = d2l.read_voc_images(voc_dir, True)

    n = 5
    imgs = train_features[0:n] + train_labels[0:n]
    imgs = [img.permute(1, 2, 0) for img in imgs]
    d2l.show_images(imgs, 2, n)

    y = d2l.voc_label_indices(train_labels[0], d2l.voc_colormap2label())
    print(y[105:115, 130:140], d2l.VOC_CLASSES[1])

    imgs = []
    for _ in range(n):
        imgs += d2l.voc_rand_crop(train_features[0], train_labels[0], 200, 300)

    imgs = [img.permute(1, 2, 0) for img in imgs]
    d2l.show_images(imgs[::2] + imgs[1::2], 2, n)

    crop_size = (320, 480)
    voc_train = d2l.VOCSegDataset(True, crop_size, voc_dir)
    voc_test = d2l.VOCSegDataset(False, crop_size, voc_dir)

    batch_size = 64
    train_iter = torch.utils.data.DataLoader(voc_train, batch_size, shuffle=True, drop_last=True,
                                             num_workers=d2l.get_dataloader_workers())
    for X, Y in train_iter:
        print(X.shape)
        print(Y.shape)
        break
