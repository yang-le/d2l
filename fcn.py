import torch
import torchvision
from torch import nn
from torch.nn import functional as F
import d2l


def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1), torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels, kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight


def predict(img):
    X = test_iter.dataset.normalize_image(img).unsqueeze(0)
    pred = net(X.to(device)).argmax(dim=1)
    return pred.reshape(pred.shape[1], pred.shape[2])


def label2image(pred):
    colormap = torch.tensor(d2l.VOC_COLORMAP, device=device)
    X = pred.long()
    return colormap[X, :]


if __name__ == "__main__":
    pretrained_net = torchvision.models.resnet18(pretrained=True)
    net = nn.Sequential(*list(pretrained_net.children())[:-2])

    X = torch.rand(size=(1, 3, 320, 480))
    print(net(X).shape)

    num_classes = 21
    net.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1))
    net.add_module('transpose_conv',
                   nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, padding=16, stride=32))

    # conv_trans = nn.ConvTranspose2d(3, 3, kernel_size=4, padding=1, stride=2, bias=False)
    # conv_trans.weight.data.copy_(bilinear_kernel(3, 3, 4))
    #
    # img = torchvision.transforms.ToTensor()(d2l.Image.open('../img/catdog.jpg'))
    # X = img.unsqueeze(0)
    # Y = conv_trans(X)
    # out_img = Y[0].permute(1, 2, 0).detach()
    #
    # d2l.set_figsize()
    # print('input image shape:', img.permute(1, 2, 0).shape)
    # print('output image shape:', out_img.shape)
    # d2l.plt.imshow(out_img)
    # d2l.plt.show()

    W = bilinear_kernel(num_classes, num_classes, 64)
    net.transpose_conv.weight.data.copy_(W)

    batch_size, crop_size = 32, (320, 480)
    train_iter, test_iter = d2l.load_data_voc(batch_size, crop_size)

    num_epochs, lr, wd, device = 5, 0.001, 1e-3, d2l.try_gpu()
    net.to(device)
    trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)
    loss = torch.nn.CrossEntropyLoss()
    # d2l.train_on_dev_(net, train_iter, test_iter, loss, trainer, num_epochs, device)

    voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
    test_images, test_labels = d2l.read_voc_images(voc_dir, False)
    n, imgs = 4, []
    for i in range(n):
        crop_rect = (0, 0, 320, 480)
        X = torchvision.transforms.functional.crop(test_images[i], *crop_rect)
        pred = label2image(predict(X))
        imgs += [X.permute(1, 2, 0), pred.cpu(),
                 torchvision.transforms.functional.crop(test_labels[i], *crop_rect).permute(1, 2, 0)]
    d2l.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2)
