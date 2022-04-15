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


if __name__ == "__main__":
    pretrained_net = torchvision.models.resnet18(pretrained=True)
    net = nn.Sequential(*list(pretrained_net.children())[:-2])

    X = torch.rand(size=(1, 3, 320, 480))
    print(net(X).shape)

    num_classes = 21
    net.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1))
    net.add_module('transpose_conv',
                   nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, padding=16, stride=32))

    conv_trans = nn.ConvTranspose2d(3, 3, kernel_size=4, padding=1, stride=2, bias=False)
    conv_trans.weight.data.copy_(bilinear_kernel(3, 3, 4))

    img = torchvision.transforms.ToTensor()(d2l.Image.open())