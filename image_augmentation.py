import torch
import torchvision
from torch import nn
import d2l


def load_cifar10(is_train, augs, batch_size):
    dataset = torchvision.datasets.CIFAR10(root='../data', train=is_train, transform=augs, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train,
                                             num_workers=d2l.get_dataloader_workers())
    return dataloader


def init_weights(m):
    if type(m) in [nn.Linear, nn.Conv2d]:
        nn.init.xavier_uniform_(m.weight)


def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    loss = nn.CrossEntropyLoss()
    device = d2l.try_dml()
    net.to(device)
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    d2l.train_on_dev_(net, train_iter, test_iter, loss, trainer, 10, device)


if __name__ == "__main__":
    train_augs = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor()
    ])
    test_augs = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    batch_size, net = 256, d2l.resnet18(10, 3)
    net.apply(init_weights)
    train_with_data_aug(train_augs, test_augs, net)
