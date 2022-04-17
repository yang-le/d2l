import os
import pandas as pd
import torch
import torchvision
from torch import nn
import d2l


def reorg_cifar10_data(data_dir, valid_ratio):
    labels = d2l.read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
    d2l.reorg_train_valid(data_dir, labels, valid_ratio)
    d2l.reorg_test(data_dir)


def get_net():
    num_classes = 10
    net = d2l.resnet18(num_classes, 3)
    return net


def train(net, train_iter, valid_iter, num_epochs, lr, wd, device, lr_period, lr_decay):
    net.to(device)
    trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['train loss', 'train acc']
    if valid_iter is not None:
        legend.append('valid acc')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], legend=legend)
    for epoch in range(num_epochs):
        net.train()
        metric = d2l.Accumulator(3)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            trainer.zero_grad()
            if isinstance(features, list):
                # 微调BERT中所需（稍后讨论）
                features = [feature.to(device) for feature in features]
            else:
                features = features.to(device)
            labels = labels.to(device)
            labels_hat = net(features)
            l = loss(labels_hat, labels).sum()
            l.backward()
            trainer.step()
            metric.add(l, d2l.accuracy(labels_hat, labels), labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (metric[0] / metric[2], metric[1] / metric[2], None))
        if valid_iter is not None:
            valid_acc = d2l.evaluate_accuracy_on_dev(net, valid_iter, device)
            animator.add(epoch + 1, (None, None, valid_acc))
        scheduler.step()
    measures = (f'train loss {metric[0] / metric[2]:.3f}, '
                f'train acc {metric[1] / metric[2]:.3f}')
    if valid_iter is not None:
        measures += f', valid acc {valid_acc:.3f}'
    print(measures + f'\n{metric[2] * num_epochs / timer.sum():.1f}'
                     f' examples/sec on {str(device)}')


if __name__ == "__main__":
    # 如果你使用完整的Kaggle竞赛的数据集，设置demo为False
    demo = False

    if demo:
        data_dir = d2l.download_extract('cifar10_tiny')
    else:
        data_dir = '../data/cifar-10/'

    labels = d2l.read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
    print('# 训练样本 :', len(labels))
    print('# 类别 :', len(set(labels.values())))

    batch_size = 32 if demo else 128
    valid_ratio = 0.1
    # reorg_cifar10_data(data_dir, valid_ratio)

    transform_train = torchvision.transforms.Compose([
        # 在高度和宽度上将图像放大到40像素的正方形
        torchvision.transforms.Resize(40),
        # 随机裁剪出一个高度和宽度均为40像素的正方形图像，
        # 生成一个面积为原始图像面积0.64到1倍的小正方形，
        # 然后将其缩放为高度和宽度均为32像素的正方形
        torchvision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0), ratio=(1.0, 1.0)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        # 标准化图像的每个通道
        torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    train_ds, train_valid_ds = [
        torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train_valid_test', folder), transform=transform_train)
        for folder in ['train', 'train_valid']]

    valid_ds, test_ds = [
        torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train_valid_test', folder), transform=transform_test)
        for folder in ['valid', 'test']]

    train_iter, train_valid_iter = [torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, drop_last=True) for
                                    dataset in (train_ds, train_valid_ds)]

    valid_iter = torch.utils.data.DataLoader(valid_ds, batch_size, shuffle=False, drop_last=True)
    test_iter= torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False, drop_last=False)

    loss = nn.CrossEntropyLoss(reduction='none')

    device, num_epochs, lr, wd = d2l.try_dml(), 100, 0.1, 5e-4
    lr_period, lr_decay, net = 50, 0.1, get_net()
    # train(net, train_iter, valid_iter, num_epochs, lr, wd, device, lr_period, lr_decay)

    net, preds = get_net(), []
    train(net, train_valid_iter, None, num_epochs, lr, wd, device, lr_period, lr_decay)

    for X, _ in test_iter:
        y_hat = net(X.to(device))
        preds.extend(y_hat.cpu().argmax(dim=1).type(torch.int32).numpy())
    sorted_ids = list(range(1, len(test_ds) + 1))
    sorted_ids.sort(key=lambda x: str(x))
    df = pd.DataFrame({'id': sorted_ids, 'label': preds})
    df['label'] = df['label'].apply(lambda x: train_valid_ds.classes[x])
    df.to_csv('submission.csv', index=False)
