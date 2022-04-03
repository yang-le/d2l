import torchvision
import d2l
from torch.utils import data
from torchvision import transforms

if __name__ == "__main__":
    # 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
    # 并除以255使得所有像素的数值均在0到1之间
    trans = transforms.ToTensor()
    mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False, transform=trans, download=True)

    X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
    d2l.show_images(X.reshape(18, 28, 28), 2, 9, titles=d2l.get_fashion_mnist_labels(y))
    d2l.plt.show()

    batch_size = 256
    train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=d2l.get_dataloader_workers())

    timer = d2l.Timer()
    for X, y in train_iter:
        continue
    print(f'{timer.stop():.2f} sec')

    train_iter, test_iter = d2l.load_data_fashion_mnist(32, resize=64)
    for X, y in train_iter:
        print(X.shape, X.dtype, y.shape, y.dtype)
        break
