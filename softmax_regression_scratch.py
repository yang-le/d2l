import torch
import d2l


def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1,keepdim=True)
    return X_exp / partition    # 这里应用了广播机制


def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)


def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])


def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)


if __name__ == "__main__":
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    num_inputs = 784
    num_outputs = 10

    W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
    b = torch.zeros(num_outputs, requires_grad=True)

    y = torch.tensor([0, 2])
    y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])

    print(cross_entropy(y_hat, y))
    print(d2l.accuracy(y_hat, y) / len(y))
    print(d2l.evaluate_accuracy(net, test_iter))

    lr = 0.1
    num_epochs = 10
    d2l.train(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
    d2l.predict(net, test_iter)
