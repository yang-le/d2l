import torch
import d2l


if __name__ == "__main__":
    # data_dir = d2l.download_extract('SNLI')
    train_data = d2l.read_snli('../data/snli_1.0', is_train=True)

    for x0, x1, y in zip(train_data[0][:3], train_data[1][:3], train_data[2][:3]):
        print('前提：', x0)
        print('假设：', x1)
        print('标签：', y)

    train_iter, test_iter, vocab = d2l.load_data_snli(128, 50)
    print(len(vocab))

    for X, Y in train_iter:
        print(X[0].shape)
        print(X[1].shape)
        print(Y.shape)
        break
