import torch
import d2l


if __name__ == "__main__":
    data_dir = d2l.download_extract('aclImdb', 'aclImdb')
    train_data = d2l.read_imdb(data_dir, is_train=True)
    print('训练集数目：', len(train_data[0]))
    # for x, y in zip(train_data[0][:3], train_data[1][:3]):
    #     print('标签：', y, 'review:', x[0:60])

    train_tokens = d2l.tokenize(train_data[0], token='word')
    vocab = d2l.Vocab(train_tokens, min_freq=5, reserved_tokens=['<pad>'])

    d2l.set_figsize()
    d2l.plt.xlabel('# tokens per review')
    d2l.plt.ylabel('count')
    d2l.plt.hist([len(line) for line in train_tokens], bins=range(0, 1000, 50))
    d2l.plt.show()

