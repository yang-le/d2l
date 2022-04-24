import torch
from torch import nn
import d2l


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, kernel_sizes, num_channels, **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 这个嵌入层不需要训练
        self.constant_embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_channels), 2)
        # 最大时间汇聚层没有参数，因此可以共享此实例
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.relu = nn.ReLU()
        # 创建多个一维卷积层
        self.convs = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(2 * embed_size, c, k))

    def forward(self, inputs):
        # 沿着向量维度将两个嵌入层连结起来，
        # 每个嵌入层的输出形状都是（批量大小，词元数量，词元向量维度）连结起来
        embeddings = torch.cat((self.embedding(inputs), self.constant_embedding(inputs)), dim=2)
        # 根据一维卷积层的输入格式，重新排列张量，以便通道作为第2维
        embeddings = embeddings.permute(0, 2, 1)
        # 每个一维卷积层在最大时间汇聚层合并后，获得的张量形状是（批量大小，通道数，1）
        # 删除最后一个维度并沿通道维度连结
        encoding = torch.cat([torch.squeeze(self.relu(self.pool(conv(embeddings))), dim=-1)
                              for conv in self.convs], dim=1)
        outputs = self.decoder(self.dropout(encoding))
        return outputs


def init_weights(m):
    if type(m) in (nn.Linear, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)


if __name__ == "__main__":
    batch_size = 64
    train_iter, test_iter, vocab = d2l.load_data_imdb(batch_size)

    embed_size, kernel_sizes, num_channels = 100, [3, 4, 5], [100, 100, 100]
    device = d2l.try_gpu()
    net = TextCNN(len(vocab), embed_size, kernel_sizes, num_channels)
    net.apply(init_weights)

    glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
    embeds = glove_embedding[vocab.idx_to_token]
    net.embedding.weight.data.copy_(embeds)
    net.embedding.weight.requires_grad = False

    lr, num_epochs = 0.001, 5
    net = net.to(device)
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    d2l.train_on_dev_(net, train_iter, test_iter, loss, trainer, num_epochs, device)

    print(d2l.predict_sentiment(net, vocab, 'this movie is so great'))
    print(d2l.predict_sentiment(net, vocab, 'this movie is so bad'))
