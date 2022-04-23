import torch
import d2l


def compare_counts(token):
    return (f'"{token}"的数量：'
            f'之前={sum([l.count(token) for l in sentences])}, '
            f'之后={sum([l.count(token) for l in subsampled])}')


if __name__ == "__main__":
    sentences = d2l.read_ptb()
    vocab = d2l.Vocab(sentences, min_freq=10)
    subsampled, counter = d2l.subsample(sentences, vocab)
    d2l.show_list_len_pair_hist(['origin', 'subsampled'], '# tokens per setence', 'count', sentences, subsampled)

    print(compare_counts('the'))
    print(compare_counts('join'))

    corpus = [vocab[line] for line in subsampled]

    tiny_dataset = [list(range(7)), list(range(7, 10))]
    print('数据集', tiny_dataset)
    for center, context in zip(*d2l.get_centers_and_contexts(tiny_dataset, 2)):
        print('中心词', center, '的上下文词是', context)

    all_centers, all_contexts = d2l.get_centers_and_contexts(corpus, 5)
    print(f'# “中心词-上下文词对”的数量: {sum([len(contexts) for contexts in all_contexts])}')

    names = ['centers', 'contexts_negatives', 'masks', 'labels']
    data_iter, vocab = d2l.load_data_ptb(512, 5, 5)
    for batch in data_iter:
        for name, data in zip(names, batch):
            print(name, 'shape:', data.shape)
        break
