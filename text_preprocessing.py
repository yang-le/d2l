import collections
import d2l


lines = d2l.read_time_machine()
print(f'# 文本总行数: {len(lines)}')
print(lines[0])
print(lines[10])

tokens = d2l.tokenize(lines)
for i in range(11):
    print(tokens[i])

vocab = d2l.Vocab(tokens)
print(list(vocab.token_to_idx.items())[:10])

for i in [0, 10]:
    print('文本:', tokens[i])
    print('索引:', vocab[tokens[i]])

corpus, vocab = d2l.load_corpus_time_machine()
print(len(corpus), len(vocab))
