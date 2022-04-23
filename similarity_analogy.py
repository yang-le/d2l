import torch
import fasttext
import fasttext.util
import d2l


def knn(W, x, k):
    # 增加1e-9以获得数值稳定性
    cos = torch.mv(W, x.reshape(-1,)) / (torch.sqrt(torch.sum(W * W, axis=1) + 1e-9) * torch.sqrt((x * x).sum()))
    _, topk = torch.topk(cos, k=k)
    return topk, [cos[int(i)] for i in topk]


def get_similar_tokens(query_token, embed, k=10):
    topk, cos = knn(embed.idx_to_vec, embed[[query_token]], k + 1)
    return [(embed.idx_to_token[int(i)], float(c)) for i, c in zip(topk[1:], cos[1:])]


def get_analogy(token_a, token_b, token_c, embed, k=10):
    vecs = embed[[token_a, token_b, token_c]]
    x = vecs[1] - vecs[0] + vecs[2]
    topk, cos = knn(embed.idx_to_vec, x, k)
    return [(embed.idx_to_token[int(i)], float(c)) for i, c in zip(topk, cos)]


if __name__ == "__main__":
    embed = d2l.TokenEmbedding('glove.6b.50d')
    # print(len(embed))
    # print(embed.token_to_idx['beautiful'], embed.idx_to_token[3367])
    #
    print(get_similar_tokens('chip', embed))
    # get_similar_tokens('baby', embed)
    # get_similar_tokens('beautiful', embed)

    print(get_analogy('man', 'woman', 'son', embed))
    # print(get_analogy('beijing', 'china', 'tokyo', embed))
    # print(get_analogy('bad', 'worst', 'big', embed))
    # print(get_analogy('do', 'did', 'go', embed))

    model = fasttext.util.download_model('zh', if_exists='ignore')
    ft = fasttext.load_model(model)
    print(ft.get_analogies('女', '男', '儿子'))
