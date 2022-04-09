import torch
from torch import nn
import d2l


batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

num_hiddens = 256
rnn_layer = nn.RNN(len(vocab), num_hiddens)

state = torch.zeros((1, batch_size, num_hiddens))

device = d2l.try_gpu()
net = d2l.RNNModel(rnn_layer, vocab_size=len(vocab))
net = net.to(device)
print(d2l.predict_rnn('time traveller', 10, net, vocab, device))

num_epochs, lr = 500, 1
d2l.train_rnn(net, train_iter, vocab, lr, num_epochs, device)
