import torch
from torch import nn
import d2l


batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

vocab_size, num_hidens, num_layers = len(vocab), 256, 2
device = d2l.try_gpu()
lstm_layer = nn.LSTM(vocab_size, num_hidens, num_layers)
model = d2l.RNNModel(lstm_layer, vocab_size)
model = model.to(device)

num_epochs, lr = 500, 2
d2l.train_rnn(model, train_iter, vocab, lr, num_epochs, device)
