import torch
import torch.nn as nn


class BiLSTM_Model(nn.Module):
    def __init__(self, vocab_size, rnn_units, num_classes, max_len=50):
        super(BiLSTM_Model, self).__init__()
        self.max_len = max_len
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=64,
                                      padding_idx=0)
        self.bilstm = nn.LSTM(input_size=64,
                              hidden_size=rnn_units//2,
                              num_layers=1,
                              batch_first=True,
                              bidirectional=True
                              )
        self.classifier = nn.Linear(rnn_units, num_classes)

    def forward(self, x):
        x = x.long()
        h = self.embedding(x).view(x.shape[0], self.max_len, -1)
        output, _ = self.bilstm(h)
        output = self.classifier(output)
        return output
