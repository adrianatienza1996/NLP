import torch
import torch.nn as nn


class BiLSTM_Model(nn.Module):
    def __init__(self, vocab_size, rnn_units, num_classes):
        super(BiLSTM_Model, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=64,
                                      padding_idx=0)
        self.bilstm = nn.LSTM(input_size=64,
                              hidden_size=rnn_units,
                              batch_first=True,
                              bidirectional=True
                              )
        self.classifier = nn.Linear(rnn_units * 2, num_classes)

    def forward (self, x):
        h = self.embedding(x)
        output, _ = self.bilstm(h)
        return self.classifier(output)
