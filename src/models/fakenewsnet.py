import torch
import torch.nn as nn


class FakeNewsNet(nn.Module):
    def __init__(self, emb):
        super(FakeNewsNet, self).__init__()
        self.hidden_size = emb.dim // 2
        self.emb = nn.Embedding.from_pretrained(emb.vectors)
        self.rnn = nn.GRU(emb.dim,
                          self.hidden_size,
                          batch_first=True,
                          bidirectional=True)
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(4 * self.hidden_size, 1)

    def forward(self, x):
        x = self.emb(x)  # look up the embedding
        # RNNs
        out, _ = self.rnn(x)  # forward propagate the RNNs
        # Pooling
        out = torch.cat([torch.max(out, dim=1)[0],
                         torch.mean(out, dim=1)],
                        dim=1)
        # Dropout
        out = self.drop(out)
        # Fully-connected layer
        out = self.fc(out)  # run through fully connected layer
        return out
