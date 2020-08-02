import torch
import torch.nn as nn
import torch.nn.functional as F


class FakeNewsNet(nn.Module):
    def __init__(self, emb):
        super(FakeNewsNet, self).__init__()
        self.hs = 16
        self.emb = nn.Embedding.from_pretrained(emb.vectors)
        self.rnn = nn.GRU(emb.dim, self.hs, num_layers=2, batch_first=True)
        self.fc = nn.Linear(self.hs, 1)

    def forward(self, x):
        x = self.emb(x)  # look up the embedding
        # RNNs
        h0 = torch.zeros(2, x.shape[0], self.hs)  # set an initial hidden state
        out, _ = self.rnn(x, h0)  # forward propagate the RNNs
        # Pooling
        out = torch.max(out, dim=1)[0]  # concatenate pooled outputs
        # Fully-connected layer
        out = self.fc(out)  # run through fully connected layer
        return out
