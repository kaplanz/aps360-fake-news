import torch
import torch.nn as nn
import torch.nn.functional as F


class FakeNewsNet(nn.Module):
    def __init__(self, emb):
        super(FakeNewsNet, self).__init__()
        self.emb = nn.Embedding.from_pretrained(emb.vectors)
        self.rnn = nn.RNN(emb.dim, 128, batch_first=True)
        self.fc = nn.Linear(128 * 2, 1)

    def forward(self, x):
        # Look up the embedding
        x = self.emb(x)
        # Set an initial hidden state
        h0 = torch.zeros(1, x.size(0), 128)
        # Forward propagate the RNN
        out, _ = self.rnn(x, h0)
        # Concatenate pooled outputs
        out = torch.cat([torch.max(out, dim=1)[0],
                         torch.mean(out, dim=1)],
                        dim=1)
        # Run through fully connected layer
        out = self.fc(out)
        # Return output
        return out
