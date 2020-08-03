import random

import torch

from .dataloader import DataLoader


class BatchLoader(DataLoader):
    """Iterate over data samples in random batches."""
    def __init__(self, samples, batch_size=32, shuffle=True):
        super().__init__(samples, batch_size, shuffle)

    def __iter__(self):  # called by Python to create an iterator
        self.batches = self.get_batches()
        # Make an iterator for every batch
        iters = [iter(loader) for loader in self.get_loaders()]
        while iters:
            # Pick an iterator (a batch)
            im = random.choice(iters)
            try:
                yield next(im)
            except StopIteration:
                # No more elements in the iterator, remove it
                iters.remove(im)

    def get_batches(self):
        # Shuffle if needed
        if self.shuffle:
            random.shuffle(self.samples)
        # Divide samples into batches
        batches = []
        for batch in list(DataLoader.divide(self.samples, self.batch_size)):
            text, labels = zip(*batch)
            # Pad text samples in each batch
            text = [t[:100] for t in text]
            text = torch.nn.utils.rnn.pad_sequence(text, batch_first=True)
            # Combine labels into single tensor
            labels = torch.tensor(labels)
            # Store this batch
            batches.append((text, labels))
        return batches

    def get_loaders(self):
        # Create a DataLoader for each batch of the same length
        return [
            torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(*batch),
                batch_size=self.batch_size,
                shuffle=self.shuffle
            )  # omit last batch if smaller than batch_size
            for batch in self.batches
        ]
