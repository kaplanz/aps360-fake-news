import logging
import random

import torch


class DataLoader():
    """Iterate over data samples in batches."""
    def __init__(self, samples, batch_size=32):
        # Store samples sorted by length
        self.samples = sorted(samples, key=lambda x: x[0].shape[0])

        # Create batches of similar length samples
        self.batches = []
        for batch in list(DataLoader.divide(self.samples, batch_size)):
            text, labels = zip(*batch)
            # Pad text samples in each batch
            text = torch.nn.utils.rnn.pad_sequence(text, batch_first=True)
            # Combine labels into single tensor
            labels = torch.tensor(labels)
            # Store this batch
            self.batches.append((text, labels))
        logging.debug('Created {} batches of size {} from {} samples'.format(
            len(self.batches), self.batches[0][0].shape[0], len(samples)))

        # Create a DataLoader for each batch of the same length
        self.loaders = [
            torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(*batch),
                batch_size=batch_size,
                shuffle=True)  # omit last batch if smaller than batch_size
            for batch in self.batches
        ]

    def __iter__(self):  # called by Python to create an iterator
        # Make an iterator for every batch
        iters = [iter(loader) for loader in self.loaders]
        while iters:
            # Pick an iterator (a batch)
            im = random.choice(iters)
            try:
                yield next(im)
            except StopIteration:
                # No more elements in the iterator, remove it
                iters.remove(im)

    def __len__(self):
        return len(self.loaders)

    @staticmethod
    def divide(arr, n):
        """Divide `arr` into sublists of size `n`."""
        for i in range(0, len(arr), n):
            yield arr[i:i + n]

    @staticmethod
    def split(arr, n):
        """Split `arr` into `n` roughly equal sublists."""
        k, m = divmod(len(arr), n)
        return (arr[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]
                for i in range(n))

    @staticmethod
    def splits(arr, split_ratio):
        """Split `arr` into `len(split_ratio)` sublists according to `split_ratio`."""
        assert sum(split_ratio) <= 1

        end = 0
        for r in split_ratio:
            start = end
            end += int(len(arr) * r)
            yield arr[start:end]


def get_loaders(samples, split_ratio=[.6, .2, .2], batch_size=32):
    # Split samples
    random.shuffle(samples)
    splits = list(DataLoader.splits(samples, split_ratio))

    # Get data loaders
    loaders = [DataLoader(split, batch_size=batch_size) for split in splits]

    return loaders
