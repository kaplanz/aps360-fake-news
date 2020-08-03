import logging
import random

import torch


class DataLoader():
    """Iterate over data samples in batches."""
    def __init__(self, samples, batch_size=32, shuffle=True):
        self.samples = samples
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batches = self.get_batches()
        logging.info(self)

    def __iter__(self):  # called by Python to create an iterator
        raise NotImplementedError

    def __len__(self):
        return len(self.batches)

    def __str__(self):
        return '{}: {} samples, {} batches'.format(self.__class__.__name__,
                                                   len(self.samples),
                                                   len(self))

    def get_batches(self):
        """Get batches from the data samples."""
        raise NotImplementedError

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
