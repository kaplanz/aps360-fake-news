import logging
import random


class DataLoader():
    """Iterate over data samples in batches."""
    def __init__(self, samples, batch_size=32, shuffle=True):
        self.samples = samples
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batches = self.get_batches()
        self.loaders = self.get_loaders()
        logging.info(self)

    def __iter__(self):  # called by Python to create an iterator
        if self.shuffle:
            # Update batches, loaders
            self.batches = self.get_batches()
            self.loaders = self.get_loaders()
        # Make an iterator for every batch
        iters = [iter(loader) for loader in self.loaders]
        while iters:
            # Pick an iterator (a batch)
            if self.shuffle:
                im = random.choice(iters)
            else:
                im = iters[0]
            # Use this iterator until it is depleted
            try:
                yield next(im)
            except StopIteration:
                # No more elements in the iterator, remove it
                iters.remove(im)

    def __len__(self):
        return len(self.batches)

    def __str__(self):
        return '{}: {} samples, {} batches'.format(self.__class__.__name__,
                                                   len(self.samples),
                                                   len(self))

    def get_batches(self):
        """Get batches from the data samples."""
        raise NotImplementedError

    def get_loaders(self):
        """Get loaders from the data batches."""
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
