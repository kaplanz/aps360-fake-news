import torch

from .dataloader import DataLoader


class BucketLoader(DataLoader):
    """Iterate over data samples in padded batches of similar size."""
    def __init__(self, samples, batch_size=32, shuffle=True):
        self.samples = sorted(
            samples, key=lambda x: x[0].shape[0])  # sort samples by length
        super().__init__(samples, batch_size, shuffle)
        self.loaders = self.get_loaders()

    def get_batches(self):
        # Create batches of similar length samples
        batches = []
        for batch in list(DataLoader.divide(self.samples, self.batch_size)):
            text, labels = zip(*batch)
            # Pad text samples in each batch
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
