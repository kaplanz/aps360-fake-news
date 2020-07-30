class Dataset():
    """Object for managing and loading a raw dataset."""
    def __init__(self, root='./data'):
        self.root = root
        self.data = None

    def path(self):
        """Return the base dataset path."""
        raise NotImplementedError

    def load(self):
        """Return a pandas DataFrame containing raw data."""
        raise NotImplementedError
