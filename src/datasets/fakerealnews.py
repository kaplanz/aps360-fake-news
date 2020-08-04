import logging
import os
import pickle
import urllib.request

import pandas as pd

from .dataset import Dataset


class FakeRealNews(Dataset):
    """Fake and real news dataset.

    Avaliable: <https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset>

    """
    def __init__(self, root='./data'):
        super().__init__(root)
        # URLs to download entities
        self.fakeurl = 'https://drive.google.com/uc?export=download&id=14TOZZ_9m1oHNpZ_Px0taoCB-0-vP73Kd'
        self.trueurl = 'https://drive.google.com/uc?export=download&id=1NfOcHzRV3vXERqDypVHSdckkDf_J-D-5'

    def path(self):
        return self.root + '/FakeRealNews'

    def load(self, download=True):
        """Load and return the dataset."""
        if self.data is None:
            # Load or parse the data
            args = {
                'file': 'Fake.csv',
                'url': self.fakeurl if download else None
            }
            fake = self.load_raw_file(**args)
            args = {
                'file': 'True.csv',
                'url': self.trueurl if download else None
            }
            true = self.load_raw_file(**args)

            # Create labels for the datasets
            fake['label'] = 0
            true['label'] = 1

            # Construct DataFrame
            data = pd.concat([fake, true])
            data.reset_index(drop=True, inplace=True)
            self.data = data

        return self.data

    def load_raw_file(self, file, url=None):
        """Load a file of the raw dataset."""
        path = self.path() + '/raw/' + file
        pklpath = path.replace('csv', 'pkl')

        # Load from pickle (if exists)
        if os.path.isfile(pklpath):
            logging.info('Loading binary from {}'.format(pklpath))
            with open(pklpath, 'rb') as f:
                return pickle.load(f)
        # Load from csv
        elif os.path.isfile(path):
            logging.info('Loading csv from {}'.format(path))
            data = pd.read_csv(path)
            # Dump raw DataFrame to pickle
            logging.info('Saving binary to {}'.format(path))
            with open(pklpath, 'wb') as f:
                pickle.dump(data, f)
            return data
        elif url is not None:
            logging.info('Downloading dataset file {}'.format(file))
            # Attempt to download the file
            with urllib.request.urlopen(url) as response:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, 'wb') as f:
                    f.write(response.read())
                    # Recurse to load the file
                    return self.load_raw_file(file)
        else:
            logging.error('Could not load file {}'.format(file))
