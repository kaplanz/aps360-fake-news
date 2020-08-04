import logging
import os
import pickle
import urllib.request

import pandas as pd

from .dataset import Dataset


class FakeNews(Dataset):
    """Fake news dataset.

    Avaliable: <https://www.kaggle.com/c/fake-news>

    """
    def __init__(self, root='./data'):
        super().__init__(root)
        # URLs to download entities
        self.trainurl = 'https://drive.google.com/uc?export=download&id=1xhrxWBjpUehaEEvXdHMeiTdf_kWyJ2ra'
        self.testurl = 'https://drive.google.com/uc?export=download&id=18XCKwKCfJE-xBb5047VrHtsMrzec8qMY'
        self.submiturl = 'https://drive.google.com/uc?export=download&id=1HdKRVo3-hHwpl3dejTegJ5EO0CXFWR9k'

    def path(self):
        return self.root + '/FakeNews'

    def load(self, download=True):
        """Load and return the dataset."""
        if self.data is None:
            # Load or parse the data
            args = {
                'file': 'train.csv',
                'url': self.trainurl if download else None
            }
            data = self.load_raw_file(**args)
            # Download test, submit files
            args = {
                'file': 'test.csv',
                'url': self.testurl if download else None
            }
            self.load_raw_file(**args)
            args = {
                'file': 'submit.csv',
                'url': self.submiturl if download else None
            }
            self.load_raw_file(**args)

            # Swap labels for the datasets (raw uses 1: unreliable, 0: reliable)
            data['label'] ^= 1

            # Store DataFrame
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
