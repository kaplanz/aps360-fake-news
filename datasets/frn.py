import logging
import os
import pickle
import re

import pandas as pd


def path(root='./data'):
    return root + '/FRN'


def get_dataset(root='./data'):
    # Load or parse the data
    fake = load_from_file(path(root) + '/raw/Fake.csv')
    true = load_from_file(path(root) + '/raw/True.csv')

    # Create labels for the datasets
    fake['label'] = 0
    true['label'] = 1

    # Construct DataFrame
    return pd.concat([fake, true])


def load_from_file(path):
    pklpath = path.replace('csv', 'pkl')

    # Load from pickle
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
