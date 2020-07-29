import logging
import re

import numpy as np
import pandas as pd
import tqdm


def preprocess(data):
    # Extract the text from the data
    text = data['text']

    # Preprocess data samples
    logging.info('Processing samples')
    for i, sample in tqdm.tqdm(text.iteritems(), total=text.size):
        # Keep only alphanumeric characters
        sample = re.sub('(?<! )(?=[.,!?()])|(?<=[.,!?()])(?! )', r' ', sample)
        # Convert to lowercase
        sample = sample.lower()

        # Store in dataframe
        data.at[i, 'clean'] = sample


def get_samples(data, vocab):
    logging.info('Extracting cleaned samples')

    samples = []
    for _, row in tqdm.tqdm(data.iterrows(), total=data.shape[0]):
        samples.append((np.array([
            vocab.stoi[w] for w in row['clean'].split() if w in vocab.stoi
        ]), row['label']))

    return samples
