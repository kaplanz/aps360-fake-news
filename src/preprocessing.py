import logging
import os
import pickle
import re

import numpy as np
import torch
import tqdm


def get_samples(dataset, vocab, init=False):
    """Extract preprocessed samples from the dataset."""
    # Check if cleaning has already occurred
    path = dataset.path() + '/processed/samples.pkl'
    if os.path.isfile(path) and not init:
        # Load samples from pickle
        logging.info('Loading binary from {}'.format(path))
        with open(path, 'rb') as f:
            samples = pickle.load(f)
    else:
        # Load raw data for cleaning
        data = dataset.load()
        # Drop missing values
        data.dropna(subset=['text', 'label'], inplace=True)
        # Perform preprocessing
        clean(data)
        # Extract samples for PyTorch
        samples = encode(data, vocab)
        # Dump samples to pickle
        logging.info('Saving binary to {}'.format(path))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(samples, f)

    # Convert samples to tensors
    logging.debug('Converting samples to tensors')
    samples = [(torch.tensor(text).long(), label) for text, label in samples]

    return samples


def clean(data):
    """Clean the text samples for the model."""
    logging.info('Cleaning samples')

    # Extract raw samples to iterate over
    text = data['text']

    # Clean each sample individually
    for i, sample in tqdm.tqdm(text.iteritems(), total=text.size):
        # Keep only alphanumeric characters
        sample = re.sub(r'(?<! )(?=[^\s\w])|(?<=[^\s\w])(?! )', r' ', sample)
        # Convert to lowercase
        sample = sample.lower()
        # Strip title sequences "LONDON (Reuters) - "
        if len(sample.split('-', 1)[0]) < 50:
            sample = sample.split('-', 1)[-1]

        # Store cleaned samples in DataFrame
        data.at[i, 'clean'] = sample


def encode(data, vocab):
    """Encode the cleaned text samples using the vocabulary."""
    logging.info('Encoding samples')

    samples = []
    for _, row in tqdm.tqdm(data.iterrows(), total=data.shape[0]):
        samples.append((np.array([
            vocab.stoi[w] for w in row['clean'].split() if w in vocab.stoi
        ]), row['label']))

    return samples
