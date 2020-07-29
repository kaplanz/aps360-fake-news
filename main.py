#!/usr/bin/env python

import logging
import os
import pathlib
import pickle

import torchtext

import datasets
import utils


def main():
    # Configure logging
    logging.basicConfig(level=logging.DEBUG)

    # Determine which dataset to use
    dataset = {'FRN': datasets.frn}['FRN']

    # Load the dataset and glove vocabulary
    data = dataset.get_dataset()
    glove = torchtext.vocab.GloVe(name='6B', dim=50)

    # Check if preprocessing has already occured
    path = dataset.path() + '/processed/samples.pkl'
    if (os.path.isfile(path)):
        # Load samples from pickle
        logging.info('Loading binary from {}'.format(path))
        with open(path, 'rb') as f:
            samples = pickle.load(f)
    else:
        # Perform preprocessing
        utils.preprocess.preprocess(data)
        # Extract samples for PyTorch
        samples = utils.preprocess.get_samples(data, glove)
        # Dump samples to pickle
        logging.info('Saving binary to {}'.format(path))
        pathlib.Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(samples, f)

    # Create data loader
    train_loader, valid_loader, test_loader = utils.dataloader.get_loader(
        train=.6, valid=.2, test=.2)


if __name__ == '__main__':
    main()
