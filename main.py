#!/usr/bin/env python

import logging

import torchtext

import datasets
import utils


def main():
    # Configure logging
    logging.basicConfig(level=logging.DEBUG)

    # Determine which dataset to use
    dataset = {'FRN': datasets.frn.FRN}['FRN']()

    # Preload the dataset
    dataset.load()
    # Load GloVe vocabulary
    glove = torchtext.vocab.GloVe(name='6B', dim=50)

    # Get preprocessed vectors
    samples = utils.preprocess.get_samples(dataset, glove)

    # Create data loader
    train_loader, valid_loader, test_loader = utils.dataloader.get_loader(
        samples, train=.6, valid=.2, test=.2)


if __name__ == '__main__':
    main()
