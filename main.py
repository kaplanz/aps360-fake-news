#!/usr/bin/env python

import logging
import random

import torchtext

import datasets
import utils


def main():
    # Configure logging
    logging.basicConfig(level=logging.DEBUG)

    # Set random seeds
    random.seed(0)

    # Determine which dataset to use
    dataset = {'FRN': datasets.frn.FRN}['FRN']()

    # Preload the dataset
    dataset.load()
    # Load GloVe vocabulary
    glove = torchtext.vocab.GloVe(name='6B', dim=50)

    # Get preprocessed vectors
    samples = utils.preprocess.get_samples(dataset, glove)

    # Create data loaders
    train_loader, valid_loader, test_loader = utils.dataloader.get_loaders(
        samples)


if __name__ == '__main__':
    main()
