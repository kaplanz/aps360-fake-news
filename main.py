#!/usr/bin/env python

import logging
import random

import torch
import torchtext

import datasets
import models
import utils

# Configure logger
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.WARNING)


def main():
    # Set random seeds
    random.seed(0)
    torch.manual_seed(0)

    # Determine which dataset to use
    dataset = {
        'FRN': datasets.frn.FRN
    }['FRN']()  # yapf: disable

    # Preload the dataset
    dataset.load()
    # Load GloVe vocabulary
    glove = torchtext.vocab.GloVe(name='6B', dim=50)

    # Get preprocessed vectors
    samples = utils.preprocess.get_samples(dataset, glove)
    random.shuffle(samples)
    samples = samples[:1000]  # limit training samples

    # Create data loaders
    train_loader, valid_loader, test_loader = utils.dataloader.get_loaders(
        samples, batch_size=32)

    # Create the model
    model = {
        'FakeNewsNet': models.fakenewsnet.FakeNewsNet
    }['FakeNewsNet'](glove)  # yapf: disable

    # Train the model
    utils.train.train(model,
                      train_loader,
                      valid_loader,
                      epochs=30,
                      learning_rate=1e-4)


if __name__ == '__main__':
    main()
