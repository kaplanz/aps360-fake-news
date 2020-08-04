#!/usr/bin/env python
#
#  main.py
#  Fake News Classifier.
#
#  Created on 2020-07-30.
#

import argparse
import logging
import os
import random

import torch
import torchtext

import src.dataloaders as dataloaders
import src.datasets as datasets
import src.models as models
import src.utils as utils


def main():
    # Configure logger
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    # yapf: disable
    parser = argparse.ArgumentParser(description='Fake News Classifier')
    # Modes
    parser.add_argument('--init', action='store_true', default=False,
                        help='perform initialization')
    parser.add_argument('--train', action='store_true', default=False,
                        help='train the model')
    parser.add_argument('--test', action='store_true', default=False,
                        help='test the model (must either train or load a model)')
    parser.add_argument('--plot', action='store_true', default=False,
                        help='plot training data (must either train or load training data)')
    # Options
    parser.add_argument('-b', '--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--data-loader', type=str, default='BatchLoader',
                        help='data loader to use (default: "BatchLoader")')
    parser.add_argument('--dataset', type=str, default='FakeRealNews',
                        help='dataset to use (default: "FakeRealNews")')
    parser.add_argument('-e', '--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', '--learning-rate', type=float, default=1e-4,
                        help='learning rate (default: 1e-4)')
    parser.add_argument('-l', '--load', type=int, metavar='EPOCH',
                        help='load a model and its training data')
    parser.add_argument('--loss', type=str, default='BCEWithLogitsLoss',
                        help='loss function for training (default: "BCEWithLogitsLoss")')
    parser.add_argument('--model', type=str, default='FakeNewsNet',
                        help='model architecture to train (default: "FakeNewsNet")')
    parser.add_argument('-s', '--sample-size', type=int, metavar='N',
                        help='sample size for training')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 0)')
    parser.add_argument('--save', action='store_true', default=True,
                        help='save the model and its training data (default: True)')
    parser.add_argument('--no-save', dest='save', action='store_false')
    parser.add_argument('--truncate', type=int, metavar='N',
                        help='number of words after which samples are truncated')
    args = parser.parse_args()
    # yapf: enable

    # Exit if no mode is specified
    if not args.init and not args.train and not args.test and not args.plot:
        logging.error(
            'No mode specified. Please choose one of "--train", "--test", "--plot"'
        )
        exit(1)

    # Set random seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Variable declarations
    training_data = None

    # Perform initialization
    if args.init or args.train or args.test:
        # Determine which dataset to use
        dataset = getattr(getattr(datasets, args.dataset.lower()),
                          args.dataset)()

        # Preload the dataset
        dataset.load()
        # Load GloVe vocabulary
        glove = torchtext.vocab.GloVe(name='6B', dim=50)

        # Get preprocessed vectors
        samples = utils.preprocessing.get_samples(dataset, glove, args.init)
        random.shuffle(samples)

    # Setup for train, test
    if args.train or args.test:
        # Select data loader to use
        DataLoader = getattr(getattr(dataloaders, args.data_loader.lower()),
                             args.data_loader)

        # Split samples
        split_ratio = [.6, .2, .2]
        trainset, validset, testset = list(
            DataLoader.splits(samples, split_ratio))
        if args.sample_size is not None:  # limit samples used in training
            trainset = trainset[:args.sample_size]
            validset = validset[:int(args.sample_size * split_ratio[1] /
                                     split_ratio[0])]

        # Get data loaders
        train_loader, valid_loader, test_loader = [
            DataLoader(split, batch_size=args.batch_size)
            for split in [trainset, validset, testset]
        ]

        # Create the model
        model = getattr(getattr(models, args.model.lower()), args.model)(glove)

        # Load a pretrained model
        if args.load is not None:
            utils.training.load_model(model, args)

    # Run "--train"
    if args.train:
        training_data = utils.training.train(model, train_loader, valid_loader,
                                             args)

    # Run "--test"
    if args.test:
        if args.train or args.load is not None:
            criterion = utils.training.get_criterion(args.loss)
            acc, loss = utils.training.evaluate(model, test_loader, criterion)
            logging.info('Testing accuracy: {:.4%}, loss: {:.6f}'.format(
                acc, loss))
        else:
            logging.error('No model loaded for testing')
            exit(1)

    # Run "--plot"
    if args.plot:
        if training_data is None:
            if args.load:
                training_data = utils.training.load_training_data(
                    args, allow_missing=False)
            else:
                logging.error('No training data loaded for plotting')
                exit(1)

        logging.info('Plotting training data')
        utils.training.plot(training_data)


if __name__ == '__main__':
    main()
