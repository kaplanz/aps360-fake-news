import json
import logging
import os
import statistics
from multiprocessing import Process

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def train(model, train_loader, valid_loader, args):
    """Train the model."""
    criterion = get_criterion(args.loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Allocate space for training data
    train_acc, valid_acc, train_loss, valid_loss = [[None] * args.epochs
                                                    for _ in range(4)]

    # Determine starting and ending epoch
    start = 0
    end = args.epochs
    if args.load is not None:
        start += args.load + 1
        end += args.load
        # Load previous training data (if any)
        td = load_training_data(args)
        train_acc = td['train_acc'] + train_acc
        valid_acc = td['valid_acc'] + valid_acc
        train_loss = td['train_loss'] + train_loss
        valid_loss = td['valid_loss'] + valid_loss

    logging.info('Training for {} epochs'.format(args.epochs))
    # Outer training loop
    for epoch in range(start, end):
        model.train()  # set model to training mode

        correct, total = 0, 0
        losses = []
        # Inner training loop
        for text, labels in train_loader:
            out = model(text).flatten()
            loss = criterion(out, labels.float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # Record statistics for this batch
            pred = (out >
                    0.5).long()  # predict `true` for values greater than 0.5
            correct += pred.eq(labels).sum().item()
            total += labels.shape[0]
            losses.append(loss.item())

        # Evaluate model performance
        train_acc[epoch], train_loss[epoch] = (correct /
                                               total), statistics.mean(losses)
        valid_acc[epoch], valid_loss[epoch] = evaluate(model, valid_loader,
                                                       criterion)
        logging.debug(
            'Epoch {}: train_acc: {:.4%}, valid_acc: {:.4%}, train_loss: {:.6f}, valid_loss: {:.6f}'
            .format(epoch, train_acc[epoch], valid_acc[epoch],
                    train_loss[epoch], valid_loss[epoch]))
        # Save model checkpoint
        if args.save:
            path = get_model_path(epoch, args) + '.pt'
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(model.state_dict(), path)

    # Bundle training data
    td = {
        'train_acc': train_acc,
        'valid_acc': valid_acc,
        'train_loss': train_loss,
        'valid_loss': valid_loss
    }

    # Save training data
    if args.save:
        path = get_model_path(epoch, args) + '_data.json'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(td, f)
            logging.info('Saved training data to {}'.format(path))

    # Return training data
    return td


def evaluate(model, data_loader, criterion):
    """Evaluate the model."""
    model.eval()

    correct, total = 0, 0
    losses = []
    for text, labels in data_loader:
        out = model(text).flatten()
        loss = criterion(out, labels.float())
        pred = (out > 0.5).long()  # predict `true` for values greater than 0.5
        correct += pred.eq(labels).sum().item()
        total += labels.shape[0]
        losses.append(loss.item())
    return (correct / total), statistics.mean(losses)


def plot(td):
    """Plot training data."""
    procs = [
        Process(target=plot_training_curve,
                args=('Accuracy', td['train_acc'], td['valid_acc'])),
        Process(target=plot_training_curve,
                args=('Loss', td['train_loss'], td['valid_loss']))
    ]
    [proc.start() for proc in procs]
    [proc.join() for proc in procs]


def get_criterion(loss):
    """Returns loss function as specified in input."""
    return getattr(nn, loss)()


def get_model_path(epoch, args, root='./output'):
    """Generate a name for the model consisting of all the hyperparameter values

    Args:
        config: Configuration object containing the hyperparameters
    Returns:
        path: A string with the hyperparameter name and value concatenated
    """
    return root + '/{}_{}_ss{}_bs{}_lr{}_epoch{}'.format(
        args.model.lower(),
        args.loss.lower(),
        args.sample_size,
        args.batch_size,
        args.lr,
        epoch,
    )


def load_model(model, args):
    path = get_model_path(args.load, args) + '.pt'
    if os.path.isfile(path):
        model.load_state_dict(torch.load(path))
        logging.info('Successfully loaded pretrained model')
    else:
        logging.error('Pretrained model missing, expected at {}'.format(path))
        exit(1)


def load_training_data(args, allow_missing=True):
    """Load training data from a file."""
    path = get_model_path(args.load, args) + '_data.json'
    td = None
    if os.path.isfile(path):
        with open(path, 'r') as f:
            td = json.load(f)
            logging.info('Successfully loaded previous training data')
    elif allow_missing:
        logging.warning('Training data missing, continuing without')
    else:
        logging.error('Training data missing, expected at {}'.format(path))
        exit(1)
    return td


def plot_training_curve(attribute, dtrain, dvalid):
    """Plots the training curve for a model run."""
    n = len(dtrain)  # number of epochs
    plt.title('Train vs Validation {}'.format(attribute))
    plt.plot(range(n), dtrain, label='Train')
    plt.plot(range(n), dvalid, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel(attribute)
    plt.legend(loc='best')
    plt.show()
