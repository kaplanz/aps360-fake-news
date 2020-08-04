import logging
import statistics
from multiprocessing import Process

import torch
import torch.optim as optim

from . import utils


def train(model, train_loader, valid_loader, args):
    """Train the model."""
    criterion = utils.get_criterion(args.loss)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Allocate space for training data
    train_acc, valid_acc, train_loss, valid_loss = [[None] * args.epochs
                                                    for _ in range(4)]

    # Determine starting and ending epoch
    start = 0
    end = args.epochs
    if args.load is not None:
        start += args.load + 1
        end += args.load
        # Load training data (if any)
        td = utils.load_training_data(args)
        if start < len(td['train_acc']):
            logging.warning(
                'Training data will be overwritten from epoch {}'.format(
                    start))
        train_acc = td['train_acc'][:start] + train_acc
        valid_acc = td['valid_acc'][:start] + valid_acc
        train_loss = td['train_loss'][:start] + train_loss
        valid_loss = td['valid_loss'][:start] + valid_loss

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
            utils.save_model(epoch, model, args)

        # Condense and save training data
        td = {
            'train_acc': train_acc,
            'valid_acc': valid_acc,
            'train_loss': train_loss,
            'valid_loss': valid_loss
        }
        if args.save:
            utils.save_training_data(td, args, silent=True)

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
        Process(target=utils.plot_attribute,
                args=('Accuracy', td['train_acc'], td['valid_acc'])),
        Process(target=utils.plot_attribute,
                args=('Loss', td['train_loss'], td['valid_loss']))
    ]
    [proc.start() for proc in procs]
    [proc.join() for proc in procs]