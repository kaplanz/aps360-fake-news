import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def train(model, train_loader, valid_loader, epochs=30, learning_rate=1e-4, plot=False):
    """Train the model."""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_acc, valid_acc, train_loss, valid_loss = [[None] * epochs
                                                    for _ in range(4)]

    logging.info('Training for {} epochs'.format(epochs))
    for epoch in range(epochs):
        # Train model over all batches
        model.train()
        for text, labels in train_loader:
            out = model(text).flatten()
            loss = criterion(out, labels.float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Evaluate model performance
        train_acc[epoch], train_loss[epoch] = evaluate(model, train_loader, criterion)
        valid_acc[epoch], valid_loss[epoch] = evaluate(model, valid_loader, criterion)
        logging.debug('Epoch {}: train_acc: {:.4%}, valid_acc: {:.4%}, train_loss: {:.6f}, valid_loss: {:.6f}'.format(
            epoch, train_acc[epoch], valid_acc[epoch], train_loss[epoch], valid_loss[epoch]))

    if plot:
        # Print training curves
        plot_training_curve('Accuracy', train_acc, valid_acc)
        plot_training_curve('Loss', train_loss, valid_loss)


def evaluate(model, data_loader, criterion):
    """Evaluate the model."""
    model.eval()

    correct, total = 0, 0
    for text, labels in data_loader:
        out = model(text).flatten()
        loss = criterion(out, labels.float())
        pred = (out > 0.5).long()  # predict `true` for values greater than 0.5
        correct += pred.eq(labels).sum().item()
        total += labels.shape[0]
    return (correct / total), loss.item()


def get_model_name(batch_size, learning_rate, epoch, root='./output'):
    """Generate a name for the model consisting of all the hyperparameter values

    Args:
        config: Configuration object containing the hyperparameters
    Returns:
        path: A string with the hyperparameter name and value concatenated
    """
    return root + '/model_bs{}_lr{}_epoch{}'.format(batch_size, learning_rate,
                                                    epoch)


def plot_training_curve(title, dtrain, dvalid):
    """Plots the training curve for a model run, given the csv files
    containing the train/validation accuracy/loss.

    Args:
        path: The base path of the csv files produced during training
    """
    n = len(dtrain)  # number of epochs
    plt.title('Train vs Validation Accuracy')
    plt.plot(range(n), dtrain, label='Train')
    plt.plot(range(n), dvalid, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.show()
