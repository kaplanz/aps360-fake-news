import json
import logging
import os
import zlib

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

from . import dataloaders, datasets, models


def get_config(args):
    """Get run configuration from arguments."""
    return {
        'batch_size': args.batch_size,
        'data_loader': args.data_loader,
        'dataset': args.dataset,
        'learning_rate': args.learning_rate,
        'loss': args.loss,
        'model': args.model,
        'sample_size': args.sample_size,
        'seed': args.seed,
    }


def load_config(args):
    """Load configuration from a file."""
    path = args.config
    with open(path, 'r') as f:
        config = json.load(f)
        for k, v in config.items():
            setattr(args, k, v)


def save_config(args):
    """Save run configuration."""
    path = get_path(args) + '/config.json'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(get_config(args), f)


def get_criterion(loss):
    """Get loss function as specified in arguments."""
    return getattr(nn, loss)()


def get_data_loader(args):
    """Get data loader as specified in arguments."""
    return getattr(getattr(dataloaders, args.data_loader.lower()),
                   args.data_loader)


def get_dataset(args):
    """Get dataset as specified in arguments."""
    return getattr(getattr(datasets, args.dataset.lower()), args.dataset)()


def get_model(vocab, args):
    """Get model as specified in arguments."""
    return getattr(getattr(models, args.model.lower()), args.model)(vocab)


def load_model(epoch, model, args):
    """Load a model checkpoint from a file."""
    path = get_path(args) + '/models/epoch{}.pt'.format(epoch)
    if os.path.isfile(path):
        model.load_state_dict(torch.load(path))
        logging.info('Successfully loaded model')
    else:
        logging.error('Model checkpoint missing, expected at {}'.format(path))
        exit(1)


def save_model(epoch, model, args):
    """Save a model checkpoint to a file."""
    path = get_path(args) + '/models/epoch{}.pt'.format(epoch)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def get_path(args, root='./output'):
    """Generate a name for the model consisting of all the hyperparameter values."""
    return root + '/' + format(
        zlib.adler32(str(get_config(args)).encode('utf-8')), 'x')


def load_training_data(args, allow_missing=True):
    """Load training data from a file."""
    path = get_path(args) + '/training.json'
    td = None
    if os.path.isfile(path):
        with open(path, 'r') as f:
            td = json.load(f)
            logging.info('Successfully loaded training data')
    elif allow_missing:
        logging.warning('Training data missing, continuing without')
    else:
        logging.error('Training data missing, expected at {}'.format(path))
        exit(1)
    return td


def save_training_data(td, args, verbose=False):
    """Save training data to a file."""
    path = get_path(args) + '/training.json'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(td, f)
        if verbose:
            logging.info('Saved training data to {}'.format(path))


def plot_attribute(attribute, dtrain, dvalid):
    """Plots the training curve for a model run."""
    n = len(dtrain)  # number of epochs
    plt.title('Train vs Validation {}'.format(attribute))
    plt.plot(range(n), dtrain, label='Train')
    plt.plot(range(n), dvalid, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel(attribute)
    plt.legend(loc='best')
    plt.show()


def plot_confusion_matrix(y_true,
                          y_pred,
                          classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel='True label',
        xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(),
             rotation=45,
             ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j,
                    i,
                    format(cm[i, j], fmt),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
