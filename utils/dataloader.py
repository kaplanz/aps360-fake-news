import torchtext


def get_loader(train=0.6, valid=.2, test=.2):
    assert train + valid + test <= 1

    return [], [], []
