import os
import pickle
import numpy as np

from torch.utils import data
import torch


def create_simulated_dataset(window_size=50, path='./data/simulated_data/', batch_size=100):
    if not os.listdir(path):
        raise ValueError('Data does not exist')
    x = pickle.load(open(os.path.join(path, 'x_train.pkl'), 'rb'))
    y = pickle.load(open(os.path.join(path, 'state_train.pkl'), 'rb'))
    x_test = pickle.load(open(os.path.join(path, 'x_test.pkl'), 'rb'))
    y_test = pickle.load(open(os.path.join(path, 'state_test.pkl'), 'rb'))

    n_train = int(0.8*len(x))
    n_valid = len(x) - n_train
    n_test = len(x_test)
    x_train, y_train = x[:n_train], y[:n_train]
    x_valid, y_valid = x[n_train:], y[n_train:]

    datasets = []
    for set in [(x_train, y_train, n_train), (x_test, y_test, n_test), (x_valid, y_valid, n_valid)]:
        inds = np.random.randint(0, x.shape[-1] - window_size, set[2] * 4)
        windows = np.array([set[0][int(i % set[2]), :, ind:ind + window_size] for i, ind in enumerate(inds)])
        labels = [np.round(np.mean(set[1][i % set[2], ind:ind + window_size], axis=-1)) for i, ind in enumerate(inds)]
        datasets.append(data.TensorDataset(torch.Tensor(windows), torch.Tensor(labels)))

    trainset, testset, validset = datasets[0], datasets[1], datasets[2]
    train_loader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, sampler=None, batch_sampler=None,
                                   num_workers=0, collate_fn=None, pin_memory=False, drop_last=False)
    valid_loader = data.DataLoader(validset, batch_size=batch_size, shuffle=True, sampler=None, batch_sampler=None,
                                   num_workers=0, collate_fn=None, pin_memory=False, drop_last=False)
    test_loader = data.DataLoader(testset, batch_size=batch_size, shuffle=True, sampler=None, batch_sampler=None,
                                   num_workers=0, collate_fn=None, pin_memory=False, drop_last=False)

    return train_loader, valid_loader, test_loader


