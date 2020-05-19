import torch
from torch.utils import data

import numpy as np
import random
import os
import pickle
import matplotlib.pyplot as plt

from tcl.models import RnnEncoder, WFEncoder
from tcl.utils import plot_distribution
from tcl.evaluations import ClassificationPerformanceExperiment, WFClassificationExperiment


class CPCDataset(data.Dataset):
    def __init__(self, x, window_size, state=None):
        super(CPCDataset, self).__init__()
        self.time_series = x
        self.T = x.shape[-1]
        self.state = state
        self.window_size = window_size

    def __len__(self):
        return len(self.time_series)

    def __getitem__(self, ind):
        windows = np.split(self.time_series[ind, :, self.T//self.window_size*self.window_size], self.window_size)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def epoch_run(x, ds_estimator, encoder, device, window_size, optimizer=None, train=True):
    if train:
        encoder.train()
        ds_estimator.train()
    else:
        encoder.eval()
        ds_estimator.eval()
    encoder.to(device)
    ds_estimator.to(device)

    epoch_loss = 0
    acc = 0
    for sample in x:
        T = sample.shape[-1]
        windowed_sample = np.split(sample[:, :(T // window_size) * window_size], (T // window_size), -1)
        windowed_sample = torch.tensor(windowed_sample, device=device, requires_grad=True).float()
        encodings = encoder(windowed_sample)
        # print(encodings.shape)
        window_ind = torch.randint(0,len(encodings)-2, size=(1,))
        density_ratios = torch.matmul(encodings, ds_estimator(encodings[window_ind]).permute(1,0))
        # print(torch.argmax(density_ratios, 0), window_ind)
        # print(density_ratios.shape)
        rnd_n = np.random.choice(list(set(range(len(encodings) - 2)) - {window_ind[0] - 1, window_ind[0], window_ind[0] + 1}), 5)
        X_N = torch.cat([density_ratios[rnd_n], density_ratios[window_ind[0] + 1].unsqueeze(0)], 0)
        if torch.argmax(X_N)==len(X_N)-1:
            acc += 1
        labels = torch.Tensor([len(X_N)-1]).to(device)
        # print(X_N.view(1, -1), labels.long())
        loss = torch.nn.CrossEntropyLoss()(X_N.view(1, -1), labels.long())
        # loss = -torch.log(X_N[-1] / torch.sum(X_N))
        epoch_loss += loss.item()
        # print(loss)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return epoch_loss / len(x), acc/len(x)


def learn_encoder(x, encoder, window_size, lr=0.001, decay=0,
                  n_epochs=50, data='simulation', device='cpu'):
    n_train = int(len(x)*0.8)
    inds = list(range(len(x)))
    random.shuffle(inds)
    x = x[inds]
    ds_estimator = torch.nn.Linear(encoder.encoding_size, encoder.encoding_size)
    params = list(ds_estimator.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=decay)

    train_loss, test_loss = [], []
    for epoch in range(n_epochs):
        epoch_loss, acc = epoch_run(x[:n_train], ds_estimator, encoder, device, window_size, optimizer=optimizer, train=True)
        epoch_loss_test, acc_test = epoch_run(x[n_train:], ds_estimator, encoder, device, window_size, optimizer=optimizer, train=False)
        print('\nEpoch ', epoch)
        print('Train ===> Loss: ', epoch_loss, '\t Accuracy: ', acc)
        print('Test ===> Loss: ', epoch_loss_test, '\t Accuracy: ', acc_test)
        train_loss.append(epoch_loss)
        test_loss.append(epoch_loss_test)
    plt.figure()
    plt.plot(np.arange(n_epochs), train_loss, label="Train")
    plt.plot(np.arange(n_epochs), test_loss, label="Test")
    plt.title("CPC Loss")
    plt.legend()
    plt.savefig(os.path.join("./plots/%s_cpc/loss.pdf"%data))
    state = {
        'epoch': epoch,
        'encoder_state_dict': encoder.state_dict(),
        'density_estimator_dict': ds_estimator.state_dict(),
    }
    torch.save(state, './ckpt/%s_cpc/checkpoint.pth.tar'%data)
    if data=='simulation':
        path = './data/simulated_data/'
    else:
        path = './data/waveform_data/processed'
    with open(os.path.join(path, 'x_test.pkl'), 'rb') as f:
        x_test = pickle.load(f)
    with open(os.path.join(path, 'state_test.pkl'), 'rb') as f:
        y_test = pickle.load(f)
    plot_distribution(x_test, y_test, encoder, window_size=window_size, path='%s_cpc'%data, device=device)
    exp = ClassificationPerformanceExperiment(path='%s_cpc'%data)
    exp.run(data='%s_cpc'%data, n_epochs=70, lr_e2e=0.01, lr_cls=0.01)

data='simulation'

if data =='waveform':
    path = './data/waveform_data/processed'
    encoding_size = 64
    window_size = 2500
    encoder = WFEncoder(encoding_size=64).to(device)
    with open(os.path.join(path, 'x_train.pkl'), 'rb') as f:
        x = pickle.load(f)
    x = x[:, :, :50000]
    learn_encoder(x, encoder, window_size, n_epochs=50, lr=1e-5, decay=1e-3, data=data)
    with open(os.path.join(path, 'x_test.pkl'), 'rb') as f:
        x_test = pickle.load(f)
    with open(os.path.join(path, 'state_test.pkl'), 'rb') as f:
        y_test = pickle.load(f)
    plot_distribution(x_test, y_test, encoder, window_size=window_size, path='%s_cpc' % data, device=device)
    exp = WFClassificationExperiment(window_size=window_size)
    exp.run(data='%s_cpc'%data, n_epochs=15, lr_e2e=0.001, lr_cls=0.001)


else:
    path = './data/simulated_data/'
    window_size = 50
    encoder = RnnEncoder(hidden_size=100, in_channel=3, encoding_size=10, device=device)
    with open(os.path.join(path, 'x_train.pkl'), 'rb') as f:
        x = pickle.load(f)
    # learn_encoder(x, encoder, window_size, lr=1e-3, decay=1e-3, data=data, device=device)
    with open(os.path.join(path, 'x_test.pkl'), 'rb') as f:
        x_test = pickle.load(f)
    with open(os.path.join(path, 'state_test.pkl'), 'rb') as f:
        y_test = pickle.load(f)
    plot_distribution(x_test, y_test, encoder, window_size=window_size, path='%s_cpc' % data, device=device)
    exp = ClassificationPerformanceExperiment()
    exp.run(data='%s_cpc'%data, n_epochs=70, lr_e2e=0.01, lr_cls=0.01)


# window_size = 50
# data = 'waveform'
#
# if data =='waveform':
#     path = './data/waveform_data/processed'
#     encoding_size = 64
#     window_size = 2500
#     encoder = WFEncoder(encoding_size=64).to(device)
# else:
#     path = './data/simulated_data/'
#     window_size = 50
#     encoder = RnnEncoder(hidden_size=100, in_channel=3, encoding_size=10, device='cpu')
#
# with open(os.path.join(path, 'x_train.pkl'), 'rb') as f:
#     x = pickle.load(f)
#
# x = x[:,:,:50000]
#
#
# print('Dataset Shape: ', x.shape)
# learn_encoder(x, encoder, window_size, lr=1e-5, data=data)
