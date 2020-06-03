import torch
from torch.utils import data

import numpy as np
import random
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.model_selection import KFold

from tcl.models import RnnEncoder, WFEncoder
from tcl.utils import plot_distribution, model_distribution
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


def epoch_run(data, ds_estimator, encoder, device, window_size, optimizer=None, train=True):
    if train:
        encoder.train()
        ds_estimator.train()
    else:
        encoder.eval()
        ds_estimator.eval()
    encoder.to(device)
    ds_estimator.to(device)

    dataset = torch.utils.data.TensorDataset(torch.Tensor(data), torch.zeros((len(data), 1)))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=20, shuffle=True, sampler=None, batch_sampler=None,
                                              num_workers=0, collate_fn=None, pin_memory=False, drop_last=False)
    print(data.shape)
    epoch_loss = 0
    acc = 0
    # for x,_ in data_loader:
    for sample in data:
        # loss_var = torch.zeros(size=(1,), requires_grad=True).to(device)

        # for sample in x:
        start = np.random.randint(0,10)
        sample = torch.Tensor(sample[:,start:])
        T = sample.shape[-1]

        windowed_sample = np.split(sample[:, :(T // window_size) * window_size], (T // window_size), -1)
        windowed_sample = torch.tensor(np.stack(windowed_sample, 0), device=device)
        encodings = encoder(windowed_sample)
            # print(encodings.shape)
        window_ind = torch.randint(2,len(encodings)-2, size=(1,))
            # density_ratios = torch.matmul(encodings, ds_estimator(encodings[window_ind]).permute(1,0))
        density_ratios = torch.bmm(encodings.unsqueeze(1),
                                       ds_estimator(encodings[window_ind]).expand_as(encodings).unsqueeze(-1))
        r = set(range(0, window_ind[0] - 1))
        r.update(set(range(window_ind[0] + 2, len(encodings))))
        rnd_n = np.random.choice(list(r), 5)
        X_N = torch.cat([density_ratios[rnd_n], density_ratios[window_ind[0] + 1].unsqueeze(0)], 0)
        if torch.argmax(X_N)==len(X_N)-1:
            acc += 1
        labels = torch.Tensor([len(X_N)-1]).to(device)
            # print(X_N.view(1, -1), labels.long())
        loss = torch.nn.CrossEntropyLoss()(X_N.view(1, -1), labels.long())
        # loss_var = loss_var + loss
            # labels = torch.zeros((len(X_N),)).to(device)
            # labels[-1] = 1
            # loss = torch.nn.BCEWithLogitsLoss()(X_N.view(-1,), labels)
            # loss = -torch.log(X_N[-1] / torch.sum(X_N))
        epoch_loss += loss.item()
        # print(loss.item())

        if train:
            optimizer.zero_grad()
            # loss_var.backward()
            loss.backward()
            optimizer.step()
    return epoch_loss / len(data), acc/(len(data))


def learn_encoder(x, encoder, window_size, lr=0.001, decay=0,
                  n_epochs=50, data='simulation', device='cpu'):
    # n_train = int(len(x)*0.8)
    # inds = list(range(len(x)))
    # random.shuffle(inds)
    # x = x[inds]

    ds_estimator = torch.nn.Linear(encoder.encoding_size, encoder.encoding_size)
    params = list(ds_estimator.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=decay)

    # cv = 0
    # kf = KFold(n_splits=4)
    # for train_index, test_index in kf.split(x):
    for cv in range(4):
        inds = list(range(len(x)))
        random.shuffle(inds)
        x = x[inds]
        n_train = int(0.8*len(x))
        performance = []
        best_acc = 0
        best_loss = np.inf
        train_loss, test_loss = [], []
        best_loss = np.inf
        best_loss = np.inf
        train_loss, test_loss = [], []
        for epoch in range(n_epochs):
            epoch_loss, acc = epoch_run(x[:n_train], ds_estimator, encoder, device, window_size, optimizer=optimizer,
                                        train=True)
            epoch_loss_test, acc_test = epoch_run(x[n_train:], ds_estimator, encoder, device, window_size,
                                                  optimizer=optimizer, train=False)
            # epoch_loss, acc = epoch_run(x[:n_train], ds_estimator, encoder, device, window_size, optimizer=optimizer, train=True)
            # epoch_loss_test, acc_test = epoch_run(x[n_train:], ds_estimator, encoder, device, window_size, optimizer=optimizer, train=False)
            print('\nEpoch ', epoch)
            print('Train ===> Loss: ', epoch_loss, '\t Accuracy: ', acc)
            print('Test ===> Loss: ', epoch_loss_test, '\t Accuracy: ', acc_test)
            train_loss.append(epoch_loss)
            test_loss.append(epoch_loss_test)
            if epoch_loss_test<best_loss:
                print('Save new ckpt')
                state = {
                    'epoch': epoch,
                    'encoder_state_dict': encoder.state_dict()
                }
                best_loss = epoch_loss_test
                torch.save(state, './ckpt/%s_cpc/checkpoint_%d.pth.tar' %(data, cv))
        plt.figure()
        plt.plot(np.arange(n_epochs), train_loss, label="Train")
        plt.plot(np.arange(n_epochs), test_loss, label="Test")
        plt.title("CPC Loss")
        plt.legend()
        plt.savefig(os.path.join("./plots/%s_cpc/loss_%d.pdf"%(data, cv)))


def main(data):
    if data =='waveform':
        path = './data/waveform_data/processed'
        encoding_size = 64
        window_size = 2500
        encoder = WFEncoder(encoding_size=encoding_size).to(device)
        with open(os.path.join(path, 'x_train.pkl'), 'rb') as f:
            x = pickle.load(f)
        # with open(os.path.join(path, 'state_train.pkl'), 'rb') as f:
        #     y = pickle.load(f)
        T = x.shape[-1]
        x_window = np.concatenate(np.split(x[:, :, :T // 5 * 5], 5, -1), 0)
        # y_window = np.concatenate(np.split(y[:, :5 * (T // 5)], 5, -1), 0).astype(int)
        learn_encoder(x, encoder, window_size, n_epochs=50, lr=1e-5, decay=1e-3, data=data)

        # y_window = np.array([np.bincount(yy).argmax() for yy in y_window])
        # shiffled_inds = list(range(len(x_window)))
        # random.shuffle(shiffled_inds)
        # x_window = x_window[shiffled_inds]
        # y_window = y_window[shiffled_inds]
        # cv = 0
        # for train_index, test_index in kf.split(x_window):
        #     X_train, X_test = x_window[train_index], x_window[test_index]
        #     y_train, y_test = y_window[train_index], y_window[test_index]
        #     print(X_train.shape, y_train.shape)
        #     learn_encoder(X_train, encoder, window_size, n_epochs=50, lr=1e-5, decay=1e-3, data=data, cv=cv)
        #     cv += 1
        #     if cv>0:
        #         break
        # T = x.shape[-1]
        # x = np.concatenate(np.split(x[:, :, :T // 20 * 20], 20, -1), 0)
        # learn_encoder(x, encoder, window_size, n_epochs=50, lr=1e-5, decay=1e-3, data=data)
        # with open(os.path.join(path, 'x_test.pkl'), 'rb') as f:
        #     x_test = pickle.load(f)
        # with open(os.path.join(path, 'state_test.pkl'), 'rb') as f:
        #     y_test = pickle.load(f)
        # plot_distribution(x_test, y_test, encoder, window_size=window_size, path='%s_cpc' % data, device=device, augment=100)
        # model_distribution(None, None, x_test, y_test, encoder, window_size, 'waveform', device)
        # exp = WFClassificationExperiment(window_size=window_size)
        # exp.run(data='%s_cpc'%data, n_epochs=15, lr_e2e=0.001, lr_cls=0.001)

    else:
        path = './data/simulated_data/'
        kf = KFold(n_splits=4)
        window_size = 50
        encoder = RnnEncoder(hidden_size=100, in_channel=3, encoding_size=10, device=device)
        with open(os.path.join(path, 'x_train.pkl'), 'rb') as f:
            x = pickle.load(f)
        with open(os.path.join(path, 'state_train.pkl'), 'rb') as f:
            y = pickle.load(f)
        learn_encoder(x, encoder, window_size, n_epochs=50, lr=1e-5, decay=1e-3, data=data, device=device)


        # # learn_encoder(x, encoder, window_size, lr=1e-3, decay=0.0001, data=data, n_epochs=100, device=device)
        # with open(os.path.join(path, 'x_test.pkl'), 'rb') as f:
        #     x_test = pickle.load(f)
        # with open(os.path.join(path, 'state_test.pkl'), 'rb') as f:
        #     y_test = pickle.load(f)
        # # plot_distribution(x_test, y_test, encoder, window_size=window_size, path='%s_cpc' % data, title='CPC', device=device)
        # model_distribution(x, y, x_test, y_test, encoder, window_size, 'simulation_cpc', device)
        # # exp = ClassificationPerformanceExperiment(path='simulation_cpc')
        # # exp.run(data='%s_cpc'%data, n_epochs=70, lr_e2e=0.01, lr_cls=0.001)

if __name__=="__main__":
    data = 'waveform'
    random.seed(1234)
    main(data)
