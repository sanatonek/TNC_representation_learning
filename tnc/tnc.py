import torch
from torch.utils import data
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import seaborn as sns; sns.set()

import numpy as np
import pickle
import os
import random

from tnc.models import RnnEncoder, MimicEncoder, WFEncoder
from tnc.utils import PatientData, plot_distribution, model_distribution, track_encoding
from tnc.evaluations import WFClassificationExperiment, ClassificationPerformanceExperiment
from sklearn.model_selection import KFold

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Discriminator(torch.nn.Module):
    def __init__(self, input_size, device):
        super(Discriminator, self).__init__()
        self.device = device
        self.input_size = input_size

        self.model = torch.nn.Sequential(torch.nn.Linear(2*self.input_size, 4*self.input_size),
                                         torch.nn.ReLU(inplace=True),
                                         torch.nn.Dropout(0.5),
                                         # torch.nn.BatchNorm1d(4*self.input_size),
                                         torch.nn.Linear(4*self.input_size, 1),
                                         torch.nn.Sigmoid())

        torch.nn.init.xavier_uniform_(self.model[0].weight)
        torch.nn.init.xavier_uniform_(self.model[3].weight)

    def forward(self, x, x_tild):
        """
        Predict the probability of the two inputs belonging to the same neighbourhood.
        """
        x_all = torch.cat([x, x_tild], -1)
        p = self.model(x_all)
        return p.view((-1,))


class TNCDataset(data.Dataset):
    def __init__(self, x, mc_sample_size, epsilon, delta, window_size, augmentation, state=None):
        super(TNCDataset, self).__init__()
        self.time_series = x
        self.T = x.shape[-1]
        self.epsilon = epsilon
        self.delta = delta
        self.window_size = window_size
        self.sliding_gap = int(window_size*25.2)
        self.window_per_sample = (self.T-2*self.window_size)//self.sliding_gap
        self.mc_sample_size = mc_sample_size
        self.state = state
        self.augmentation = augmentation

    def __len__(self):
        return len(self.time_series)*self.augmentation

    def __getitem__(self, ind):
        ind = ind%len(self.time_series)
        t = np.random.randint(self.window_size+2*self.epsilon, self.T-self.window_size-2*self.epsilon)
        x_t = self.time_series[ind][:,t-self.window_size//2:t+self.window_size//2]  # TODO: add padding for windows that are smaller
        X_close = self._find_neighours(self.time_series[ind], t)
        X_distant = self._find_non_neighours(self.time_series[ind], t)

        # i = ind//self.window_per_sample
        # t = ind%self.window_per_sample + self.window_size//2 + np.random.randint(0,self.window_size//2)
        # x_t = self.time_series[i][:, t - self.window_size // 2:t + self.window_size // 2]
        # X_close = self._find_neighours(self.time_series[i], t)
        # X_distant = self._find_non_neighours(self.time_series[i], t)

        if self.state is None:
            y_t = -1
        else:
            y_t = np.round(np.mean(self.state[ind][t-self.window_size//2:t+self.window_size//2]))# self.state[t] # TODO: Maybe change this to the average of states within the window
        return x_t, X_close, X_distant, y_t

    def _find_neighours(self, x, t):
        T = self.time_series.shape[-1]
        ## Random within a distance
        # t_p = np.random.randint(max(0, t - self.epsilon - self.window_size), min(t + self.window_size + self.epsilon, T - self.window_size), self.mc_sample_size)
        # t_p = np.random.randint(max(0, t - self.epsilon), min(t + self.window_size + self.epsilon, T - self.window_size), self.mc_sample_size)

        ## Random from a Gaussian
        t_p = [int(t+np.random.randn()*self.epsilon*self.window_size) for _ in range(self.mc_sample_size)]
        t_p = [max(self.window_size//2+1,min(t_pp,T-self.window_size//2)) for t_pp in t_p]
        x_p = torch.stack([x[:, t_ind-self.window_size//2:t_ind+self.window_size//2] for t_ind in t_p])
        return x_p

    def _find_non_neighours(self, x, t):
        T = self.time_series.shape[-1]
        if t>T/2:
            t_n = np.random.randint(min(self.window_size//2+1, t - self.delta), (t - self.delta + 1), self.mc_sample_size)
        else:
            t_n = np.random.randint(min((t + self.delta), (T - self.window_size-1)), (T - self.window_size//2), self.mc_sample_size)
        x_n = torch.stack([x[:, t_ind-self.window_size//2:t_ind+self.window_size//2] for t_ind in t_n])
        return x_n


def epoch_run(loader, disc_model, encoder, device, optimizer=None, train=True):
    if train:
        encoder.train()
        disc_model.train()
    else:
        encoder.eval()
        disc_model.eval()
    loss_fn = torch.nn.BCELoss()
    encoder.to(device)
    disc_model.to(device)
    epoch_loss = 0
    epoch_acc = 0
    batch_count = 0
    for x_t, x_p, x_n, _ in loader:
        mc_sample = x_p.shape[1]
        batch_size, f_size, len_size = x_t.shape
        x_p = x_p.reshape((-1, f_size, len_size))
        x_n = x_n.reshape((-1, f_size, len_size))
        x_t = np.repeat(x_t, mc_sample, axis=0)
        neighbors = torch.ones((len(x_p))).to(device)
        non_neighbors = torch.zeros((len(x_n))).to(device)
        x_t, x_p, x_n = x_t.to(device), x_p.to(device), x_n.to(device)

        z_t = encoder(x_t)
        z_p = encoder(x_p)
        z_n = encoder(x_n)

        d_p = disc_model(z_t, z_p)
        d_n = disc_model(z_t, z_n)
        p_loss = loss_fn(d_p, neighbors)
        n_loss = loss_fn(d_n, non_neighbors)
        loss = (p_loss + 0.7*n_loss)/1.7

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        p_acc = torch.sum(d_p > 0.5).item() / len(z_p)
        n_acc = torch.sum(d_n < 0.5).item() / len(z_n)
        epoch_acc = epoch_acc + (p_acc+n_acc)/2
        epoch_loss += loss.item()
        batch_count += 1
    return epoch_loss/batch_count, epoch_acc/batch_count


def learn_encoder(x, encoder, window_size, lr=0.001, decay=0.005, epsilon=20, delta=150, mc_sample_size=20,
                  n_epochs=100, path='simulation', device='cpu', augmentation=1):
    for cv in range(4):
        if 'waveform' in path:
            encoder = WFEncoder(encoding_size=64).to(device)
        else:
            encoder = RnnEncoder(hidden_size=100, in_channel=3, encoding_size=10, device=device)
        disc_model = Discriminator(encoder.encoding_size, device)
        params = list(disc_model.parameters()) + list(encoder.parameters())
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=decay)
        inds = list(range(len(x)))
        random.shuffle(inds)
        x = x[inds]
        n_train = int(0.8*len(x))
        performance = []
        best_acc = 0
        best_loss = np.inf
        for epoch in range(n_epochs):

            trainset = TNCDataset(x=torch.Tensor(x[:n_train]), mc_sample_size=mc_sample_size, epsilon=epsilon, delta=delta,
                                  window_size=window_size, augmentation=augmentation)
            print('Train: ', len(trainset))
            train_loader = data.DataLoader(trainset, batch_size=5, shuffle=True, num_workers=3)
            validset = TNCDataset(x=torch.Tensor(x[n_train:]), mc_sample_size=mc_sample_size,
                                  epsilon=epsilon, delta=delta, window_size=window_size, augmentation=augmentation)
            print('Validation: ', len(validset))
            valid_loader = data.DataLoader(validset, batch_size=5, shuffle=True)

            epoch_loss, epoch_acc = epoch_run(train_loader, disc_model, encoder, optimizer=optimizer, train=True, device=device)
            test_loss, test_acc = epoch_run(valid_loader, disc_model, encoder, train=False,  device=device)
            performance.append((epoch_loss, test_loss, epoch_acc, test_acc))
            print('Epoch %d Loss =====> Training Loss: %.5f \t Training Accuracy: %.5f \t Test Loss: %.5f \t Test Accuracy: %.5f'
                  % (epoch, epoch_loss, epoch_acc, test_loss, test_acc))
            # if best_acc<test_acc:
            if best_loss > test_loss:
                print('Saving a new best')
                best_acc = test_acc
                best_loss = test_loss
                state = {
                    'epoch': epoch,
                    'encoder_state_dict': encoder.state_dict(),
                    'discriminator_state_dict': disc_model.state_dict(),
                    'best_accuracy': test_acc
                }
                torch.save(state, './ckpt/%s/checkpoint_%d.pth.tar'%(path,cv))

        # Save performance plots
        if not os.path.exists('./plots/%s'%path):
            os.mkdir('./plots/%s'%path)
        train_loss = [t[0] for t in performance]
        test_loss = [t[1] for t in performance]
        train_acc = [t[2] for t in performance]
        test_acc = [t[3] for t in performance]
        plt.figure()
        plt.plot(np.arange(n_epochs), train_loss, label="Train")
        plt.plot(np.arange(n_epochs), test_loss, label="Test")
        plt.title("Loss")
        plt.legend()
        plt.savefig(os.path.join("./plots/%s"%path, "loss_%d.pdf"%cv))
        plt.figure()
        plt.plot(np.arange(n_epochs), train_acc, label="Train")
        plt.plot(np.arange(n_epochs), test_acc, label="Test")
        plt.title("Accuracy")
        plt.legend()
        plt.savefig(os.path.join("./plots/%s"%path, "accuracy_%d.pdf"%cv))

    return encoder


def main(is_train, data_type):
    if not os.path.exists("./plots"):
        os.mkdir("./plots")
    if not os.path.exists("./ckpt/"):
        os.mkdir("./ckpt/")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if is_train and data_type=='simulation':
        window_size = 50
        encoder = RnnEncoder(hidden_size=100, in_channel=3, encoding_size=10, device=device)
        path = './data/simulated_data/'

        with open(os.path.join(path, 'x_train.pkl'), 'rb') as f:
            x = pickle.load(f)
        with open(os.path.join(path, 'state_train.pkl'), 'rb') as f:
            y = pickle.load(f)

        track_encoding(x[0,:,400:], y[0,400:], encoder, window_size, 'simulation')
        learn_encoder(x, encoder, lr=1e-3, decay=1e-5, window_size=window_size, epsilon=2., delta=300, n_epochs=80,
                      mc_sample_size=20, path='simulation', device=device, augmentation=5)

        ## Plot a sample
        # f, axes = plt.subplots(3, 1)
        # f.set_figheight(3)
        # f.set_figwidth(13)
        # color = [[0.6350, 0.0780, 0.1840], [0.4660, 0.6740, 0.1880], [0, 0.4470, 0.7410]]
        # for i, ax in enumerate(axes):
        #     ax.set(ylabel='Feature %d'%i, xlabel='time')
        #     ax.plot(x[10, i, :], c=color[i])
        #     for t in range(x[10, i, :].shape[-1]):
        #         ax.axvspan(t, min(t + 1, x.shape[-1] - 1), facecolor=['y', 'g', 'b', 'r'][y[10, t]],
        #                    alpha=0.6)
        # f.set_figheight(6)
        # f.set_figwidth(12)
        # plt.savefig('./simulation_sample.pdf')

        with open(os.path.join(path, 'x_test.pkl'), 'rb') as f:
            x_test = pickle.load(f)
        with open(os.path.join(path, 'state_test.pkl'), 'rb') as f:
            y_test = pickle.load(f)
        plot_distribution(x_test, y_test, encoder, window_size=window_size, path='simulation', title='Our Approach', device=device)
        model_distribution(x, y, x_test, y_test, encoder, window_size, 'simulation', device)
        exp = ClassificationPerformanceExperiment()
        exp.run(data='simulation', n_epochs=70, lr_e2e=0.01, lr_cls=0.001)

    if is_train and data_type == 'mimic':
        p_data = PatientData()
        n_patient, n_features, length = p_data.train_data.shape
        encoder = MimicEncoder(input_size=2, in_channel=n_features, encoding_size=10)
        learn_encoder(p_data.train_data, encoder, window_size=2, delta=5, epsilon=2,
                      path='mimic', mc_sample_size=5)

    if is_train and data_type == 'wf':
        window_size = 2500
        path = './data/waveform_data/processed'
        encoder = WFEncoder(encoding_size=64).to(device)
        with open(os.path.join(path, 'x_train.pkl'), 'rb') as f:
            x = pickle.load(f)
        T = x.shape[-1]
        x_window = np.concatenate(np.split(x[:, :, :T // 5 * 5], 5, -1), 0)
        # exp = WFClassificationExperiment(window_size=window_size)
        # exp.run(data='waveform', n_epochs=5, lr_e2e=0.001, lr_cls=0.001)
        learn_encoder(torch.Tensor(x_window), encoder, lr=1e-5, decay=1e-3, n_epochs=150, window_size=window_size, delta=400000,
                      epsilon=3, path='waveform', mc_sample_size=10, device=device, augmentation=5)

        with open(os.path.join(path, 'x_test.pkl'), 'rb') as f:
            x_test = pickle.load(f)
        with open(os.path.join(path, 'state_test.pkl'), 'rb') as f:
            y_test = pickle.load(f)
        plot_distribution(x_test, y_test, encoder, window_size=window_size, path='waveform', device=device, augment=100)    # if is_train:
        # model_distribution(None, None, x_test, y_test, encoder, window_size, 'waveform', device)

        exp = WFClassificationExperiment(window_size=window_size)
        exp.run(data='waveform', n_epochs=5, lr_e2e=0.001, lr_cls=0.001)


if __name__=='__main__':
    random.seed(1234)
    parser = argparse.ArgumentParser(description='Run TNC')
    parser.add_argument('--data', type=str, default='simulation')
    args = parser.parse_args()
    main(True, args.data)


