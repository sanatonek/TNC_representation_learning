import torch
from torch.utils import data
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
import pickle
import os
import random

from tcl.models import RnnEncoder, MimicEncoder, WFEncoder
from tcl.utils import PatientData, plot_distribution, model_distribution

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Discriminator(torch.nn.Module):
    def __init__(self, input_size, device):
        super(Discriminator, self).__init__()
        self.device = device
        self.input_size = input_size
        # self.model = torch.nn.Sequential(torch.nn.Linear(2*self.input_size, 4*self.input_size),
        #                                  # torch.nn.BatchNorm1d(32),
        #                                  torch.nn.ReLU(),
        #                                  torch.nn.Linear(4*self.input_size, 1),
        #                                  torch.nn.Sigmoid())
        # torch.nn.init.xavier_uniform_(self.model[0].weight)
        # torch.nn.init.xavier_uniform_(self.model[2].weight)

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


class TCLDataset(data.Dataset):
    def __init__(self, x, mc_sample_size, epsilon, delta, window_size, augmentation, state=None):
        super(TCLDataset, self).__init__()
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
        # return len(self.time_series) * self.window_per_sample

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
        ## Autocorrelation
        # s = pd.Series(x[1,t-self.window_size//2:])
        # m = np.mean(x[1,t-self.window_size//2:t+self.window_size//2])
        # c = np.std(x[1,t-self.window_size//2:t+self.window_size//2])
        # corr = []
        # mean, cov = [], []
        # for i in range(0,30,5):
        #     mean.append(np.abs(np.mean(x[1,t+i-self.window_size//2:t+i+self.window_size//2])-m))
        #     cov.append(np.abs(np.std(x[1,t+i-self.window_size//2:t+i+self.window_size//2])-c))
        #     # corr.append(np.abs(s.autocorr(i)))
        # f, axs = plt.subplots(2)
        # axs[0].plot(mean)
        # axs[1].plot(cov)
        # plt.show()

        ## Random within a distance
        # t_p = np.random.randint(max(0, t - self.epsilon - self.window_size), min(t + self.window_size + self.epsilon, T - self.window_size), self.mc_sample_size)
        # t_p = np.random.randint(max(0, t - self.epsilon), min(t + self.window_size + self.epsilon, T - self.window_size), self.mc_sample_size)

        ## Random from a Gaussian
        t_p = [int(t+np.random.randn()*self.epsilon) for _ in range(self.mc_sample_size)]
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
        # d = torch.cat((d_p, d_n), 0)
        # d_label = torch.cat((neighbors, non_neighbors), 0)
        # shuffled_inds = torch.randperm(len(d))
        # d_label = d_label[shuffled_inds]
        # d = d[shuffled_inds]
        # loss = loss_fn(d, d_label)

        # ratio = torch.sum(disc_model(z_t, z_p), -1) / (torch.sum(disc_model(z_t, z_n), -1)+torch.sum(disc_model(z_t, z_p), -1))
        # loss = -1*torch.mean(torch.log(ratio))
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
        # if batch_count%100==0:
        #     print('\t Batch %d performance: '%(batch_count), '\t Accuracy: ', (epoch_acc/batch_count))
    return epoch_loss/batch_count, epoch_acc/batch_count


def learn_encoder(x, encoder, window_size, lr=0.001, decay=0.005, epsilon=20, delta=150, mc_sample_size=20,
                  n_epochs=100, path='simulation', device='cpu', augmentation=1):
    disc_model = Discriminator(encoder.encoding_size, device)
    n_train = int(len(x)*0.8)
    inds = list(range(len(x)))
    random.shuffle(inds)
    x = x[inds]

    params = list(disc_model.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=decay)
    performance = []
    best_acc = 0
    # inds = list(range(n_train))
    trainset = TCLDataset(x=torch.Tensor(x[:n_train]), mc_sample_size=mc_sample_size, epsilon=epsilon, delta=delta,
                          window_size=window_size, augmentation=augmentation)
    print('Train: ', len(trainset))
    train_loader = data.DataLoader(trainset, batch_size=5, shuffle=True, num_workers=3)
    validset = TCLDataset(x=torch.Tensor(x[n_train:]), mc_sample_size=mc_sample_size,
                          epsilon=epsilon, delta=delta, window_size=window_size, augmentation=augmentation)
    print('Validation: ', len(validset))
    valid_loader = data.DataLoader(validset, batch_size=5, shuffle=True)
    for epoch in range(n_epochs):

        epoch_loss, epoch_acc = epoch_run(train_loader, disc_model, encoder, optimizer=optimizer, train=True, device=device)
        test_loss, test_acc = epoch_run(valid_loader, disc_model, encoder, train=False,  device=device)
        performance.append((epoch_loss, test_loss, epoch_acc, test_acc))
        print('Epoch %d Loss =====> Training Loss: %.5f \t Training Accuracy: %.5f \t Test Loss: %.5f \t Test Accuracy: %.5f'
              % (epoch, epoch_loss, epoch_acc, test_loss, test_acc))
        if test_acc>best_acc:
            print('Saving a new best')
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'discriminator_state_dict': disc_model.state_dict(),
                'best_accuracy': best_acc
            }
            torch.save(state, './ckpt/%s/checkpoint.pth.tar'%path)

    # Save checkpoints
    if not os.path.exists('./ckpt/%s'%path):
        os.mkdir('./ckpt/%s'%path)
    torch.save(encoder.state_dict(), './ckpt/%s/tcl_encoder.pt'%path)
    torch.save(disc_model.state_dict(), './ckpt/%s/discriminator.pt'%path)

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
    plt.savefig(os.path.join("./plots/%s"%path, "loss.pdf"))
    plt.figure()
    plt.plot(np.arange(n_epochs), train_acc, label="Train")
    plt.plot(np.arange(n_epochs), test_acc, label="Test")
    plt.title("Accuracy")
    plt.legend()
    plt.savefig(os.path.join("./plots/%s"%path, "accuracy.pdf"))

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
        learn_encoder(torch.Tensor(x), encoder, lr=1e-3, decay=1e-5, window_size=window_size, epsilon=50,
                      delta=300, n_epochs=100, mc_sample_size=20, path='simulation', device=device, augmentation=5)

        with open(os.path.join(path, 'x_test.pkl'), 'rb') as f:
            x_test = pickle.load(f)
        with open(os.path.join(path, 'state_test.pkl'), 'rb') as f:
            y_test = pickle.load(f)
        plot_distribution(x_test, y_test, encoder, window_size=window_size, path='simulation', device=device)
        # exp = ClassificationPerformanceExperiment()
        # exp.run(data='simulation', n_epochs=70, lr_e2e=0.01, lr_cls=0.001)
        # track_encoding(sample=x_test[10], label=y_test[10], encoder=encoder, window_size=window_size, path='simulation')

    if is_train and data_type == 'mimic':
        p_data = PatientData()
        n_patient, n_features, length = p_data.train_data.shape
        encoder = MimicEncoder(input_size=2, in_channel=n_features, encoding_size=10)
        learn_encoder(p_data.train_data, encoder, window_size=2, delta=5, epsilon=2,
                      path='mimic', mc_sample_size=5)

    if is_train and data_type == 'wf':
        window_size = 2500
        path = './data/waveform_data/processed'
        with open(os.path.join(path, 'x_train.pkl'), 'rb') as f:
            x = pickle.load(f)
        with open(os.path.join(path, 'state_train.pkl'), 'rb') as f:
            y = pickle.load(f)

        # encoder = CausalCNNEncoder(in_channels=2, channels=30, depth=10, reduced_size=80,
        #          out_channels=64, kernel_size=3)
        encoder = WFEncoder(encoding_size=64).to(device)
        learn_encoder(torch.Tensor(x), encoder, lr=1e-5, decay=0.1, n_epochs=50, window_size=window_size, delta=400000,
                      epsilon=5000, path='waveform', mc_sample_size=25, device=device, augmentation=50)
        with open(os.path.join(path, 'x_test.pkl'), 'rb') as f:
            x_test = pickle.load(f)
        with open(os.path.join(path, 'state_test.pkl'), 'rb') as f:
            y_test = pickle.load(f)
        plot_distribution(x_test, y_test, encoder, window_size=window_size, path='waveform', device=device, augment=100)    # if is_train:
        model_distribution(x, y, x_test, y_test, encoder, window_size, 'waveform', device)

        # exp = WFClassificationExperiment(window_size=window_size)
        # exp.run(data='waveform', n_epochs=15, lr_e2e=0.001, lr_cls=0.001)


if __name__=='__main__':
    np.random.seed(1234)
    is_train = True
    main(is_train, data_type='simulation')


