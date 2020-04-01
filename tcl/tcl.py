import torch
from torch.utils import data
from torch.autograd import Variable
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

import numpy as np
import pickle
import os
import pandas as pd

from tcl.models import RnnEncoder
from tcl.evaluations import ClassificationPerformanceExperiment


class Discriminator(torch.nn.Module):
    def __init__(self, input_size, device):
        super(Discriminator, self).__init__()
        self.device = device
        self.input_size = input_size
        self.model = torch.nn.Sequential(torch.nn.Linear(2*self.input_size, 32),
                                         torch.nn.ReLU(),
                                         torch.nn.Linear(32, 1),
                                         torch.nn.Sigmoid())

    def forward(self, x, x_tild):
        """
        Predict the probability of the two inputs belonging to the same neighbourhood.
        """
        x_all = torch.cat([x, x_tild], -1)
        p = self.model(x_all)
        return p.view((-1,))


class TCLDataset(data.Dataset):
    def __init__(self, x, mc_sample_size, neighbouring_delta, window_size, state=None):
        super(TCLDataset, self).__init__()
        self.time_series = x
        self.delta = neighbouring_delta
        self.window_size = window_size
        self.mc_sample_size = mc_sample_size
        self.state = state

    def __len__(self):
        return len(self.time_series)

    def __getitem__(self, ind):
        T = self.time_series.shape[-1]
        t = np.random.randint(self.delta, T-self.window_size-self.delta)
        x_t = self.time_series[ind,:,t:t+self.window_size]
        X_close = self._find_neighours(self.time_series[ind], t)
        X_distant = self._find_non_neighours(self.time_series[ind], t)
        if self.state is None:
            y_t = -1
        else:
            y_t = self.state[t]
        return x_t, X_close, X_distant, y_t

    def _find_neighours(self, x, t):
        T = self.time_series.shape[-1]
        t_p = np.random.randint(max(0, t - self.delta - self.window_size), min(t + self.window_size + self.delta, T - self.window_size), self.mc_sample_size)
        x_p = torch.stack([x[:, t_ind:t_ind + self.window_size] for t_ind in t_p])
        return x_p

    def _find_non_neighours(self, x, t):
        T = self.time_series.shape[-1]
        if t>T/2:
            t_n = np.random.randint(min(0, t - 5*self.delta), (t - 5*self.delta + 1), self.mc_sample_size)
        else:
            t_n = np.random.randint((t + 5*self.delta), (T - self.window_size), self.mc_sample_size)
        x_n = torch.stack([x[:, t_ind:t_ind + self.window_size] for t_ind in t_n])
        return x_n


def train(train_loader, disc_model, optimizer, encoder):
    encoder.train()
    disc_model.train()
    loss_fn = torch.nn.BCELoss()
    epoch_loss = 0
    epoch_acc = 0
    batch_count = 0
    for x_t, x_p, x_n, _ in train_loader:
        # print(x_p.shape, x_n.shape, x_t.shape)
        mc_sample = x_p.shape[1]
        batch_size, f_size, len_size = x_t.shape
        x_p = x_p.reshape((-1, f_size, len_size))
        x_n = x_n.reshape((-1, f_size, len_size))
        x_t = np.repeat(x_t, mc_sample, axis=0)
        # print(x_p.shape, x_n.shape, x_t.shape)
        neighbors = torch.ones((len(x_p)))
        non_neighbors = torch.zeros((len(x_n)))

        optimizer.zero_grad()
        z_t = encoder(x_t)
        z_p = encoder(x_p)
        z_n = encoder(x_n)
        p_loss = loss_fn(disc_model(z_t, z_p), neighbors)
        n_loss = loss_fn(disc_model(z_t, z_n), non_neighbors)
        loss = (p_loss + n_loss) / 2
        loss.backward()
        optimizer.step()
        p_acc = torch.sum(disc_model(z_t, z_p) > 0.5).item() / len(z_p)
        n_acc = torch.sum(disc_model(z_t, z_n) < 0.5).item() / len(z_n)
        epoch_acc = epoch_acc + (p_acc+n_acc)/2
        epoch_loss += loss.item()
        batch_count += 1
    return epoch_loss/batch_count, epoch_acc/batch_count


def test(test_loader, disc_model, encoder):
    encoder.eval()
    disc_model.eval()
    loss_fn = torch.nn.BCELoss()
    epoch_loss = 0
    epoch_acc = 0
    batch_count = 0
    for x_t, x_p, x_n, _ in test_loader:
        mc_sample = x_p.shape[1]
        batch_size, f_size, len_size = x_t.shape
        x_p = x_p.reshape((-1, f_size, len_size))
        x_n = x_n.reshape((-1, f_size, len_size))
        x_t = np.repeat(x_t, mc_sample, axis=0)
        neighbors = torch.ones((len(x_p)))
        non_neighbors = torch.zeros((len(x_n)))
        z_t = encoder(x_t)
        z_p = encoder(x_p)
        z_n = encoder(x_n)
        p_loss = loss_fn(disc_model(z_t, z_p), neighbors)
        n_loss = loss_fn(disc_model(z_t, z_n), non_neighbors)
        loss = (p_loss + n_loss) / 2
        p_acc = torch.sum(disc_model(z_t, z_p) > 0.5).item() / len(z_p)
        n_acc = torch.sum(disc_model(z_t, z_n) < 0.5).item() / len(z_n)
        epoch_acc = epoch_acc + (p_acc + n_acc) / 2
        epoch_loss += loss.item()
        batch_count += 1
    return epoch_loss/batch_count, epoch_acc/batch_count


def plot_distribution():
    encoder.load_state_dict(torch.load('./ckpt/tcl_encoder.pt'))
    with open(os.path.join(path, 'x_test.pkl'), 'rb') as f:
        x_test = pickle.load(f)
    with open(os.path.join(path, 'state_test.pkl'), 'rb') as f:
        y_test = pickle.load(f)
    n_test = len(x_test)
    inds = np.random.randint(0, x_test.shape[-1] - 50, n_test * 4)
    windows = np.array([x_test[int(i % n_test), :, ind:ind + 50] for i, ind in enumerate(inds)])
    windows_state = [np.round(np.mean(y_test[i % n_test, ind:ind + window_size], axis=-1)) for i, ind in
                     enumerate(inds)]
    encodings = encoder(torch.Tensor(windows))

    embedding = TSNE(n_components=2).fit_transform(encodings.detach().cpu().numpy())
    # embedding = PCA(n_components=2).fit_transform(encodings.detach().cpu().numpy())
    # original_embedding = PCA(n_components=2).fit_transform(windows.reshape((len(windows), -1)))
    original_embedding = TSNE(n_components=2).fit_transform(windows.reshape((len(windows), -1)))

    df_original = pd.DataFrame({"f1": original_embedding[:, 0], "f2": original_embedding[:, 1], "state": windows_state})
    df_encoding = pd.DataFrame({"f1": embedding[:, 0], "f2": embedding[:, 1], "state": windows_state})
    # df_encoding = pd.DataFrame({"f1": embedding[:, 0], "f2": embedding[:, 1], "state": y_test[np.arange(4*n_test)%n_test, inds]})

    # Save plots
    plt.figure()
    plt.title("Origianl signals TSNE")
    sns.scatterplot(x="f1", y="f2", data=df_original, hue="state")
    plt.savefig(os.path.join("./plots", "signal_distribution.pdf"))

    plt.figure()
    plt.title("Signal Encoding TSNE")
    sns.scatterplot(x="f1", y="f2", data=df_encoding, hue="state")
    plt.savefig(os.path.join("./plots", "encoding_distribution.pdf"))



np.random.seed(1234)

window_size = 50



disc_model = Discriminator(10, 'cpu')
encoder = RnnEncoder(hidden_size=100, in_channel=3, encoding_size=10)

if not os.path.exists("./plots"):
    os.mkdir("./plots")
if not os.path.exists("./ckpt/"):
    os.mkdir("./ckpt/")


is_train=False

if is_train:
    path = './data/simulated_data/'
    with open(os.path.join(path, 'x_train.pkl'), 'rb') as f:
        x = pickle.load(f)
    with open(os.path.join(path, 'state_train.pkl'), 'rb') as f:
        y = pickle.load(f)
    n_train = int(0.8 * len(x))

    # f, axes = plt.subplots(3, 1)
    # f.set_figheight(3)
    # f.set_figwidth(10)
    # for i, ax in enumerate(axes):
    #     ax.plot(x[0, i, :])
    #     for t in range(x.shape[-1] - 1):
    #         ax.axvspan(t, t + 1, color=['red', 'green', 'blue', 'yellow'][y[0, t]], alpha=0.05)
    # plt.savefig(os.path.join("./plots", "example_TS.pdf"))

    trainset = TCLDataset(x=torch.Tensor(x[:n_train]), mc_sample_size=20, neighbouring_delta=10,
                          window_size=window_size)
    train_loader = data.DataLoader(trainset, batch_size=10, shuffle=True, sampler=None, batch_sampler=None,
                                   num_workers=0,
                                   collate_fn=None, pin_memory=False, drop_last=False)
    validset = TCLDataset(x=torch.Tensor(x[n_train:]), mc_sample_size=20, neighbouring_delta=10,
                          window_size=window_size)
    valid_loader = data.DataLoader(validset, batch_size=10, shuffle=True, sampler=None, batch_sampler=None,
                                   num_workers=0,
                                   collate_fn=None, pin_memory=False, drop_last=False)


    params = list(disc_model.parameters())+list(encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=0.01)
    performance = []
    n_epochs = 250
    for epoch in range(n_epochs):
        epoch_loss, epoch_acc = train(train_loader, disc_model, optimizer, encoder)
        test_loss, test_acc = test(valid_loader, disc_model, encoder)
        performance.append((epoch_loss, test_loss, epoch_acc, test_acc))
        print('Epoch %d Loss =====> Training Loss: %.5f \t Training Accuracy: %.5f \t Test Loss: %.5f \t Test Accuracy: %.5f'
              %(epoch, epoch_loss, epoch_acc, test_loss, test_acc))

    # Save checkpoints
    torch.save(encoder.state_dict(), './ckpt/tcl_encoder.pt')
    torch.save(disc_model.state_dict(), './ckpt/discriminator.pt')

    # Save performance plots
    train_loss = [t[0] for t in performance]
    test_loss = [t[1] for t in performance]
    train_acc = [t[2] for t in performance]
    test_acc = [t[3] for t in performance]
    plt.figure()
    plt.plot(np.arange(n_epochs), train_loss, label="Train")
    plt.plot(np.arange(n_epochs), test_loss, label="Test")
    plt.title("Loss")
    plt.legend()
    plt.savefig(os.path.join("./plots", "loss.pdf"))
    plt.figure()
    plt.plot(np.arange(n_epochs), train_acc, label="Train")
    plt.plot(np.arange(n_epochs), test_acc, label="Test")
    plt.title("Accuracy")
    plt.legend()
    plt.savefig(os.path.join("./plots", "accuracy.pdf"))
    plot_distribution()

else:
    # plot_distribution()
    exp = ClassificationPerformanceExperiment()
    exp.run()





