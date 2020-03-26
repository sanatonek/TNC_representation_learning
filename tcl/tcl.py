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
        t_p = np.random.randint(max(0, t - self.delta), min(t + self.delta, T - self.window_size), self.mc_sample_size)
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
        loss = Variable(torch.zeros(1, ), requires_grad=True)
        acc = 0
        for i in range(len(x_t)):
            neighbors = torch.ones((len(x_p[i])))
            non_neighbors = torch.zeros((len(x_n[i])))
            optimizer.zero_grad()
            z_t = encoder(x_t[i].unsqueeze(0))
            z_p = encoder(x_p[i])
            z_n = encoder(x_n[i])
            p_loss = loss_fn(disc_model(z_t.expand_as(z_p), z_p), neighbors)
            n_loss = loss_fn(disc_model(z_t.expand_as(z_n), z_n), non_neighbors)
            loss = loss + (p_loss + n_loss)/2
            p_acc = torch.sum(disc_model(z_t.expand_as(z_p), z_p)>0.5).item()/len(z_p)
            n_acc = torch.sum(disc_model(z_t.expand_as(z_n), z_n) < 0.5).item() / len(z_n)
            acc = acc + (p_acc+n_acc)/2
        loss.backward()
        optimizer.step()
        # TODO: possibly an iterative update
        epoch_loss = epoch_loss + loss.item()/len(x_t)
        epoch_acc += acc/len(x_t)
        batch_count += 1
        # print('Loss: ', loss.item())
    return epoch_loss/batch_count, epoch_acc/batch_count


def test(test_loader, disc_model, encoder):
    encoder.eval()
    disc_model.eval()
    loss_fn = torch.nn.BCELoss()
    epoch_loss = 0
    epoch_acc = 0
    batch_count = 0
    for x_t, x_p, x_n, _ in test_loader:
        loss = 0
        acc = 0
        for i in range(len(x_t)):
            neighbors = torch.ones((len(x_p[i])))
            non_neighbors = torch.zeros((len(x_n[i])))
            optimizer.zero_grad()
            z_t = encoder(x_t[i].unsqueeze(0))
            z_p = encoder(x_p[i])
            z_n = encoder(x_n[i])
            p_loss = loss_fn(disc_model(z_t.expand_as(z_p), z_p), neighbors)
            n_loss = loss_fn(disc_model(z_t.expand_as(z_n), z_n), non_neighbors)
            p_acc = (disc_model(z_t.expand_as(z_p), z_p)>0.5).sum().item()/len(z_p)
            n_acc = (disc_model(z_t.expand_as(z_n), z_n) < 0.5).sum().item() / len(z_n)
            acc = acc + (p_acc+n_acc)/2
            loss += (p_loss + n_loss)/2
        epoch_loss += loss/len(x_t)
        epoch_acc += acc/len(x_t)
        batch_count += 1
    return epoch_loss/batch_count, epoch_acc/batch_count


# x_t = torch.randn((1000, 2000))
# x_n = torch.randn((1000, 2000)) + 2
# x = torch.cat((x_t, x_n), -1)
path = './data/simulated_data/'

with open(os.path.join(path, 'x_train.pkl'), 'rb') as f:
    x = pickle.load(f)

n_train = int(0.8*len(x))

trainset = TCLDataset(x=torch.Tensor(x[:n_train]), mc_sample_size=20, neighbouring_delta=20, window_size=50)
train_loader = data.DataLoader(trainset, batch_size=50, shuffle=True, sampler=None, batch_sampler=None, num_workers=0,
                               collate_fn=None, pin_memory=False, drop_last=False)
validset = TCLDataset(x=torch.Tensor(x[n_train:]), mc_sample_size=20, neighbouring_delta=20, window_size=50)
valid_loader = data.DataLoader(validset, batch_size=50, shuffle=True, sampler=None, batch_sampler=None, num_workers=0,
                               collate_fn=None, pin_memory=False, drop_last=False)

disc_model = Discriminator(10, 'cpu')
encoder = RnnEncoder(hidden_size=100, in_channel=1, encoding_size=10)

is_train=False

if is_train:
    params = list(disc_model.parameters())+list(encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=0.01)
    n_epochs = 50
    for epoch in range(n_epochs):
        epoch_loss, epoch_acc = train(train_loader, disc_model, optimizer, encoder)
        test_loss, test_acc = test(valid_loader, disc_model, encoder)
        print('Epoch %d Loss =====> Training Loss: %.5f \t Training Accuracy: %.5f \t Test Loss: %.5f \t Training Accuracy: %.5f'
              %(epoch, epoch_loss, epoch_acc, test_loss, test_acc))

    if not os.path.exists("./ckpt/"):
        os.mkdir("./ckpt/")
    torch.save(encoder.state_dict(), './ckpt/encoder.pt')
    torch.save(disc_model.state_dict(), './ckpt/discriminator.pt')
else:
    encoder.load_state_dict(torch.load('./ckpt/encoder.pt'))
    with open(os.path.join(path, 'x_test.pkl'), 'rb') as f:
        x_test = pickle.load(f)
    with open(os.path.join(path, 'state_test.pkl'), 'rb') as f:
        y_test = pickle.load(f)
    n_test = len(x_test)
    inds = np.random.randint(0,x_test.shape[-1]-50, n_test)
    windows = np.array([x_test[i,:,ind:ind+50] for i,ind in enumerate(inds)])
    # windows_state = np.array([y_test[i,ind] for i,ind in enumerate(inds)])
    encodings = encoder(torch.Tensor(windows))

    # embedding = TSNE(n_components=2).fit_transform(encodings.detach().cpu().numpy())
    embedding = PCA(n_components=2).fit_transform(encodings.detach().cpu().numpy())
    # embedding = PCA(n_components=2).fit_transform(windows[:,0,:])
    # embedding = encoding.detach().cpu().numpy()[:,[2,4]]

    df = pd.DataFrame({"f1": embedding[:, 0], "f2": embedding[:, 1], "label": y_test[np.arange(n_test),inds]})

    # Separate plots
    # f = plt.plot()
    # f.set_figheight(10)
    # f.set_figwidth(20)
    # axes[0].set_title("aEncoding distribution")
    sns.scatterplot(x="f1", y="f2", data=df, hue="label")

    # Single plot
    # sns.scatterplot(x="f1", y="f2", hue="label", style="label", data=df)
    plt.show()
    # plt.savefig(os.path.join("./plots", "encoding_distribution.pdf"))



