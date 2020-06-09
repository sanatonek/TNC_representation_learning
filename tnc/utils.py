import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

from torch.utils import data
import torch
from sklearn.manifold import TSNE
from sklearn.mixture import BayesianGaussianMixture as DPGMM
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier


class PatientData():
    """Dataset of patient vitals, demographics and lab results
    Args:
        root: Root directory of the pickled dataset
        train_ratio: train/test ratio
        shuffle: Shuffle dataset before separating train/test
        transform: Preprocessing transformation on the dataset
    """
    def __init__(self, path='./data/mimic_data', train_ratio=0.8, shuffle=False, random_seed=1234):
        self.data_dir = os.path.join(path, 'patient_vital_preprocessed.pkl')
        self.train_ratio = train_ratio
        self.random_seed = np.random.seed(random_seed)

        if not os.path.exists(self.data_dir):
            raise RuntimeError('Dataset not found')
        with open(self.data_dir, 'rb') as f:
            self.data = pickle.load(f)
        with open(os.path.join(path,'patient_interventions.pkl'), 'rb') as f:
            self.intervention = pickle.load(f)
        if shuffle:
            inds = np.arange(len(self.data))
            np.random.shuffle(inds)
            self.data = self.data[inds]
            self.intervention = self.intervention[inds,:,:]
        self.feature_size = len(self.data[0][0])
        self.n_train = int(len(self.data) * self.train_ratio)
        self.n_test = len(self.data) - self.n_train
        self.train_data = np.array([x for (x, y, z) in self.data[0:self.n_train]])
        self.test_data = np.array([x for (x, y, z) in self.data[self.n_train:]])
        self.train_label = np.array([y for (x, y, z) in self.data[0:self.n_train]])
        self.test_label = np.array([y for (x, y, z) in self.data[self.n_train:]])
        self.train_missing = np.array([np.mean(z) for (x, y, z) in self.data[0:self.n_train]])
        self.test_missing = np.array([np.mean(z) for (x, y, z) in self.data[self.n_train:]])
        self.train_intervention = self.intervention[0:self.n_train,:,:]
        self.test_intervention = self.intervention[self.n_train:,:,:]
        self._normalize()

    def _normalize(self):
        """ Calculate the mean and std of each feature from the training set
        """
        feature_means = np.mean(self.train_data, axis=(0, 2))
        feature_std = np.std(self.train_data, axis=(0, 2))
        np.seterr(divide='ignore', invalid='ignore')
        train_data_n = self.train_data - feature_means[np.newaxis, :, np.newaxis] / \
                       np.where(feature_std == 0, 1, feature_std)[np.newaxis, :, np.newaxis]
        test_data_n = self.test_data - feature_means[np.newaxis, :, np.newaxis] / \
                      np.where(feature_std == 0, 1, feature_std)[np.newaxis, :, np.newaxis]
        self.train_data, self.test_data = train_data_n, test_data_n


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


def track_encoding(sample, label, encoder, window_size, path, sliding_gap=5):
    T = sample.shape[-1]
    windows_label = []
    encodings = []
    device = 'cuda'
    encoder.to(device)
    encoder.eval()
    for t in range(0,T-window_size,sliding_gap):
        windows = sample[:, t:t+window_size]
        windows_label.append((np.bincount(label[:t+window_size].astype(int)).argmax()))
        encodings.append(encoder(torch.Tensor(windows).unsqueeze(0).to(device)))
    for t in range(sliding_gap):
        encodings.append(encodings[-1])
    encodings = torch.stack(encodings, 0)


    f, axs = plt.subplots(2)#, gridspec_kw={'height_ratios': [1, 2]})
    f.set_figheight(10)
    f.set_figwidth(25)
    # axs[0].plot(sample[0])
    axs[0].set_title('Time series Sample Trajectory', fontsize=34)
    # axs[0].plot(sample[1])
    # axs[0].plot(sample[2])
    sns.lineplot(np.arange(sample.shape[1]), sample[0], ax=axs[0])
    sns.lineplot(np.arange(sample.shape[1]), sample[1], ax=axs[0])
    sns.lineplot(np.arange(sample.shape[1]), sample[2], ax=axs[0])
    # sns.lineplot(np.arange(sample.shape[1]-window_size), sample[0, :-window_size], ax=axs[0])
    # sns.lineplot(np.arange(sample.shape[1]-window_size), sample[1, :-window_size], ax=axs[0])
    # sns.lineplot(np.arange(sample.shape[1]-window_size), sample[2, :-window_size], ax=axs[0])
    axs[0].xaxis.set_tick_params(labelsize=22)
    axs[0].yaxis.set_tick_params(labelsize=22)
    axs[1].xaxis.set_tick_params(labelsize=22)
    axs[1].yaxis.set_tick_params(labelsize=22)
    axs[1].set_ylabel('Encoding dimensions', fontsize=30)
    axs[0].margins(x=0)
    for t in range(label.shape[-1]):#-window_size):
        axs[0].axvspan(t, min(t+1, label.shape[-1]-1), facecolor=['y', 'g', 'b', 'r'][label[t]], alpha=0.3)
    axs[1].set_title('Encoding Trajectory', fontsize=30)
    sns.heatmap(encodings.detach().cpu().numpy().T, cbar=False, linewidth=0.5, ax=axs[1], linewidths=0.05, xticklabels=False)
    f.tight_layout()


    # sns.heatmap(encodings.detach().cpu().numpy().T, linewidth=0.5)
    plt.savefig(os.path.join("./plots/%s" % path, "embedding_trajectory_hm.pdf"))

    # windows = np.split(sample[:, :window_size * (T // window_size)], (T // window_size), -1)
    # windows = torch.Tensor(np.stack(windows, 0)).to(encoder.device)
    # windows_label = np.split(label[:window_size * (T // window_size)], (T // window_size), -1)
    # windows_label = torch.Tensor(np.mean(np.stack(windows_label, 0), -1 ) ).to(encoder.device)
    # encoder.to(encoder.device)
    # encodings = encoder(windows)

    pca = PCA(n_components=2)
    embedding = pca.fit_transform(encodings.detach().cpu().numpy())
    d = {'f1':embedding[:,0], 'f2':embedding[:,1], 'time':np.arange(len(embedding))}#, 'label':windows_label}
    df = pd.DataFrame(data=d)
    fig, ax = plt.subplots()
    ax.set_title("Trajectory")
    # sns.jointplot(x="f1", y="f2", data=df, kind="kde", size='time', hue='label')
    sns.scatterplot(x="f1", y="f2", data=df, hue="time")
    plt.savefig(os.path.join("./plots/%s" % path, "embedding_trajectory.pdf"))


def plot_distribution(x_test, y_test, encoder, window_size, path, device, title="", augment=4, cv=0):
    checkpoint = torch.load('./ckpt/%s/checkpoint_%d.pth.tar'%(path, cv))
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    encoder = encoder.to(device)
    n_test = len(x_test)
    print(x_test.shape[-1] - window_size)
    inds = np.random.randint(0, x_test.shape[-1] - window_size, n_test * augment)
    windows = np.array([x_test[int(i % n_test), :, ind:ind + window_size] for i, ind in enumerate(inds)])
    windows_state = [np.round(np.mean(y_test[i % n_test, ind:ind + window_size], axis=-1)) for i, ind in
                     enumerate(inds)]
    encodings = encoder(torch.Tensor(windows).to(device))

    s_score = silhouette_score(encodings.detach().cpu().numpy(), np.array(windows_state))

    tsne = TSNE(n_components=2)
    embedding = tsne.fit_transform(encodings.detach().cpu().numpy())
    # pca = PCA(n_components=2)
    # embedding = pca.fit_transform(encodings.detach().cpu().numpy())
    # original_embedding = PCA(n_components=2).fit_transform(windows.reshape((len(windows), -1)))
    original_embedding = TSNE(n_components=2).fit_transform(windows.reshape((len(windows), -1)))


    df_original = pd.DataFrame({"f1": original_embedding[:, 0], "f2": original_embedding[:, 1], "state": windows_state})
    df_encoding = pd.DataFrame({"f1": embedding[:, 0], "f2": embedding[:, 1], "state": windows_state})
    # df_encoding = pd.DataFrame({"f1": embedding[:, 0], "f2": embedding[:, 1], "state": y_test[np.arange(4*n_test)%n_test, inds]})


    # Save plots
    # plt.figure()
    fig, ax = plt.subplots()
    ax.set_title("Origianl signals TSNE")
    # sns.jointplot(x="f1", y="f2", data=df_original, kind="kde", hue='state')
    sns.scatterplot(x="f1", y="f2", data=df_original, hue="state")
    plt.savefig(os.path.join("./plots/%s"%path, "signal_distribution.pdf"))

    fig, ax = plt.subplots()
    # plt.figure()
    ax.set_title("Encodings Distribution using %s"%title)
    sns.scatterplot(x="f1", y="f2", data=df_encoding, hue="state")
    # sns.jointplot(x="f1", y="f2", data=df_encoding, kind="kde", hue='state')

    from sklearn.mixture import GaussianMixture
    # dpgmm = DPGMM(10)
    # # dpgmm = GaussianMixture(4)
    # dpgmm.fit(embedding)
    # print('Number of components: ', dpgmm.n_components)
    # for i,n in enumerate(range(dpgmm.n_components)):
    #     mu = dpgmm.means_[i]
    #     sigma = dpgmm.covariances_[i]
    #     confidence_ellipse(mu, sigma, ax)
    #     ax.scatter(mu[0], mu[1], c='navy', s=3)
    plt.savefig(os.path.join("./plots/%s"%path, "encoding_distribution_%d.pdf"%cv))


def model_distribution(x_train, y_train, x_test, y_test, encoder, window_size, path, device):
    checkpoint = torch.load('./ckpt/%s/checkpoint.pth.tar'%path)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    encoder.eval()
    # n_train = len(x_train)
    n_test = len(x_test)
    augment = 100

    # inds = np.random.randint(0, x_train.shape[-1] - window_size, n_train * 20)

    # windows = np.array([x_train[int(i % n_train), :, ind:ind + window_size] for i, ind in enumerate(inds)])
    # windows_label = [np.round(np.mean(y_train[i % n_train, ind:ind + window_size], axis=-1))
    #                  for i, ind in enumerate(inds)]
    inds = np.random.randint(0, x_test.shape[-1] - window_size, n_test * augment)
    x_window_test = np.array([x_test[int(i % n_test), :, ind:ind + window_size] for i, ind in enumerate(inds)])
    y_window_test = np.array([np.round(np.mean(y_test[i % n_test, ind:ind + window_size], axis=-1)) for i, ind in
                     enumerate(inds)])
    # T = x_test.shape[-1]
    # x_window_test = np.split(x_test[:, :, :window_size * (T // window_size)], (T // window_size), -1)
    # y_window_test = np.split(y_test[:, :window_size * (T // window_size)], (T // window_size), -1)
    # x_window_test = torch.Tensor(np.concatenate(x_window_test, 0))
    # y_window_test = np.round(np.mean(np.concatenate(y_window_test, 0), -1))
    train_count = []
    if 'waveform' in path:
        encoder.to('cpu')
        x_window_test = torch.Tensor(x_window_test)
    else:
        encoder.to(device)
        x_window_test = torch.Tensor(x_window_test).to(device)

    # trainset = data.TensorDataset(torch.Tensor(windows), torch.Tensor(windows_label))
    # train_loader = data.DataLoader(trainset, batch_size=10, shuffle=True, sampler=None, batch_sampler=None,
    #                                num_workers=0, collate_fn=None, pin_memory=False, drop_last=False)
    # encodings = []
    # for x,_ in train_loader:
    #     x = x.to(device)
    #     encodings.append(encoder(x).detach().cpu().numpy())
    # encodings = np.concatenate(encodings, 0)

    encodings_test = encoder(x_window_test).detach().cpu().numpy()

    neigh = KNeighborsClassifier(n_neighbors=10)
    neigh.fit(encodings_test, y_window_test)
    # preds = neigh.predict(encodings_test)
    _, neigh_inds = neigh.kneighbors(encodings_test)
    neigh_ind_labels = [np.mean(y_window_test[ind]) for ind in (neigh_inds)]
    label_var = [(y_window_test[ind]==y_window_test[i]).sum() for i, ind in enumerate(neigh_inds)]
    dist = (label_var)/10
    # print(neigh_ind_labels[:10])
    # print(accuracy_score(y_window_test, preds))
    # dist = np.linalg.norm(neigh_ind_labels - y_window_test)**2/len(y_window_test)


def confidence_ellipse(mean, cov, ax, n_std=1.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.
    """
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, edgecolor='navy',
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mean[0]

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mean[1]

    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def trend_decompose(x, filter_size):
    df = pd.DataFrame(data=x.T)
    df = df.rolling(filter_size, win_type='triang').sum()
    s = df.loc[:, 0]
    # print(s[:10])
    # corr = []
    # for i in range(0, 30, 5):
    #     corr.append(np.abs(s.autocorr(i)))
    # print(corr)
    f, axs = plt.subplots(1)
    print(s[filter_size-1:].shape, x[0,:-filter_size+1].shape)
    axs.plot(s[filter_size-1:], c='red')
    axs.plot(x[0,:-filter_size+1], c='blue')
    plt.show()

