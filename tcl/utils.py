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


def track_encoding(sample, label, encoder, window_size, path):
    T = sample.shape[-1]

    windows_label = []
    encodings = []
    encoder.to(encoder.device)
    for t in range(0,100,10):
        windows = sample[:, :t+window_size]
        windows_label.append(int(np.mean(label[:t+window_size],-1)))
        encodings.append(encoder(torch.Tensor(windows).unsqueeze(0).to(encoder.device)))
    encodings = torch.stack(encodings, 0)

    # windows = np.split(sample[:, :window_size * (T // window_size)], (T // window_size), -1)
    # windows = torch.Tensor(np.stack(windows, 0)).to(encoder.device)
    # windows_label = np.split(label[:window_size * (T // window_size)], (T // window_size), -1)
    # windows_label = torch.Tensor(np.mean(np.stack(windows_label, 0), -1 ) ).to(encoder.device)
    # encoder.to(encoder.device)
    # encodings = encoder(windows)

    pca = PCA(n_components=2)
    embedding = pca.fit_transform(encodings.detach().cpu().numpy())
    d = {'f1':embedding[:,0], 'f2':embedding[:,1], 'time':np.arange(len(embedding)), 'label':windows_label}
    df = pd.DataFrame(data=d)
    # print(df)
    fig, ax = plt.subplots()
    ax.set_title("Trajectory")
    # sns.jointplot(x="f1", y="f2", data=df, kind="kde", size='time', hue='label')
    sns.scatterplot(x="f1", y="f2", data=df, hue="time", size='label')
    plt.savefig(os.path.join("./plots/%s" % path, "embedding_trajectory.pdf"))


def plot_distribution(x_test, y_test, encoder, window_size, path, device, augment=4):
    checkpoint = torch.load('./ckpt/%s/checkpoint.pth.tar'%path)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    encoder = encoder.to(device)
    n_test = len(x_test)
    print(x_test.shape[-1] - window_size)
    inds = np.random.randint(0, x_test.shape[-1] - window_size, n_test * augment)
    windows = np.array([x_test[int(i % n_test), :, ind:ind + window_size] for i, ind in enumerate(inds)])
    windows_state = [np.round(np.mean(y_test[i % n_test, ind:ind + window_size], axis=-1)) for i, ind in
                     enumerate(inds)]
    encodings = encoder(torch.Tensor(windows).to(device))

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
    ax.set_title("Signal Encoding TSNE 2")
    sns.scatterplot(x="f1", y="f2", data=df_encoding, hue="state")
    # sns.jointplot(x="f1", y="f2", data=df_encoding, kind="kde", hue='state')

    # dpgmm = DPGMM(20)
    # dpgmm.fit(embedding)
    # print('Number of components: ', dpgmm.n_components)
    # for i,n in enumerate(range(dpgmm.n_components)):
    #     mu = dpgmm.means_[i]
    #     sigma = dpgmm.covariances_[i]
    #     confidence_ellipse(mu, sigma, ax)
    #     ax.scatter(mu[0], mu[1], c='navy', s=3)
    plt.savefig(os.path.join("./plots/%s"%path, "encoding_distribution.pdf"))


def model_distribution(x_train, y_train, x_test, y_test, encoder, window_size, path, device):
    checkpoint = torch.load('./ckpt/%s/checkpoint.pth.tar'%path)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    encoder.eval()
    encoder.to(device)
    n_train = len(x_train)
    n_test = len(x_test)
    inds = np.random.randint(0, x_train.shape[-1] - window_size, n_train * 1000)

    windows = np.array([x_train[int(i % n_train), :, ind:ind + window_size] for i, ind in enumerate(inds)])
    windows_label = [np.round(np.mean(y_train[i % n_train, ind:ind + window_size], axis=-1))
                     for i, ind in enumerate(inds)]
    train_count = []
    for i in range(4):
        train_count.append(windows_label.count(i)/len(windows_label))
    print('Class distribution in train set: ', train_count)

    trainset = data.TensorDataset(torch.Tensor(windows), torch.Tensor(windows_label))
    train_loader = data.DataLoader(trainset, batch_size=10, shuffle=True, sampler=None, batch_sampler=None,
                                   num_workers=0, collate_fn=None, pin_memory=False, drop_last=False)
    encodings = []
    for x,_ in train_loader:
        x = x.to(device)
        encodings.append(encoder(x).detach().cpu().numpy())
    encodings = np.concatenate(encodings, 0)
    # encodings = encoder(torch.Tensor(windows).to(device))
    dpgmm = DPGMM(20)
    dpgmm.fit(encodings)
    log_lik_train = dpgmm.score(encodings)
    print('Training Log likelihood: ', log_lik_train)
    ind_1 = np.argwhere(y_train == 2.)
    ind_1 = [ind_ii for ii, ind_ii in enumerate(ind_1) if ii%2000==0]
    rare_samples = [x_train[k[0], :, k[1]:k[1]+window_size] for k in ind_1[0:min(10, len(ind_1))]]
    sample_1 = encoder(torch.Tensor(rare_samples).to(device))
    # sample_1 = encoder(torch.Tensor(x_train[ind_1[0], :, ind_1[1]:ind_1[1] + window_size]).unsqueeze(0).to(device))
    score_1 = dpgmm.score(sample_1.detach().cpu().numpy())
    print('Log likelihood of rare samples from training set: ', score_1)

    # ind_3 = np.argwhere(y_train == 3.)
    # ind_3 = [ind_ii for ii, ind_ii in enumerate(ind_3) if ii%20000==0]
    # common_samples = [x_train[k[0], :, k[1]:k[1] + window_size] for k in ind_3]
    # # sample_0 = encodings[windows_label.index(3)]
    # sample_0 = encoder(torch.Tensor(common_samples).to(device))
    # score_0 = dpgmm.score(sample_0.reshape((1, -1)))
    # print('Log likelihood of a common sample from the training set: ', score_0)


    test_windows = np.array([x_test[int(i % n_test), :, ind:ind + window_size] for i, ind in enumerate(inds)])
    test_windows_label = [np.round(np.mean(y_test[i % n_test, ind:ind + window_size], axis=-1))
                          for i, ind in enumerate(inds)]
    test_count = []
    for i in range(4):
        test_count.append(test_windows_label.count(i) / len(test_windows_label))
    print('Class distribution in test set: ', test_count)

    testset = data.TensorDataset(torch.Tensor(test_windows), torch.Tensor(test_windows_label))
    test_loader = data.DataLoader(testset, batch_size=10, shuffle=True, sampler=None, batch_sampler=None,
                                   num_workers=0, collate_fn=None, pin_memory=False, drop_last=False)
    test_encodings = []
    for x, _ in test_loader:
        x = x.to(device)
        test_encodings.append(encoder(x).detach().cpu().numpy())
    test_encodings = np.concatenate(test_encodings, 0)
    # test_encodings = encoder(torch.Tensor(test_windows).to(device))
    log_lik_test = dpgmm.score(test_encodings)
    print('Test Log likelihood: ', log_lik_test)
    ind_1 = np.argwhere(y_train == 2.)#[0]
    ind_1 = [ind_ii for ii, ind_ii in enumerate(ind_1) if ii%2000==0]
    rare_samples = [x_train[k[0], :, k[1]:k[1]+window_size] for k in ind_1]
    sample_1 = encoder(torch.Tensor(rare_samples).to(device))
    # ind_1 = np.argwhere(y_test == 1.)[0]
    # sample_1 = encoder(torch.Tensor(x_test[ind_1[0], :, ind_1[1]:ind_1[1] + window_size]).unsqueeze(0).to(device))
    score_1 = dpgmm.score(sample_1.detach().cpu().numpy())
    print('Log likelihood of a rare sample from th test set: ', score_1)
    # ind_3 = np.argwhere(y_train == 3.)
    # ind_3 = [ind_ii for ii, ind_ii in enumerate(ind_3) if ii%20000==0]
    # common_samples = [x_train[k[0], :, k[1]:k[1] + window_size] for k in ind_3]
    # sample_0 = encoder(torch.Tensor(common_samples).to(device))
    # # sample_0 = test_encodings[test_windows_label.index(3)]
    # score_0 = dpgmm.score(sample_0[np.newaxis,:])
    # print('Log likelihood of a common sample from the test set: ', score_0)


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