"""
Implementation of the Triplet Loss baseline based on the original code available on
https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries
"""

import torch
import numpy as np
import argparse
import os
import random
import pickle
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from tnc.models import RnnEncoder, WFEncoder
from tnc.utils import plot_distribution, model_distribution
from tnc.evaluations import ClassificationPerformanceExperiment, WFClassificationExperiment

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class TripletLoss(torch.nn.modules.loss._Loss):
    """
    Triplet loss for representations of time series. Optimized for training
    sets where all time series have the same length.
    Takes as input a tensor as the chosen batch to compute the loss,
    a PyTorch module as the encoder, a 3D tensor (`B`, `C`, `L`) containing
    the training set, where `B` is the batch size, `C` is the number of
    channels and `L` is the length of the time series, as well as a boolean
    which, if True, enables to save GPU memory by propagating gradients after
    each loss term, instead of doing it after computing the whole loss.
    The triplets are chosen in the following manner. First the size of the
    positive and negative samples are randomly chosen in the range of lengths
    of time series in the dataset. The size of the anchor time series is
    randomly chosen with the same length upper bound but the the length of the
    positive samples as lower bound. An anchor of this length is then chosen
    randomly in the given time series of the train set, and positive samples
    are randomly chosen among subseries of the anchor. Finally, negative
    samples of the chosen length are randomly chosen in random time series of
    the train set.
    @param compared_length Maximum length of randomly chosen time series. If
           None, this parameter is ignored.
    @param nb_random_samples Number of negative samples per batch example.
    @param negative_penalty Multiplicative coefficient for the negative sample
           loss.
    """
    def __init__(self, compared_length, nb_random_samples, negative_penalty):
        super(TripletLoss, self).__init__()
        self.compared_length = compared_length
        if self.compared_length is None:
            self.compared_length = np.inf
        self.nb_random_samples = nb_random_samples
        self.negative_penalty = negative_penalty

    def forward(self, batch, encoder, train, save_memory=False):
        batch=batch.to(device)
        train=train.to(device)
        encoder = encoder.to(device)
        batch_size = batch.size(0)
        train_size = train.size(0)
        length = min(self.compared_length, train.size(2))

        # For each batch element, we pick nb_random_samples possible random
        # time series in the training set (choice of batches from where the
        # negative examples will be sampled)
        samples = np.random.choice(
            train_size, size=(self.nb_random_samples, batch_size)
        )
        samples = torch.LongTensor(samples)

        # Choice of length of positive and negative samples
        length_pos_neg = self.compared_length
        # length_pos_neg = np.random.randint(1, high=length + 1)


        # We choose for each batch example a random interval in the time
        # series, which is the 'anchor'
        random_length = self.compared_length

        beginning_batches = np.random.randint(
            0, high=length - random_length + 1, size=batch_size
        )  # Start of anchors

        # The positive samples are chosen at random in the chosen anchors
        beginning_samples_pos = np.random.randint(
            0, high=random_length + 1, size=batch_size
        )  # Start of positive samples in the anchors
        # Start of positive samples in the batch examples
        beginning_positive = beginning_batches + beginning_samples_pos
        # End of positive samples in the batch examples
        end_positive = beginning_positive + length_pos_neg + np.random.randint(0,self.compared_length)

        # We randomly choose nb_random_samples potential negative samples for
        # each batch example
        beginning_samples_neg = np.random.randint(
            0, high=length - length_pos_neg + 1,
            size=(self.nb_random_samples, batch_size)
        )


        representation = encoder(torch.cat(
            [batch[
                j: j + 1, :,
                beginning_batches[j]: beginning_batches[j] + random_length
            ] for j in range(batch_size)]).to(device))  # Anchors representations

        positive_representation = encoder(torch.cat(
            [batch[
                j: j + 1, :, end_positive[j] - length_pos_neg: end_positive[j]
            ] for j in range(batch_size)]
        ))  # Positive samples representations

        size_representation = representation.size(1)
        # Positive loss: -logsigmoid of dot product between anchor and positive
        # representations
        loss = -torch.mean(torch.nn.functional.logsigmoid(torch.bmm(
            representation.view(batch_size, 1, size_representation),
            positive_representation.view(batch_size, size_representation, 1)
        )))

        # If required, backward through the first computed term of the loss and
        # free from the graph everything related to the positive sample
        if save_memory:
            loss.backward(retain_graph=True)
            loss = 0
            del positive_representation
            torch.cuda.empty_cache()

        multiplicative_ratio = self.negative_penalty / self.nb_random_samples
        for i in range(self.nb_random_samples):
            # Negative loss: -logsigmoid of minus the dot product between
            # anchor and negative representations
            negative_representation = encoder(
                torch.cat([train[samples[i, j]: samples[i, j] + 1][
                    :, :,
                    beginning_samples_neg[i, j]:
                    beginning_samples_neg[i, j] + length_pos_neg
                ] for j in range(batch_size)])
            )
            loss += multiplicative_ratio * -torch.mean(
                torch.nn.functional.logsigmoid(-torch.bmm(
                    representation.view(batch_size, 1, size_representation),
                    negative_representation.view(
                        batch_size, size_representation, 1
                    )
                ))
            )
            # If required, backward through the first computed term of the loss
            # and free from the graph everything related to the negative sample
            # Leaves the last backward pass to the training procedure
            if save_memory and i != self.nb_random_samples - 1:
                loss.backward(retain_graph=True)
                loss = 0
                del negative_representation
                torch.cuda.empty_cache()

        return loss


def epoch_run(data, encoder, device, window_size, optimizer=None, train=True):
    if train:
        encoder.train()
    else:
        encoder.eval()
    encoder = encoder.to(device)
    loss_criterion = TripletLoss(compared_length=window_size, nb_random_samples=10, negative_penalty=1)

    epoch_loss = 0
    acc = 0
    dataset = torch.utils.data.TensorDataset(torch.Tensor(data).to(device), torch.zeros((len(data),1)).to(device))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=20, shuffle=True)
    i = 0
    for x_batch,y in data_loader:
        loss = loss_criterion(x_batch.to(device), encoder, torch.Tensor(data).to(device))
        epoch_loss += loss.item()
        i += 1
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return epoch_loss/i, acc/i


def learn_encoder(x, window_size, data, lr=0.001, decay=0, n_epochs=100, device='cpu', n_cross_val=1):
    if not os.path.exists("./plots/%s_trip/"%data):
        os.mkdir("./plots/%s_trip/"%data)
    if not os.path.exists("./ckpt/%s_trip/"%data):
        os.mkdir("./ckpt/%s_trip/"%data)
    for cv in range(n_cross_val):
        if 'waveform' in data:
            encoder = WFEncoder(encoding_size=64).to(device)
        elif 'simulation' in data:
            encoder = RnnEncoder(hidden_size=100, in_channel=3, encoding_size=10, device=device).to(device)
        elif 'har' in data:
            encoder = RnnEncoder(hidden_size=100, in_channel=561, encoding_size=10, device=device).to(device)
        params = encoder.parameters()
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=decay)
        inds = list(range(len(x)))
        random.shuffle(inds)
        x = x[inds]
        n_train = int(0.8*len(x))
        train_loss, test_loss = [], []
        best_loss = np.inf
        for epoch in range(n_epochs):
            epoch_loss, acc = epoch_run(x[:n_train], encoder, device, window_size, optimizer=optimizer, train=True)
            epoch_loss_test, acc_test = epoch_run(x[n_train:], encoder, device, window_size, optimizer=optimizer, train=False)
            print('\nEpoch ', epoch)
            print('Train ===> Loss: ', epoch_loss)
            print('Test ===> Loss: ', epoch_loss_test)
            train_loss.append(epoch_loss)
            test_loss.append(epoch_loss_test)
            if epoch_loss_test<best_loss:
                print('Save new ckpt')
                state = {
                    'epoch': epoch,
                    'encoder_state_dict': encoder.state_dict()
                }
                best_loss = epoch_loss_test
                torch.save(state, './ckpt/%s_trip/checkpoint_%d.pth.tar' %(data, cv))
        plt.figure()
        plt.plot(np.arange(n_epochs), train_loss, label="Train")
        plt.plot(np.arange(n_epochs), test_loss, label="Test")
        plt.title("Loss")
        plt.legend()
        plt.savefig(os.path.join("./plots/%s_trip/loss_%d.pdf"%(data,cv)))


def main(is_train, data, cv):
    if not os.path.exists("./plots"):
        os.mkdir("./plots")
    if not os.path.exists("./ckpt/"):
        os.mkdir("./ckpt/")

    if data =='waveform':
        path = './data/waveform_data/processed'
        window_size = 2500
        encoder = WFEncoder(encoding_size=64).to(device)
        if is_train:
            with open(os.path.join(path, 'x_train.pkl'), 'rb') as f:
                x = pickle.load(f)
            T = x.shape[-1]
            x_window = np.concatenate(np.split(x[:, :, :T // 5 * 5], 5, -1), 0)
            learn_encoder(x_window, window_size, n_epochs=150, lr=1e-4, decay=1e-4, data='waveform', n_cross_val=cv)
        else:
            with open(os.path.join(path, 'x_test.pkl'), 'rb') as f:
                x_test = pickle.load(f)
            with open(os.path.join(path, 'state_test.pkl'), 'rb') as f:
                y_test = pickle.load(f)
            for cv_ind in range(cv):
                plot_distribution(x_test, y_test, encoder, window_size=window_size, path='%s_trip' % data,
                                  device=device, augment=100, cv=cv_ind, title='Triplet Loss')
            exp = WFClassificationExperiment(window_size=window_size, data='waveform_trip')
            exp.run(data='waveform_trip', n_epochs=15, lr_e2e=0.001, lr_cls=0.001)

    elif data == 'simulation':
        path = './data/simulated_data/'
        window_size = 50
        encoder = RnnEncoder(hidden_size=100, in_channel=3, encoding_size=10, device=device).to(device)
        if is_train:
            with open(os.path.join(path, 'x_train.pkl'), 'rb') as f:
                x = pickle.load(f)
            learn_encoder(x, window_size, lr=1e-3, decay=1e-5, data=data, n_epochs=150, device=device, n_cross_val=cv)
        else:
            with open(os.path.join(path, 'x_test.pkl'), 'rb') as f:
                x_test = pickle.load(f)
            with open(os.path.join(path, 'state_test.pkl'), 'rb') as f:
                y_test = pickle.load(f)
            for cv_ind in range(cv):
                plot_distribution(x_test, y_test, encoder, window_size=window_size, path='%s_trip' % data,
                                  title='Triplet Loss', device=device, cv=cv_ind)
                exp = ClassificationPerformanceExperiment(path='simulation_trip', cv=cv_ind)
                # Run cross validation for classification
                for lr in [0.001, 0.01, 0.1]:
                    print('===> lr: ', lr)
                    tnc_acc, tnc_auc, e2e_acc, e2e_auc = exp.run(data='simulation_trip', n_epochs=50, lr_e2e=lr, lr_cls=lr)
                    print('TNC acc: %.2f \t TNC auc: %.2f \t E2E acc: %.2f \t E2E auc: %.2f' % (
                        tnc_acc, tnc_auc, e2e_acc, e2e_auc))

    elif data == 'har':
        window_size = 5
        path = './data/HAR_data/'
        encoder = RnnEncoder(hidden_size=100, in_channel=561, encoding_size=10, device=device)

        if is_train:
            with open(os.path.join(path, 'x_train.pkl'), 'rb') as f:
                x = pickle.load(f)
            learn_encoder(x, window_size, lr=1e-5, decay=0.001, data=data, n_epochs=300, device=device, n_cross_val=cv)
        else:
            with open(os.path.join(path, 'x_test.pkl'), 'rb') as f:
                x_test = pickle.load(f)
            with open(os.path.join(path, 'state_test.pkl'), 'rb') as f:
                y_test = pickle.load(f)
            for cv_ind in range(cv):
                plot_distribution(x_test, y_test, encoder, window_size=window_size, path='har_trip',
                                  device=device, augment=100, cv=cv_ind, title='Triplet Loss')
                exp = ClassificationPerformanceExperiment(n_states=6, encoding_size=10, path='har_trip', hidden_size=100,
                                                      in_channel=561, window_size=5, cv=cv_ind)
                # Run cross validation for classification
                for lr in [0.001, 0.01, 0.1]:
                    print('===> lr: ', lr)
                    tnc_acc, tnc_auc, e2e_acc, e2e_auc = exp.run(data='har_trip', n_epochs=100, lr_e2e=lr, lr_cls=lr)
                    print('TNC acc: %.2f \t TNC auc: %.2f \t E2E acc: %.2f \t E2E auc: %.2f' % (
                    tnc_acc, tnc_auc, e2e_acc, e2e_auc))


if __name__=="__main__":
    random.seed(1234)
    parser = argparse.ArgumentParser(description='Run Triplet Loss')
    parser.add_argument('--data', type=str, default='simulation')
    parser.add_argument('--cv', type=int, default=1)
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()
    main(args.train, args.data, args.cv)