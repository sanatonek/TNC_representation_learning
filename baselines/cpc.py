import torch
import numpy as np
import random
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import argparse

from tnc.models import RnnEncoder, WFEncoder
from tnc.utils import plot_distribution, model_distribution
from tnc.evaluations import ClassificationPerformanceExperiment, WFClassificationExperiment

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def epoch_run(data, ds_estimator, auto_regressor, encoder, device, window_size, n_size=5, optimizer=None, train=True):
    if train:
        encoder.train()
        ds_estimator.train()
        auto_regressor.train()
    else:
        encoder.eval()
        ds_estimator.eval()
        auto_regressor.eval()
    encoder.to(device)
    ds_estimator.to(device)
    auto_regressor.to(device)

    epoch_loss = 0
    acc = 0
    for sample in data:
        rnd_t = np.random.randint(5*window_size,sample.shape[-1]-5*window_size)
        sample = torch.Tensor(sample[:,max(0,(rnd_t-20*window_size)):min(sample.shape[-1], rnd_t+20*window_size)])

        T = sample.shape[-1]
        windowed_sample = np.split(sample[:, :(T // window_size) * window_size], (T // window_size), -1)
        windowed_sample = torch.tensor(np.stack(windowed_sample, 0), device=device)
        encodings = encoder(windowed_sample)
        window_ind = torch.randint(2,len(encodings)-2, size=(1,))
        _, c_t = auto_regressor(encodings[max(0, window_ind[0]-10):window_ind[0]+1].unsqueeze(0))
        density_ratios = torch.bmm(encodings.unsqueeze(1),
                                       ds_estimator(c_t.squeeze(1).squeeze(0)).expand_as(encodings).unsqueeze(-1)).view(-1,)
        r = set(range(0, window_ind[0] - 2))
        r.update(set(range(window_ind[0] + 3, len(encodings))))
        rnd_n = np.random.choice(list(r), n_size)
        X_N = torch.cat([density_ratios[rnd_n], density_ratios[window_ind[0] + 1].unsqueeze(0)], 0)
        if torch.argmax(X_N)==len(X_N)-1:
            acc += 1
        labels = torch.Tensor([len(X_N)-1]).to(device)
        loss = torch.nn.CrossEntropyLoss()(X_N.view(1, -1), labels.long())
        epoch_loss += loss.item()

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return epoch_loss / len(data), acc/(len(data))


def learn_encoder(x, window_size, lr=0.001, decay=0, n_size=5, n_epochs=50, data='simulation', device='cpu', n_cross_val=1):
    if not os.path.exists("./plots/%s_cpc/"%data):
        os.mkdir("./plots/%s_cpc/"%data)
    if not os.path.exists("./ckpt/%s_cpc/"%data):
        os.mkdir("./ckpt/%s_cpc/"%data)
    accuracies = []
    for cv in range(n_cross_val):
        if 'waveform' in data:
            encoding_size = 64
            encoder = WFEncoder(encoding_size=64).to(device)
        elif 'simulation' in data:
            encoding_size = 10
            encoder = RnnEncoder(hidden_size=100, in_channel=3, encoding_size=10, device=device)
        elif 'har' in data:
            encoding_size = 10
            encoder = RnnEncoder(hidden_size=100, in_channel=561, encoding_size=10, device=device)
        ds_estimator = torch.nn.Linear(encoder.encoding_size, encoder.encoding_size)
        auto_regressor = torch.nn.GRU(input_size=encoding_size, hidden_size=encoding_size, batch_first=True)
        params = list(ds_estimator.parameters()) + list(encoder.parameters()) + list(auto_regressor.parameters())
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=decay)
        inds = list(range(len(x)))
        random.shuffle(inds)
        x = x[inds]
        n_train = int(0.8*len(x))
        best_acc = 0
        best_loss = np.inf
        train_loss, test_loss = [], []
        for epoch in range(n_epochs):
            epoch_loss, acc = epoch_run(x[:n_train], ds_estimator, auto_regressor, encoder, device, window_size, optimizer=optimizer,
                                        n_size=n_size, train=True)
            epoch_loss_test, acc_test = epoch_run(x[n_train:], ds_estimator, auto_regressor, encoder, device, window_size, n_size=n_size, train=False)
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
                best_acc = acc_test
                torch.save(state, './ckpt/%s_cpc/checkpoint_%d.pth.tar' %(data, cv))
        accuracies.append(best_acc)
        plt.figure()
        plt.plot(np.arange(n_epochs), train_loss, label="Train")
        plt.plot(np.arange(n_epochs), test_loss, label="Test")
        plt.title("CPC Loss")
        plt.legend()
        plt.savefig(os.path.join("./plots/%s_cpc/loss_%d.pdf"%(data, cv)))
    print('=======> Performance Summary:')
    print('Accuracy: %.2f +- %.2f' % (100 * np.mean(accuracies), 100 * np.std(accuracies)))


def main(is_train, data_type, lr,  cv):
    if not os.path.exists("./plots"):
        os.mkdir("./plots")
    if not os.path.exists("./ckpt/"):
        os.mkdir("./ckpt/")

    if data_type =='waveform':
        path = './data/waveform_data/processed'
        encoding_size = 64
        window_size = 2500
        encoder = WFEncoder(encoding_size=encoding_size).to(device)
        if is_train:
            with open(os.path.join(path, 'x_train.pkl'), 'rb') as f:
                x = pickle.load(f)
            T = x.shape[-1]
            x_window = np.concatenate(np.split(x[:, :, :T // 5 * 5], 5, -1), 0)
            learn_encoder(x_window, window_size, n_epochs=100, lr=lr, decay=1e-5,  n_size=10,
                          device=device, data=data_type, n_cross_val=cv)

        else:
            with open(os.path.join(path, 'x_test.pkl'), 'rb') as f:
                x_test = pickle.load(f)
            with open(os.path.join(path, 'state_test.pkl'), 'rb') as f:
                y_test = pickle.load(f)
            for cv_ind in range(cv):
                plot_distribution(x_test, y_test, encoder, window_size=window_size, path='%s_cpc' % data_type,
                                  device=device, augment=100, cv=cv_ind, title='CPC')
            exp = WFClassificationExperiment(window_size=window_size)
            exp.run(data='%s_cpc'%data_type, n_epochs=15, lr_e2e=0.001, lr_cls=0.001)

    elif data_type == 'simulation':
        path = './data/simulated_data/'
        window_size = 50
        encoder = RnnEncoder(hidden_size=100, in_channel=3, encoding_size=10, device=device)
        if is_train:
            with open(os.path.join(path, 'x_train.pkl'), 'rb') as f:
                x = pickle.load(f)
            learn_encoder(x, window_size, n_epochs=200, lr=lr, decay=1e-4, n_size=15,
                          data=data_type, device=device, n_cross_val=cv)

        else:
            with open(os.path.join(path, 'x_test.pkl'), 'rb') as f:
                x_test = pickle.load(f)
            with open(os.path.join(path, 'state_test.pkl'), 'rb') as f:
                y_test = pickle.load(f)
            for cv_ind in range(cv):
                plot_distribution(x_test, y_test, encoder, window_size=window_size, path='%s_cpc' % data_type,
                                  title='CPC', device=device, cv=cv_ind)
                exp = ClassificationPerformanceExperiment(path='simulation_cpc', cv=cv_ind)
                # Run cross validation for classification
                for lr in [0.001, 0.01, 0.1]:
                    print('===> lr: ', lr)
                    tnc_acc, tnc_auc, e2e_acc, e2e_auc = exp.run(data='%s_cpc'%data_type, n_epochs=50, lr_e2e=lr, lr_cls=lr)
                    print('TNC acc: %.2f \t TNC auc: %.2f \t E2E acc: %.2f \t E2E auc: %.2f' % (
                    tnc_acc, tnc_auc, e2e_acc, e2e_auc))

    elif data_type == 'har':
        window_size = 5
        path = './data/HAR_data/'
        encoder = RnnEncoder(hidden_size=100, in_channel=561, encoding_size=10, device=device)

        if is_train:
            with open(os.path.join(path, 'x_train.pkl'), 'rb') as f:
                x = pickle.load(f)
            learn_encoder(x, window_size, n_epochs=300, lr=lr, decay=1e-4, n_size=15,
                          data=data_type, device=device, n_cross_val=cv)
        else:
            with open(os.path.join(path, 'x_test.pkl'), 'rb') as f:
                x_test = pickle.load(f)
            with open(os.path.join(path, 'state_test.pkl'), 'rb') as f:
                y_test = pickle.load(f)

            for cv_ind in range(cv):
                plot_distribution(x_test, y_test, encoder, window_size=window_size, path='har_cpc',
                                  device=device, augment=100, cv=cv_ind, title='CPC')
                exp = ClassificationPerformanceExperiment(n_states=6, encoding_size=10, path='har_cpc', hidden_size=100,
                                                        in_channel=561, window_size=5, cv=cv_ind)
                # Run cross validation for classification
                for lr in [0.001, 0.01, 0.1]:
                    print('===> lr: ', lr)
                    tnc_acc, tnc_auc, e2e_acc, e2e_auc = exp.run(data='har', n_epochs=50, lr_e2e=lr, lr_cls=lr)
                    print('TNC acc: %.2f \t TNC auc: %.2f \t E2E acc: %.2f \t E2E auc: %.2f'%(tnc_acc, tnc_auc, e2e_acc, e2e_auc))


if __name__=="__main__":
    random.seed(1234)
    parser = argparse.ArgumentParser(description='Run CPC')
    parser.add_argument('--data', type=str, default='simulation')
    parser.add_argument('--cv', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()
    main(args.train, args.data, args.lr, args.cv)

