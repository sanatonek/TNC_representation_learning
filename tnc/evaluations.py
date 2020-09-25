import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import pandas as pd
import random

from tnc.models import RnnEncoder, StateClassifier, E2EStateClassifier, WFEncoder
from tnc.utils import create_simulated_dataset

from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


class ClassificationPerformanceExperiment():
    def __init__(self, n_states=4, encoding_size=10, path='simulation', cv=0, hidden_size=100, in_channel=3, window_size=50):
        # Load or train a TNC encoder
        if not os.path.exists("./ckpt/%s/checkpoint_%d.pth.tar"%(path,cv)):
            raise ValueError("No checkpoint for an encoder")
        checkpoint = torch.load('./ckpt/%s/checkpoint_%d.pth.tar'%(path, cv))
        self.encoder = RnnEncoder(hidden_size=hidden_size, in_channel=in_channel, encoding_size=encoding_size)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.classifier = StateClassifier(input_size=encoding_size, output_size=n_states)

        self.n_states = n_states
        self.cv = cv
        # Build a new encoder to train end-to-end with a classifier
        self.e2e_model = E2EStateClassifier(hidden_size=hidden_size, in_channel=in_channel, encoding_size=encoding_size, output_size=n_states)

        data_path = './data/HAR_data/' if 'har' in path else './data/simulated_data/'
        self.train_loader, self.valid_loader, self.test_loader = create_simulated_dataset\
            (window_size=window_size, path=data_path, batch_size=100)

    def _train_end_to_end(self, lr):
        self.e2e_model.train()
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.e2e_model.parameters(), lr=lr)

        epoch_loss, epoch_auc = 0, 0
        epoch_acc = 0
        batch_count = 0
        y_all, prediction_all = [], []
        for i, (x, y) in enumerate(self.train_loader):
            if i>5:
                break
            optimizer.zero_grad()
            prediction = self.e2e_model(x)
            state_prediction = torch.argmax(prediction, dim=1)
            loss = loss_fn(prediction, y.long())
            loss.backward()
            optimizer.step()
            y_all.append(y)
            prediction_all.append(prediction.detach().cpu().numpy())
            epoch_acc += torch.eq(state_prediction, y).sum().item()/len(x)
            epoch_loss += loss.item()
            batch_count += 1
        y_all = np.concatenate(y_all, 0)
        prediction_all = np.concatenate(prediction_all, 0)
        prediction_class_all = np.argmax(prediction_all, -1)
        y_onehot_all = np.zeros(prediction_all.shape)
        y_onehot_all[np.arange(len(y_onehot_all)), y_all.astype(int)] = 1
        epoch_auc = roc_auc_score(y_onehot_all, prediction_all)
        c = confusion_matrix(y_all.astype(int), prediction_class_all)
        return epoch_loss / batch_count, epoch_acc / batch_count, epoch_auc, c

    def _train_tnc_classifier(self, lr):
        self.classifier.train()
        self.encoder.eval()
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=lr)

        epoch_loss, epoch_auc = 0, 0
        epoch_acc = 0
        batch_count = 0
        y_all, prediction_all = [], []
        for i, (x, y) in enumerate(self.train_loader):
            if i > 5:
                break
            optimizer.zero_grad()
            encodings = self.encoder(x)
            prediction = self.classifier(encodings)
            state_prediction = torch.argmax(prediction, dim=1)
            loss = loss_fn(prediction, y.long())
            loss.backward()
            optimizer.step()
            y_all.append(y)
            prediction_all.append(prediction.detach().cpu().numpy())

            epoch_acc += torch.eq(state_prediction, y).sum().item()/len(x)
            epoch_loss += loss.item()
            batch_count += 1
        y_all = np.concatenate(y_all, 0)
        prediction_all = np.concatenate(prediction_all, 0)
        prediction_class_all = np.argmax(prediction_all, -1)
        y_onehot_all = np.zeros(prediction_all.shape)
        y_onehot_all[np.arange(len(y_onehot_all)), y_all.astype(int)] = 1
        epoch_auc = roc_auc_score(y_onehot_all, prediction_all)
        c = confusion_matrix(y_all.astype(int), prediction_class_all)
        return epoch_loss / batch_count, epoch_acc / batch_count, epoch_auc, c

    def _test(self, model):
        model.eval()
        loss_fn = torch.nn.CrossEntropyLoss()
        data_loader = self.valid_loader

        epoch_loss, epoch_auc = 0, 0
        epoch_acc = 0
        batch_count = 0
        y_all, prediction_all = [], []
        for x, y in data_loader:
            prediction = model(x)
            state_prediction = torch.argmax(prediction, -1)
            loss = loss_fn(prediction, y.long())
            y_all.append(y)
            prediction_all.append(prediction.detach().cpu().numpy())

            epoch_acc += torch.eq(state_prediction, y).sum().item()/len(x)
            epoch_loss += loss.item()
            batch_count += 1
        y_all = np.concatenate(y_all, 0)
        prediction_all = np.concatenate(prediction_all, 0)
        y_onehot_all = np.zeros(prediction_all.shape)
        prediction_class_all = np.argmax(prediction_all, -1)
        y_onehot_all[np.arange(len(y_onehot_all)), y_all.astype(int)] = 1
        epoch_auc = roc_auc_score(y_onehot_all, prediction_all)
        c = confusion_matrix(y_all.astype(int), prediction_class_all)
        return epoch_loss / batch_count, epoch_acc / batch_count, epoch_auc, c

    def run(self, data, n_epochs, lr_e2e, lr_cls=0.01):
        tnc_acc, tnc_loss, tnc_auc = [], [], []
        etoe_acc, etoe_loss, etoe_auc = [], [], []
        tnc_acc_test, tnc_loss_test, tnc_auc_test = [], [], []
        etoe_acc_test, etoe_loss_test, etoe_auc_test = [], [], []
        for epoch in range(n_epochs):
            loss, acc, auc, _ = self._train_tnc_classifier(lr_cls)
            tnc_acc.append(acc)
            tnc_loss.append(loss)
            tnc_auc.append(auc)
            loss, acc, auc, _ = self._train_end_to_end(lr_e2e)
            etoe_acc.append(acc)
            etoe_loss.append(loss)
            etoe_auc.append(auc)
            # Test
            loss, acc, auc, c_mtx_enc = self._test(model=torch.nn.Sequential(self.encoder, self.classifier))
            tnc_acc_test.append(acc)
            tnc_loss_test.append(loss)
            tnc_auc_test.append(auc)
            loss, acc, auc, c_mtx_e2e = self._test(model=self.e2e_model) #torch.nn.Sequential(self.encoder, self.classifier))
            etoe_acc_test.append(acc)
            etoe_loss_test.append(loss)
            etoe_auc_test.append(auc)

            if epoch%50 ==49:
                print('***** Epoch %d *****'%epoch)
                print('TNC =====> Training Loss: %.3f \t Training Acc: %.3f \t Training AUC: %.3f '
                      '\t Test Loss: %.3f \t Test Acc: %.3f \t Test AUC: %.3f'
                      % (tnc_loss[-1], tnc_acc[-1], tnc_auc[-1], tnc_loss_test[-1], tnc_acc_test[-1], tnc_auc_test[-1]))
                print('End-to-End =====> Training Loss: %.3f \t Training Acc: %.3f \t Training AUC: %.3f'
                      ' \t Test Loss: %.3f \t Test Acc: %.3f \t Test AUC: %.3f'
                      % (etoe_loss[-1], etoe_acc[-1], etoe_auc[-1], etoe_loss_test[-1], etoe_acc_test[-1], etoe_auc_test[-1]))

        # Save performance plots
        plt.figure()
        plt.plot(np.arange(n_epochs), tnc_loss_test, label="tnc test")
        plt.plot(np.arange(n_epochs), etoe_loss_test, label="e2e test")
        plt.title("Loss trend for the e2e and tnc framework")
        plt.legend()
        plt.savefig(os.path.join("./plots/%s"%data, "classification_loss_comparison.pdf"))

        plt.figure()
        plt.plot(np.arange(n_epochs), tnc_acc, label="tnc train")
        plt.plot(np.arange(n_epochs), etoe_acc, label="e2e train")
        plt.plot(np.arange(n_epochs), tnc_acc_test, label="tnc test")
        plt.plot(np.arange(n_epochs), etoe_acc_test, label="e2e test")
        plt.title("Accuracy trend for the e2e and tnc model")
        plt.legend()
        plt.savefig(os.path.join("./plots/%s"%data, "classification_accuracy_comparison_%d.pdf"%self.cv))

        plt.figure()
        plt.plot(np.arange(n_epochs), tnc_auc, label="tnc train")
        plt.plot(np.arange(n_epochs), etoe_auc, label="e2e train")
        plt.plot(np.arange(n_epochs), tnc_auc_test, label="tnc test")
        plt.plot(np.arange(n_epochs), etoe_auc_test, label="e2e test")
        plt.title("AUC trend for the e2e and TNC model")
        plt.legend()
        plt.savefig(os.path.join("./plots/%s" % data, "classification_auc_comparison_%d.pdf"%self.cv))

        df_cm = pd.DataFrame(c_mtx_enc, index=[i for i in ['']*self.n_states],
                             columns=[i for i in ['']*self.n_states])
        plt.figure(figsize=(10, 10))
        sns.heatmap(df_cm, annot=True)
        plt.savefig(os.path.join("./plots/%s"%data, "encoder_cf_matrix.pdf"))
        return tnc_acc_test[-1], tnc_auc_test[-1], etoe_acc_test[-1], etoe_auc_test[-1]


class WFClassificationExperiment(ClassificationPerformanceExperiment):
    def __init__(self, n_classes=4, encoding_size=64, window_size=2500, data='waveform', cv=0):
        # Load or train a TNC encoder and an end to end model
        if not os.path.exists("./ckpt/%s/checkpoint_%d.pth.tar"%(data, cv)):
            raise ValueError("No checkpoint for an encoder")
        checkpoint = torch.load('./ckpt/%s/checkpoint_%d.pth.tar'%(data, cv))
        # print('Loading encoder with discrimination performance accuracy of %.3f '%checkpoint['best_accuracy'])
        self.encoder = WFEncoder(encoding_size=encoding_size)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.classifier = WFEncoder(encoding_size=encoding_size, classify=True, n_classes=n_classes).classifier
        self.e2e_model = WFEncoder(encoding_size=encoding_size, classify=True, n_classes=n_classes)

        # Load data
        wf_datapath = './data/waveform_data/processed'
        with open(os.path.join(wf_datapath, 'x_train.pkl'), 'rb') as f:
            x = pickle.load(f)
        with open(os.path.join(wf_datapath, 'state_train.pkl'), 'rb') as f:
            y = pickle.load(f)
        T = x.shape[-1]
        x_window = np.split(x[:, :, :window_size * (T // window_size)],(T//window_size), -1)

        y_window = np.concatenate(np.split(y[:, :window_size * (T // window_size)], (T // window_size), -1), 0).astype(int)
        y_window = torch.Tensor(np.array([np.bincount(yy).argmax() for yy in y_window]))
        shuffled_inds = list(range(len(y_window)))
        random.shuffle(shuffled_inds)
        x_window = torch.Tensor(np.concatenate(x_window, 0))
        x_window = x_window[shuffled_inds]
        y_window = y_window[shuffled_inds]
        n_train = int(0.7*len(x_window))
        trainset = torch.utils.data.TensorDataset(x_window[:n_train], y_window[:n_train])
        validset = torch.utils.data.TensorDataset(x_window[n_train:], y_window[n_train:])

        self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True)
        self.valid_loader = torch.utils.data.DataLoader(validset, batch_size=100, shuffle=True)

