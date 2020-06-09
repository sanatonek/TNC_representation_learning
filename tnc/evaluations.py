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
    def __init__(self, n_states=4, encoding_size=10, path='simulation', cv=0):
        # Load or train a TCL encoder
        if not os.path.exists("./ckpt/%s/checkpoint_%d.pth.tar"%(path,cv)):
            raise ValueError("No checkpoint for an encoder")
        checkpoint = torch.load('./ckpt/%s/checkpoint_%d.pth.tar'%(path, cv))
        # print('Loading encoder with discrimination performance accuracy of %.3f '%checkpoint['best_accuracy'])
        self.tcl_encoder = RnnEncoder(hidden_size=100, in_channel=3, encoding_size=encoding_size)
        self.tcl_encoder.load_state_dict(checkpoint['encoder_state_dict'])
        # self.tcl_encoder.load_state_dict(torch.load('./ckpt/tcl_encoder.pt'))
        self.tcl_classifier = StateClassifier(input_size=encoding_size, output_size=n_states)

        # Build a new encoder to train end-to-end with a classifier
        # self.encoder = RnnEncoder(hidden_size=100, in_channel=3, encoding_size=encoding_size)
        # self.classifier = StateClassifier(input_size=encoding_size, output_size=n_states)
        self.e2e_model = E2EStateClassifier(hidden_size=100, in_channel=3, encoding_size=encoding_size, output_size=n_states)

        self.train_loader, self.valid_loader, self.test_loader = create_simulated_dataset()

    def _train_end_to_end(self, lr):
        self.e2e_model.train()
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.e2e_model.parameters(), lr=lr)

        epoch_loss, epoch_auc = 0, 0
        epoch_acc = 0
        batch_count = 0
        y_all, prediction_all = [], []
        for x, y in self.train_loader:
            optimizer.zero_grad()
            prediction = self.e2e_model(x)
            state_prediction = torch.argmax(prediction, dim=1)
            loss = loss_fn(prediction, y.long())
            loss.backward()
            optimizer.step()
            y_all.append(y)
            # y_onehot = np.zeros(prediction.shape)
            # y_onehot[np.arange(len(y_onehot)), y.cpu().numpy().astype(int)] = 1
            # y_onehot_all.append(y_onehot)
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

    def _train_tcl_classifier(self, lr):
        self.tcl_classifier.train()
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.tcl_classifier.parameters(), lr=lr)

        epoch_loss, epoch_auc = 0, 0
        epoch_acc = 0
        batch_count = 0
        y_all, prediction_all = [], []
        for x, y in self.train_loader:
            optimizer.zero_grad()
            encodings = self.tcl_encoder(x)
            prediction = self.tcl_classifier(encodings)
            state_prediction = torch.argmax(prediction, dim=1)
            loss = loss_fn(prediction, y.long())
            loss.backward()
            optimizer.step()
            y_all.append(y)
            # y_onehot = np.zeros(prediction.shape)
            # y_onehot[np.arange(len(y_onehot)), y.cpu().numpy().astype(int)] = 1
            # y_onehot_all.append(y_onehot)
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
        # print(y_all[:5], prediction_class_all[:5])
        c = confusion_matrix(y_all.astype(int), prediction_class_all)
        return epoch_loss / batch_count, epoch_acc / batch_count, epoch_auc, c

    def _test(self, model, set='valid'):
        model.eval()
        loss_fn = torch.nn.CrossEntropyLoss()
        data_loader = self.valid_loader#{'valid':self.valid_loader, 'test':self.test_loader}[set]

        epoch_loss, epoch_auc = 0, 0
        epoch_acc = 0
        batch_count = 0
        y_all, prediction_all = [], []
        for x, y in data_loader:
            prediction = model(x)
            state_prediction = torch.argmax(prediction, -1)
            loss = loss_fn(prediction, y.long())
            y_all.append(y)
            # y_onehot = np.zeros(prediction.shape)
            # y_onehot[np.arange(len(y_onehot)), y.cpu().numpy().astype(int)] = 1
            # y_onehot_all.append(y_onehot)
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
        tcl_acc, tcl_loss, tcl_auc = [], [], []
        etoe_acc, etoe_loss, etoe_auc = [], [], []
        tcl_acc_test, tcl_loss_test, tcl_auc_test = [], [], []
        etoe_acc_test, etoe_loss_test, etoe_auc_test = [], [], []
        for epoch in range(n_epochs):
            loss, acc, auc, _ = self._train_tcl_classifier(lr_cls)
            print(auc)
            tcl_acc.append(acc)
            tcl_loss.append(loss)
            tcl_auc.append(auc)
            loss, acc, auc, _ = self._train_end_to_end(lr_e2e)
            etoe_acc.append(acc)
            etoe_loss.append(loss)
            etoe_auc.append(auc)
            # Test
            loss, acc, auc, c_mtx_enc = self._test(model=torch.nn.Sequential(self.tcl_encoder, self.tcl_classifier))
            tcl_acc_test.append(acc)
            tcl_loss_test.append(loss)
            tcl_auc_test.append(auc)
            loss, acc, auc, c_mtx_e2e = self._test(model=self.e2e_model) #torch.nn.Sequential(self.encoder, self.classifier))
            etoe_acc_test.append(acc)
            etoe_loss_test.append(loss)
            etoe_auc_test.append(auc)

            print('***** Epoch %d *****'%epoch)
            print('TCL =====> Training Loss: %.3f \t Training Acc: %.3f \t Training AUC: %.3f '
                  '\t Test Loss: %.3f \t Test Acc: %.3f \t Test AUC: %.3f'
                  % (tcl_loss[-1], tcl_acc[-1], tcl_auc[-1], tcl_loss_test[-1], tcl_acc_test[-1], tcl_auc_test[-1]))
            print('End-to-End =====> Training Loss: %.3f \t Training Acc: %.3f \t Training AUC: %.3f'
                  ' \t Test Loss: %.3f \t Test Acc: %.3f \t Test AUC: %.3f'
                  % (etoe_loss[-1], etoe_acc[-1], etoe_auc[-1], etoe_loss_test[-1], etoe_acc_test[-1], etoe_auc_test[-1]))

        # Save performance plots
        plt.figure()
        # plt.plot(np.arange(n_epochs), tcl_loss, label="tcl train")
        # plt.plot(np.arange(n_epochs), etoe_loss, label="e2e train")
        plt.plot(np.arange(n_epochs), tcl_loss_test, label="tcl test")
        plt.plot(np.arange(n_epochs), etoe_loss_test, label="e2e test")
        plt.title("Loss trend for the e2e and tcl model")
        plt.legend()
        plt.savefig(os.path.join("./plots/%s"%data, "classification_loss_comparison.pdf"))

        plt.figure()
        plt.plot(np.arange(n_epochs), tcl_acc, label="tcl train")
        plt.plot(np.arange(n_epochs), etoe_acc, label="e2e train")
        plt.plot(np.arange(n_epochs), tcl_acc_test, label="tcl test")
        plt.plot(np.arange(n_epochs), etoe_acc_test, label="e2e test")
        plt.title("Accuracy trend for the e2e and tcl model")
        plt.legend()
        plt.savefig(os.path.join("./plots/%s"%data, "classification_accuracy_comparison.pdf"))

        plt.figure()
        plt.plot(np.arange(n_epochs), tcl_auc, label="tcl train")
        plt.plot(np.arange(n_epochs), etoe_auc, label="e2e train")
        plt.plot(np.arange(n_epochs), tcl_auc_test, label="tcl test")
        plt.plot(np.arange(n_epochs), etoe_auc_test, label="e2e test")
        plt.title("AUC trend for the e2e and tcl model")
        plt.legend()
        plt.savefig(os.path.join("./plots/%s" % data, "classification_auc_comparison.pdf"))

        df_cm = pd.DataFrame(c_mtx_enc, index=[i for i in ['', '', '', '']],
                             columns=[i for i in ['', '', '', '']])
        plt.figure(figsize=(10, 10))
        sns.heatmap(df_cm, annot=True)
        plt.savefig(os.path.join("./plots/%s"%data, "encoder_cf_matrix.pdf"))

        df_cm = pd.DataFrame(c_mtx_e2e, index=[i for i in ["AFIB", "AFL", "J", "N"]],
                             columns=[i for i in ["AFIB", "AFL", "J", "N"]])
        plt.figure(figsize=(10, 10))
        sns.heatmap(df_cm, annot=True)
        plt.savefig(os.path.join("./plots/%s"%data, "e2e_cf_matrix.pdf"))


class WFClassificationExperiment(ClassificationPerformanceExperiment):
    def __init__(self, n_classes=4, encoding_size=64, window_size=2500, data='waveform', cv=0):
        # Load or train a TCL encoder and an end to end model
        if not os.path.exists("./ckpt/%s/checkpoint_%d.pth.tar"%(data, cv)):
            raise ValueError("No checkpoint for an encoder")
        checkpoint = torch.load('./ckpt/%s/checkpoint_%d.pth.tar'%(data, cv))
        # print('Loading encoder with discrimination performance accuracy of %.3f '%checkpoint['best_accuracy'])
        self.tcl_encoder = WFEncoder(encoding_size=encoding_size)
        self.tcl_encoder.load_state_dict(checkpoint['encoder_state_dict'])
        # self.tcl_encoder.load_state_dict(torch.load('./ckpt/waveform/tcl_encoder.pt'))
        self.tcl_classifier = WFEncoder(encoding_size=encoding_size, classify=True, n_classes=n_classes).classifier
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
        # y_window = np.split(y[:, :window_size * (T // window_size)],(T//window_size), -1)
        x_window = torch.Tensor(np.concatenate(x_window, 0))
        # y_window = torch.Tensor(np.mean(np.concatenate(y_window, 0), -1))
        x_window = x_window[shuffled_inds]
        y_window = y_window[shuffled_inds]
        n_train = int(0.7*len(x_window))
        trainset = torch.utils.data.TensorDataset(x_window[:n_train], y_window[:n_train])
        validset = torch.utils.data.TensorDataset(x_window[n_train:], y_window[n_train:])

        self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True)
        self.valid_loader = torch.utils.data.DataLoader(validset, batch_size=100, shuffle=True)

