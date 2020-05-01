import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import pandas as pd

from tcl.models import RnnEncoder, StateClassifier, E2EStateClassifier, WFEncoder
from tcl.utils import create_simulated_dataset

from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


class ClassificationPerformanceExperiment():
    def __init__(self, n_states=4, encoding_size=10):
        # Load or train a TCL encoder
        if not os.path.exists("./ckpt/simulation/checkpoint.pth.tar"):
            raise ValueError("No checkpoint for an encoder")
        checkpoint = torch.load('./ckpt/simulation/checkpoint.pth.tar')
        print('Loading encoder with discrimination performance accuracy of %.3f '%checkpoint['best_accuracy'])
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
    def __init__(self, n_classes=4, encoding_size=64, window_size=2500):
        # super(WFClassificationExperiment, self).__init__()

        # Load or train a TCL encoder and an end to end model
        if not os.path.exists("./ckpt/waveform/tcl_encoder.pt"):
            raise ValueError("No checkpoint for an encoder")
        checkpoint = torch.load('./ckpt/waveform/checkpoint.pth.tar')
        print('Loading encoder with discrimination performance accuracy of %.3f '%checkpoint['best_accuracy'])
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
        y_window = np.split(y[:, :window_size * (T // window_size)],(T//window_size), -1)
        x_window = torch.Tensor(np.concatenate(x_window, 0))
        y_window = torch.Tensor(np.mean(np.concatenate(y_window, 0), -1))
        n_train = int(0.8*len(x_window))
        trainset = torch.utils.data.TensorDataset(x_window[:n_train], y_window[:n_train])
        validset = torch.utils.data.TensorDataset(x_window[n_train:], y_window[n_train:])

        self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True)
        self.valid_loader = torch.utils.data.DataLoader(validset, batch_size=100, shuffle=True)




















# class MimicExperiment():
#     def __init__(self,n):
#         self.train_loader, self.valid_loader, self.test_loader = create_mimic_dataset
#
#     def _train(self, train_loader, model, device, optimizer, loss_criterion=torch.nn.BCELoss()):
#         model = model.to(device)
#         model.train()
#         auc_train = 0
#         recall_train, precision_train, auc_train, correct_label, epoch_loss = 0, 0, 0, 0, 0
#         for i, (signals, labels) in enumerate(train_loader):
#             optimizer.zero_grad()
#             signals, labels = torch.Tensor(signals.float()).to(device), torch.Tensor(labels.float()).to(device)
#             labels = labels.view(labels.shape[0], )
#             labels = labels.view(labels.shape[0], )
#             risks = model(signals)
#             predicted_label = (risks > 0.5).view(len(labels), ).float()
#             auc, recall, precision, correct = evaluate(labels, predicted_label, risks)
#             correct_label += correct
#             auc_train = auc_train + auc
#             recall_train = + recall
#             precision_train = + precision
#
#             loss = loss_criterion(risks.view(len(labels), ), labels)
#             epoch_loss = + loss.item()
#             loss.backward()
#             optimizer.step()
#         return recall_train, precision_train, auc_train / (i + 1), correct_label, epoch_loss, i + 1
#
#     def _train_model(self, model, train_loader, valid_loader, optimizer, n_epochs, device, experiment, data='mimic'):
#         train_loss_trend = []
#         test_loss_trend = []
#
#         for epoch in range(n_epochs + 1):
#             recall_train, precision_train, auc_train, correct_label_train, epoch_loss, n_batches = train(train_loader,
#                                                                                                          model,
#                                                                                                          device,
#                                                                                                          optimizer)
#             recall_test, precision_test, auc_test, correct_label_test, test_loss = test(valid_loader, model,
#                                                                                         device)
#             train_loss_trend.append(epoch_loss)
#             test_loss_trend.append(test_loss)
#             if epoch % 10 == 0:
#                 print('\nEpoch %d' % (epoch))
#                 print('Training ===>loss: ', epoch_loss,
#                       ' Accuracy: %.2f percent' % (100 * correct_label_train / (len(train_loader.dataset))),
#                       ' AUC: %.2f' % (auc_train))
#                 print('Test ===>loss: ', test_loss,
#                       ' Accuracy: %.2f percent' % (100 * correct_label_test / (len(valid_loader.dataset))),
#                       ' AUC: %.2f' % (auc_test))
#
#         # Save model and results
#         if not os.path.exists(os.path.join("./ckpt/", data)):
#             os.mkdir("./ckpt/")
#             os.mkdir(os.path.join("./ckpt/", data))
#         if not os.path.exists(os.path.join("./plots/", data)):
#             os.mkdir("./plots/")
#             os.mkdir(os.path.join("./plots/", data))
#         torch.save(model.state_dict(), './ckpt/' + data + '/' + str(experiment) + '.pt')
#         plt.plot(train_loss_trend, label='Train loss')
#         plt.plot(test_loss_trend, label='Validation loss')
#         plt.legend()
#         plt.savefig(os.path.join('./plots', data, 'train_loss.pdf'))


