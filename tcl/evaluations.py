import os
import torch
import matplotlib.pyplot as plt
import numpy as np

from tcl.models import RnnEncoder, StateClassifier, E2EStateClassifier
from tcl.utils import create_simulated_dataset


class ClassificationPerformanceExperiment():
    def __init__(self, n_states=4, encoding_size=10):
        # Load or train a TCL encoder
        if not os.path.exists("./ckpt/tcl_encoder.pt"):
            raise ValueError("No checkpoint for an encoder")
        self.tcl_encoder = RnnEncoder(hidden_size=100, in_channel=3, encoding_size=encoding_size)
        self.tcl_encoder.load_state_dict(torch.load('./ckpt/encoder.pt'))
        self.tcl_classifier = StateClassifier(input_size=encoding_size, output_size=n_states)

        # Build a new encoder to train end-to-end with a classifier
        # self.encoder = RnnEncoder(hidden_size=100, in_channel=3, encoding_size=encoding_size)
        # self.classifier = StateClassifier(input_size=encoding_size, output_size=n_states)
        self.e2e_model = E2EStateClassifier(hidden_size=100, in_channel=3, encoding_size=encoding_size, output_size=n_states)

        self.train_loader, self.valid_loader, self.test_loader = create_simulated_dataset()

    def _train_end_to_end(self):
        model = self.e2e_model#torch.nn.Sequential(self.encoder, self.classifier)
        model.train()
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        epoch_loss = 0
        epoch_acc = 0
        batch_count = 0
        for x, y in self.train_loader:
            optimizer.zero_grad()
            prediction = model(x)
            state_prediction = torch.argmax(prediction, dim=1)
            loss = loss_fn(prediction, y.long())
            loss.backward()
            optimizer.step()

            epoch_acc += torch.eq(state_prediction, y).sum().item()/len(x)
            epoch_loss += loss.item()
            batch_count += 1
        return epoch_loss / batch_count, epoch_acc / batch_count

    def _train_tcl_classifier(self):
        self.tcl_classifier.train()
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.tcl_classifier.parameters(), lr=0.01)

        epoch_loss = 0
        epoch_acc = 0
        batch_count = 0
        for x, y in self.train_loader:
            optimizer.zero_grad()
            encodings = self.tcl_encoder(x)
            prediction = self.tcl_classifier(encodings)
            state_prediction = torch.argmax(prediction, dim=1)
            loss = loss_fn(prediction, y.long())
            loss.backward()
            optimizer.step()

            epoch_acc += torch.eq(state_prediction, y).sum().item()/len(x)
            epoch_loss += loss.item()
            batch_count += 1
        return epoch_loss / batch_count, epoch_acc / batch_count

    def _test(self, model, set='valid'):
        model.eval()
        loss_fn = torch.nn.CrossEntropyLoss()
        data_loader = {'valid':self.valid_loader, 'test':self.test_loader}[set]

        epoch_loss = 0
        epoch_acc = 0
        batch_count = 0
        for x, y in data_loader:
            prediction = model(x)
            state_prediction = torch.argmax(prediction, -1)
            loss = loss_fn(prediction, y.long())

            epoch_acc += torch.eq(state_prediction, y).sum().item()/len(x)
            epoch_loss += loss.item()
            batch_count += 1
        return epoch_loss / batch_count, epoch_acc / batch_count

    def run(self):
        n_epochs = 200
        tcl_acc, tcl_loss = [], []
        etoe_acc, etoe_loss = [], []
        tcl_acc_test, tcl_loss_test = [], []
        etoe_acc_test, etoe_loss_test = [], []
        for epoch in range(n_epochs):
            loss, acc = self._train_tcl_classifier()
            tcl_acc.append(acc)
            tcl_loss.append(loss)
            loss, acc = self._train_end_to_end()
            etoe_acc.append(acc)
            etoe_loss.append(loss)
            # Test
            loss, acc = self._test(model=torch.nn.Sequential(self.tcl_encoder, self.tcl_classifier))
            tcl_acc_test.append(acc)
            tcl_loss_test.append(loss)
            loss, acc = self._test(model=self.e2e_model) #torch.nn.Sequential(self.encoder, self.classifier))
            etoe_acc_test.append(acc)
            etoe_loss_test.append(loss)

            print('***** Epoch %d *****'%epoch)
            print('TCL =====> \t Training Loss: %.5f \t Training Accuracy: %.5f \t Test Loss: %.5f \t Test Accuracy: %.5f'
                        % (tcl_loss[-1], tcl_acc[-1], tcl_loss_test[-1], tcl_acc_test[-1]))
            print('End-to-End =====> \t Training Loss: %.5f \t Training Accuracy: %.5f \t Test Loss: %.5f \t Test Accuracy: %.5f'
                  % (etoe_loss[-1], etoe_acc[-1], etoe_loss_test[-1], etoe_acc_test[-1]))

        # Save performance plots
        plt.figure()
        plt.plot(np.arange(n_epochs), tcl_loss, label="tcl train")
        plt.plot(np.arange(n_epochs), etoe_loss, label="e2e train")
        plt.plot(np.arange(n_epochs), tcl_loss_test, label="tcl test")
        plt.plot(np.arange(n_epochs), etoe_loss_test, label="e2e test")
        plt.title("Loss trend for the e2e and tcl model")
        plt.legend()
        plt.savefig(os.path.join("./plots", "classification_loss_comparison.pdf"))

        plt.figure()
        plt.plot(np.arange(n_epochs), tcl_acc, label="tcl train")
        plt.plot(np.arange(n_epochs), etoe_acc, label="e2e train")
        plt.plot(np.arange(n_epochs), tcl_acc_test, label="tcl test")
        plt.plot(np.arange(n_epochs), etoe_acc_test, label="e2e test")
        plt.title("Accuracy trend for the e2e and tcl model")
        plt.legend()
        plt.savefig(os.path.join("./plots", "classification_accuracy_comparison.pdf"))