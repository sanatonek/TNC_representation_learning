import os
import torch
import numpy as np
import pickle

from tcl.models import RnnEncoder, StateClassifier, E2EStateClassifier, WFEncoder
from tcl.utils import create_simulated_dataset

from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def epoch_run(model, dataloader, optimizer=None, train=False):
    if train:
        model.train()
    else:
        model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()

    epoch_loss, epoch_auc = 0, 0
    epoch_acc = 0
    batch_count = 0
    y_all, prediction_all = [], []
    for x, y in dataloader:
        y = y.to(device)
        prediction = model(x.to(device))
        state_prediction = torch.argmax(prediction, dim=1)
        loss = loss_fn(prediction, y.long())
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        y_all.append(y.cpu().detach().numpy())
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


def train(train_loader, valid_loader, classifier, optimizer, n_epochs=100, type='e2e'):
    best_acc = 0
    for epoch in range(n_epochs):
        train_loss, train_acc, train_auc, _ = epoch_run(classifier, optimizer=optimizer,  dataloader=train_loader, train=True)
        test_loss, test_acc, test_auc, _ = epoch_run(classifier, dataloader=valid_loader, train=False)
        if test_acc>best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'state_dict': classifier.state_dict(),
                'best_accuracy': best_acc
            }
            torch.save(state, './ckpt/classifier_test/%s_checkpoint.pth.tar'%type)
    return best_acc


def run_test(data, e2e_lr, tcl_lr, cpc_lr, trip_lr, data_path, window_size):
    kf = KFold(n_splits=4)
    f = open("./outputs/waveform_classifiers.txt", "w")
    f.close()
    # Load data
    with open(os.path.join(data_path, 'x_train.pkl'), 'rb') as f:
        x = pickle.load(f)
    with open(os.path.join(data_path, 'state_train.pkl'), 'rb') as f:
        y = pickle.load(f)
    with open(os.path.join(data_path, 'x_test.pkl'), 'rb') as f:
        x_test = pickle.load(f)
    with open(os.path.join(data_path, 'state_test.pkl'), 'rb') as f:
        y_test = pickle.load(f)
    T = x.shape[-1]
    x_window = np.split(x[:, :, :window_size * (T // window_size)], (T // window_size), -1)
    y_window = np.split(y[:, :window_size * (T // window_size)], (T // window_size), -1)
    x_window = torch.Tensor(np.concatenate(x_window, 0))
    y_window = torch.Tensor(np.mean(np.concatenate(y_window, 0), -1))
    x_window_test = np.split(x_test[:, :, :window_size * (T // window_size)], (T // window_size), -1)
    y_window_test = np.split(y_test[:, :window_size * (T // window_size)], (T // window_size), -1)
    x_window_test = torch.Tensor(np.concatenate(x_window_test, 0))
    y_window_test = torch.Tensor(np.mean(np.concatenate(y_window_test, 0), -1))
    testset = torch.utils.data.TensorDataset(x_window_test, y_window_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=True)

    for train_index, test_index in kf.split(x_window):
        X_train, X_test = x_window[train_index], x_window[test_index]
        y_train, y_test = y_window[train_index], y_window[test_index]
        trainset = torch.utils.data.TensorDataset(X_train, y_train)
        validset = torch.utils.data.TensorDataset(X_test, y_test)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(validset, batch_size=100, shuffle=True)

        # Define baseline models
        if data == 'waveform':
            encoding_size = 64
            n_classes = 4

            e2e_model = WFEncoder(encoding_size=encoding_size, classify=True, n_classes=n_classes).to(device)

            tcl_encoder =WFEncoder(encoding_size=encoding_size)
            tcl_checkpoint = torch.load('./ckpt/waveform/checkpoint.pth.tar')
            tcl_encoder.load_state_dict(tcl_checkpoint['encoder_state_dict'])
            tcl_classifier = WFEncoder(encoding_size=encoding_size, classify=True, n_classes=n_classes).classifier
            tcl_model = torch.nn.Sequential(tcl_encoder, tcl_classifier).to(device)

            cpc_encoder = WFEncoder(encoding_size=encoding_size)
            cpc_checkpoint = torch.load('./ckpt/waveform_cpc/checkpoint.pth.tar')
            cpc_encoder.load_state_dict(cpc_checkpoint['encoder_state_dict'])
            cpc_classifier = WFEncoder(encoding_size=encoding_size, classify=True, n_classes=n_classes).classifier
            cpc_model = torch.nn.Sequential(cpc_encoder, cpc_classifier).to(device)

            trip_encoder = WFEncoder(encoding_size=encoding_size)
            trip_checkpoint = torch.load('./ckpt/waveform_trip/checkpoint.pth.tar')
            trip_encoder.load_state_dict(trip_checkpoint['encoder_state_dict'])
            trip_classifier = WFEncoder(encoding_size=encoding_size, classify=True, n_classes=n_classes).classifier
            trip_model = torch.nn.Sequential(trip_encoder, trip_classifier).to(device)

        elif data == 'simulation':
            encoding_size = 10

            e2e_model = E2EStateClassifier(hidden_size=100, in_channel=3, encoding_size=encoding_size,
                                           output_size=4, device=device)

            tcl_encoder = RnnEncoder(hidden_size=100, in_channel=3, encoding_size=encoding_size, device=device)
            tcl_checkpoint = torch.load('./ckpt/simulation/checkpoint.pth.tar')
            tcl_encoder.load_state_dict(tcl_checkpoint['encoder_state_dict'])
            tcl_classifier = StateClassifier(input_size=encoding_size, output_size=4)
            tcl_model = torch.nn.Sequential(tcl_encoder, tcl_classifier).to(device)

            cpc_encoder = RnnEncoder(hidden_size=100, in_channel=3, encoding_size=encoding_size, device=device)
            cpc_checkpoint = torch.load('./ckpt/simulation_cpc/checkpoint.pth.tar')
            cpc_encoder.load_state_dict(cpc_checkpoint['encoder_state_dict'])
            cpc_classifier = StateClassifier(input_size=encoding_size, output_size=4)
            cpc_model = torch.nn.Sequential(cpc_encoder, cpc_classifier).to(device)

            trip_encoder = RnnEncoder(hidden_size=100, in_channel=3, encoding_size=encoding_size, device=device)
            trip_checkpoint = torch.load('./ckpt/simulation_trip/checkpoint.pth.tar')
            trip_encoder.load_state_dict(trip_checkpoint['encoder_state_dict'])
            trip_classifier = StateClassifier(input_size=encoding_size, output_size=4)
            trip_model = torch.nn.Sequential(trip_encoder, trip_classifier).to(device)


        # Train the model
        e2e_optimizer = torch.optim.Adam(e2e_model.parameters(), lr=e2e_lr)
        tcl_optimizer = torch.optim.Adam(tcl_classifier.parameters(), lr=tcl_lr)
        cpc_optimizer = torch.optim.Adam(cpc_classifier.parameters(), lr=cpc_lr)
        trip_optimizer = torch.optim.Adam(trip_classifier.parameters(), lr=trip_lr)
        best_acc_e2e = train(train_loader, valid_loader, e2e_model, e2e_optimizer, n_epochs=10, type='e2e')
        best_acc_tcl = train(train_loader, valid_loader, tcl_model, tcl_optimizer, n_epochs=10, type='tcl')
        best_acc_cpc = train(train_loader, valid_loader, cpc_model, cpc_optimizer, n_epochs=10, type='cpc')
        best_acc_trip = train(train_loader, valid_loader, trip_model, trip_optimizer, n_epochs=10, type='trip')

        # Evaluate performance on Held out test
        checkpoint = torch.load('./ckpt/classifier_test/e2e_checkpoint.pth.tar')
        e2e_model.load_state_dict(checkpoint['state_dict'])
        checkpoint = torch.load('./ckpt/classifier_test/tcl_checkpoint.pth.tar')
        tcl_model.load_state_dict(checkpoint['state_dict'])
        checkpoint = torch.load('./ckpt/classifier_test/cpc_checkpoint.pth.tar')
        cpc_model.load_state_dict(checkpoint['state_dict'])
        checkpoint = torch.load('./ckpt/classifier_test/trip_checkpoint.pth.tar')
        trip_model.load_state_dict(checkpoint['state_dict'])
        print('\nPerformances:')
        _, e2e_acc, e2e_auc, _ = epoch_run(e2e_model, test_loader, train=False)
        print("End-to-End model: \t AUC: %.4f\t Accuracy: %.4f " % (e2e_auc, best_acc_e2e))
        _, tcl_acc, tcl_auc, _ = epoch_run(tcl_model, test_loader, train=False)
        print("TCL model: \t AUC: %.4f\t Accuracy: %.4f " % (tcl_auc, best_acc_tcl))
        _, cpc_acc, cpc_auc, _ = epoch_run(cpc_model, test_loader, train=False)
        print("CPC model: \t AUC: %.4f\t Accuracy: %.4f" % (cpc_auc, best_acc_cpc))
        _, trip_acc, trip_auc, _ = epoch_run(trip_model, test_loader, train=False)
        print("Triplet Loss model: \t AUC: %.4f\t Accuracy: %.4f" % (trip_auc, best_acc_trip))

        with open("./outputs/waveform_classifiers.txt", "a") as f:
            f.write("Performance result for a fold \n" )
            f.write("End-tp-End model: \t AUC: %s\t Accuracy: %s \n\n" % (str(e2e_auc), str(e2e_acc)))
            f.write("TCL model: \t AUC: %s\t Accuracy: %s \n\n" % (str(tcl_auc), str(tcl_acc)))
            f.write("CPC model: \t AUC: %s\t Accuracy: %s \n\n" % (str(cpc_auc), str(cpc_acc)))
            f.write("Triplet Loss model: \t AUC: %s\t Accuracy: %s \n\n" % (str(trip_auc), str(trip_acc)))


if __name__=='__main__':
    run_test(data='simulation', e2e_lr=0.01, tcl_lr=0.001, cpc_lr=0.01, trip_lr=0.01, data_path='./data/simulated_data/', window_size=50)


