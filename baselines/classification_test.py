import os
import torch
import numpy as np
import pickle
import random
import argparse

from tcl.models import RnnEncoder, StateClassifier, E2EStateClassifier, WFEncoder, WFClassifier
from tcl.utils import create_simulated_dataset
from tcl.tcl import learn_encoder as train_tcl
from baselines.causal_cnn import learn_encoder as train_trip
from baselines.cpc import learn_encoder as train_cpc
from baselines.knn import KnnDtw

from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, silhouette_score
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
        x = x.to(device)
        prediction = model(x)
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
    del x, y
    y_all = np.concatenate(y_all, 0)
    prediction_all = np.concatenate(prediction_all, 0)
    prediction_class_all = np.argmax(prediction_all, -1)
    y_onehot_all = np.zeros(prediction_all.shape)
    y_onehot_all[np.arange(len(y_onehot_all)), y_all.astype(int)] = 1
    epoch_auc = roc_auc_score(y_onehot_all, prediction_all)
    c = confusion_matrix(y_all.astype(int), prediction_class_all)
    return epoch_loss / batch_count, epoch_acc / batch_count, epoch_auc, c


def epoch_run_encoder(encoder, classifier, dataloader, optimizer=None, train=False):
    encoder.eval()
    if train:
        classifier.train()
    else:
        classifier.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)

    epoch_loss, epoch_auc = 0, 0
    epoch_acc = 0
    batch_count = 0
    y_all, prediction_all = [], []
    for x, y in dataloader:
        y = y.to(device)
        x = x.to(device)
        encodings = encoder(x)
        prediction = classifier(encodings)
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
    del x, y
    y_all = np.concatenate(y_all, 0)
    prediction_all = np.concatenate(prediction_all, 0)
    prediction_class_all = np.argmax(prediction_all, -1)
    y_onehot_all = np.zeros(prediction_all.shape)
    y_onehot_all[np.arange(len(y_onehot_all)), y_all.astype(int)] = 1
    epoch_auc = roc_auc_score(y_onehot_all, prediction_all)
    c = confusion_matrix(y_all.astype(int), prediction_class_all)
    return epoch_loss / batch_count, epoch_acc / batch_count, epoch_auc, c


def train(train_loader, valid_loader, classifier, optimizer, data_type, encoder=None, n_epochs=100, type='e2e', cv=0):
    best_auc, best_acc = 0, 0
    for epoch in range(n_epochs):
        if type=='e2e':
            train_loss, train_acc, train_auc, _ = epoch_run(classifier, optimizer=optimizer,  dataloader=train_loader, train=True)
            test_loss, test_acc, test_auc, _ = epoch_run(classifier, dataloader=valid_loader, train=False)
        else:
            train_loss, train_acc, train_auc, _  = epoch_run_encoder(encoder=encoder, classifier=classifier, dataloader=train_loader, optimizer=optimizer, train=True)
            test_loss, test_acc, test_auc, _  = epoch_run_encoder(encoder=encoder, classifier=classifier, dataloader=valid_loader, train=False)
        print(train_loss, train_acc, train_auc, '\t||\t', test_loss, test_acc, test_auc)
        # print(train_loss, train_acc, train_auc)
        # print(test_loss, test_acc, test_auc)
        if test_auc>best_auc:
            best_auc = test_auc
            best_acc = test_acc
            if type == 'e2e':
                state = {
                    'epoch': epoch,
                    'state_dict': classifier.state_dict(),
                    'best_accuracy': test_acc,
                    'best_accuracy': best_auc
                }
            else:
                state = {
                    'epoch': epoch,
                    'state_dict': torch.nn.Sequential(encoder, classifier).state_dict(),
                    'best_accuracy': test_acc,
                    'best_accuracy': best_auc
                }
            if not os.path.exists( './ckpt/classifier_test/%s'%data_type):
                os.mkdir( './ckpt/classifier_test/%s'%data_type)
            torch.save(state, './ckpt/classifier_test/%s/%s_checkpoint_%d.pth.tar'%(data_type, type, cv))
    return best_acc, best_auc


def run_test(data, e2e_lr, tcl_lr, cpc_lr, trip_lr, data_path, window_size):
    # kf = KFold(n_splits=2)
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
    # x_window = np.concatenate(np.split(x[:, :, :T // 5 * 5], 5, -1), 0)
    # y_window = np.concatenate(np.split(y[:, :5 * (T // 5)], 5, -1), 0).astype(int)
    # y_window = np.array([np.bincount(yy).argmax() for yy in y_window])

    # T = x.shape[-1]
    x_window = np.split(x[:, :, :window_size * (T // window_size)], (T // window_size), -1)
    y_window = np.concatenate(np.split(y[:, :window_size * (T // window_size)], (T // window_size), -1), 0).astype(int)
    x_window = torch.Tensor(np.concatenate(x_window, 0))
    y_window = torch.Tensor(np.array([np.bincount(yy).argmax() for yy in y_window]))

    x_window_test = np.split(x_test[:, :, :window_size * (T // window_size)], (T // window_size), -1)
    y_window_test = np.concatenate(np.split(y_test[:, :window_size * (T // window_size)], (T // window_size), -1), 0).astype(int)
    x_window_test = torch.Tensor(np.concatenate(x_window_test, 0))
    y_window_test = torch.Tensor(np.array([np.bincount(yy).argmax() for yy in y_window_test]))

    testset = torch.utils.data.TensorDataset(x_window_test, y_window_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=True)

    del x, y, x_test, y_test
    # for train_index, test_index in kf.split(x_window):
    for cv in range(4):
        shuffled_inds = list(range(len(x_window)))
        random.shuffle(shuffled_inds)
        x_window = x_window[shuffled_inds]
        y_window = y_window[shuffled_inds]
        # X_train, X_test = x_window[train_index], x_window[test_index]
        # y_train, y_test = y_window[train_index], y_window[test_index]
        n_train = int(0.7*len(x_window))
        X_train, X_test = x_window[:n_train], x_window[n_train:]
        y_train, y_test = y_window[:n_train], y_window[n_train:]
        print(X_train.shape, y_train.shape)
        # print(np.split(y_train[:window_size * (T // window_size)],0).shape)

        # T = X_train.shape[-1]
        # x_chopped = np.split(X_train[:,:, :window_size * (T // window_size)], (T // window_size), -1)
        # y_chopped = np.concatenate(np.split(y_train[:, :window_size * (T // window_size)], (T // window_size), -1), 0).astype(int)
        # x_chopped = torch.Tensor(np.concatenate(x_chopped, 0))
        # y_chopped = torch.Tensor(np.array([np.bincount(yy).argmax() for yy in y_chopped]))
        # T = X_test.shape[-1]
        # x_chopped_test = np.split(X_test[:, :, :window_size * (T // window_size)], (T // window_size), -1)
        # y_chopped_test = np.concatenate(np.split(y_test[:, :window_size * (T // window_size)], (T // window_size), -1),
        #                            0).astype(int)
        # x_chopped_test = torch.Tensor(np.concatenate(x_chopped_test, 0))
        # y_chopped_test = torch.Tensor(np.array([np.bincount(yy).argmax() for yy in y_chopped_test]))
        # shuffled_inds = list(range(len(y_chopped_test)))
        # random.shuffle(shuffled_inds)

        x_chopped, y_chopped = X_train, y_train
        x_chopped_test, y_chopped_test = X_test, y_test
        print('Distribution of Traning and Test set')
        print('Train: ', (y_chopped.cpu().numpy()==0).astype(int).sum(), (y_chopped.cpu().numpy()==1).astype(int).sum(),
              (y_chopped.cpu().numpy()==2).astype(int).sum(), (y_chopped.cpu().numpy()==3).astype(int).sum())
        print('Test: ', (y_chopped_test.cpu().numpy()==0).astype(int).sum(), (y_chopped_test.cpu().numpy()==1).astype(int).sum(),
              (y_chopped_test.cpu().numpy()==2).astype(int).sum(), (y_chopped_test.cpu().numpy()==3).astype(int).sum())



        trainset = torch.utils.data.TensorDataset(X_train, y_train)
        validset = torch.utils.data.TensorDataset(X_test, y_test)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=False)
        valid_loader = torch.utils.data.DataLoader(validset, batch_size=200, shuffle=False)

        # Define baseline models
        if data == 'waveform':
            encoding_size = 64
            n_classes = 4

            e2e_model = WFEncoder(encoding_size=encoding_size, classify=True, n_classes=n_classes).to(device)

            tcl_encoder = WFEncoder(encoding_size=encoding_size).to(device)
            # train_tcl(torch.Tensor(X_train), tcl_encoder, lr=1e-6, decay=1e-3, n_epochs=100,
            #               window_size=window_size, delta=400000,
            #               epsilon=1.5, path='waveform', mc_sample_size=10, device=device, augmentation=5, cv=cv)
            tcl_checkpoint = torch.load('./ckpt/waveform/checkpoint_%d.pth.tar'%cv)
            tcl_encoder.load_state_dict(tcl_checkpoint['encoder_state_dict'])
            print('TCL best accuracy: ', tcl_checkpoint['best_accuracy'])
            tcl_classifier = WFClassifier(encoding_size=encoding_size, output_size=4)#WFEncoder(encoding_size=encoding_size, classify=True, n_classes=n_classes).classifier
            tcl_model = torch.nn.Sequential(tcl_encoder, tcl_classifier).to(device)

            cpc_encoder = WFEncoder(encoding_size=encoding_size).to(device)
            # train_cpc(X_train, cpc_encoder, window_size, n_epochs=30, lr=1e-5, decay=1e-3, data='waveform', cv=cv)
            cpc_checkpoint = torch.load('./ckpt/waveform_cpc/checkpoint_%d.pth.tar'%cv)
            cpc_encoder.load_state_dict(cpc_checkpoint['encoder_state_dict'])
            cpc_classifier = WFClassifier(encoding_size=encoding_size, output_size=4)#WFEncoder(encoding_size=encoding_size, classify=True, n_classes=n_classes).classifier
            cpc_model = torch.nn.Sequential(cpc_encoder, cpc_classifier).to(device)

            trip_encoder = WFEncoder(encoding_size=encoding_size).to(device)
            # train_trip(X_train, trip_encoder, window_size, n_epochs=50, lr=1e-5, decay=1e-2, data='waveform', cv=cv)
            trip_checkpoint = torch.load('./ckpt/waveform_trip/checkpoint_%d.pth.tar'%cv)
            trip_encoder.load_state_dict(trip_checkpoint['encoder_state_dict'])
            trip_classifier = WFClassifier(encoding_size=encoding_size, output_size=4)#WFEncoder(encoding_size=encoding_size, classify=True, n_classes=n_classes).classifier
            trip_model = torch.nn.Sequential(trip_encoder, trip_classifier).to(device)
            n_epochs = 5
            n_epoch_e2e = 5


        elif data == 'simulation':
            m_1 = KnnDtw()
            m_1.fit(X_train[:, 0, :], y_train)
            m_2 = KnnDtw()
            m_2.fit(X_train[:, 1, :], y_train)
            m_3 = KnnDtw()
            m_3.fit(X_train[:, 2, :], y_train)
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
            n_epochs = 200
            n_epoch_e2e = 200


        # Train the model
        e2e_optimizer = torch.optim.Adam(e2e_model.parameters(), lr=e2e_lr)
        tcl_optimizer = torch.optim.Adam(tcl_classifier.parameters(), lr=tcl_lr)
        cpc_optimizer = torch.optim.Adam(cpc_classifier.parameters(), lr=cpc_lr)
        trip_optimizer = torch.optim.Adam(trip_classifier.parameters(), lr=trip_lr)
        print('Starting E2E ......')
        best_acc_e2e, best_auc_e2e = train(train_loader, valid_loader, e2e_model, e2e_optimizer,
                             data_type=data, n_epochs=n_epoch_e2e, type='e2e', cv=cv)
        print('E2E: ', best_acc_e2e, best_auc_e2e)
        print('Starting TCL ......')
        best_acc_tcl, best_auc_tcl = train(train_loader, valid_loader, tcl_classifier, tcl_optimizer,
                                           encoder=tcl_encoder, data_type=data, n_epochs=n_epochs, type='tcl', cv=cv)
        print('TCL: ', best_acc_tcl, best_auc_tcl)
        print('Starting CPC ......')
        best_acc_cpc, best_auc_cpc = train(train_loader, valid_loader, cpc_classifier, cpc_optimizer,
                                           encoder=cpc_encoder, data_type=data, n_epochs=n_epochs, type='cpc', cv=cv)
        print('CPC: ', best_acc_cpc, best_auc_cpc)
        print('Starting Trip ......')
        best_acc_trip, best_auc_trip = train(train_loader, valid_loader, trip_classifier, trip_optimizer,
                                             encoder=trip_encoder, data_type=data, n_epochs=n_epochs, type='trip', cv=cv)
        print('TRIP: ', best_acc_trip, best_auc_trip)

        # Evaluate performance on Held out test
        # checkpoint = torch.load('./ckpt/classifier_test/%s/e2e_checkpoint_%d.pth.tar'%(data, cv))
        # e2e_model.load_state_dict(checkpoint['state_dict'])
        # checkpoint = torch.load('./ckpt/classifier_test/%s/tcl_checkpoint_%d.pth.tar'%(data, cv))
        # tcl_model.load_state_dict(checkpoint['state_dict'])
        # checkpoint = torch.load('./ckpt/classifier_test/%s/cpc_checkpoint_%d.pth.tar'%(data, cv))
        # cpc_model.load_state_dict(checkpoint['state_dict'])
        # checkpoint = torch.load('./ckpt/classifier_test/%s/trip_checkpoint_%d.pth.tar'%(data, cv))
        # trip_model.load_state_dict(checkpoint['state_dict'])
        # torch.cuda.empty_cache()
        # print('\nPerformances:')
        # _, e2e_acc, e2e_auc, _ = epoch_run(e2e_model, test_loader, train=False)
        # _, tcl_acc, tcl_auc, _ = epoch_run(tcl_model, test_loader, train=False)
        # _, cpc_acc, cpc_auc, _ = epoch_run(cpc_model, test_loader, train=False)
        # _, trip_acc, trip_auc, _ = epoch_run(trip_model, test_loader, train=False)
        #
        # print(e2e_acc, e2e_auc)
        # print(tcl_acc, tcl_auc)
        # print(cpc_acc, cpc_auc)
        # print(trip_acc, trip_auc)

        # rndm_ind = np.random.randint(0, len(x_test), 100)
        # label_1, _ = m_1.predict(x_test[rndm_ind, 0, :])
        # label_2, _ = m_2.predict(x_test[rndm_ind, 1, :])
        # label_3, _ = m_3.predict(x_test[rndm_ind, 2, :])
        # stacked_label = np.stack([label_1, label_2, label_3]).astype(int)
        # label = []
        # for vote in stacked_label.T:
        #     label.append(np.bincount(vote).argmax())
        # dtw_acc = accuracy_score(y_test, np.array(label))
        # dtw_auc = roc_auc_score(y_test, np.array(label))
        # print("DTW model: \t AUC: %.4f\t Accuracy: %.4f" % (dtw_auc, dtw_acc))

        ## Silhouette test
        # tcl_encodings = tcl_model[0](torch.Tensor(x_window_test).to(device))
        # tcl_predictions = tcl_model[]
        # tcl_sil_score = silhouette_score(tcl_encodings.cpu().detach().numpy(), y_window_test)

        # cv += 1
        with open("./outputs/waveform_classifiers.txt", "a") as f:
            f.write("Performance result for a fold \n" )
            f.write("End-tp-End model: \t AUC: %s\t Accuracy: %s \n\n" % (str(best_auc_e2e), str(best_acc_e2e)))
            f.write("TCL model: \t AUC: %s\t Accuracy: %s \n\n" % (str(best_auc_tcl), str(best_acc_tcl)))
            f.write("CPC model: \t AUC: %s\t Accuracy: %s \n\n" % (str(best_auc_cpc), str(best_acc_cpc)))
            f.write("Triplet Loss model: \t AUC: %s\t Accuracy: %s \n\n" % (str(best_acc_trip), str(best_acc_trip)))
            # f.write("DTW model: \t AUC: %s\t Accuracy: %s \n\n" % (str(dtw_auc), str(dtw_acc)))

        torch.cuda.empty_cache()


if __name__=='__main__':
    # print('***********', torch.cuda.is_available())
    parser = argparse.ArgumentParser(description='Run baseline model for explanation')
    # parser.add_argument('--data', type=str, default='simulation')
    # parser.add_argument('--cv', type=int, default=0)
    args = parser.parse_args()
    random.seed(1234)
    f = open("./outputs/waveform_classifiers.txt", "w")
    f.close()
    run_test(data='waveform', e2e_lr=0.0001, tcl_lr=0.01, cpc_lr=0.001, trip_lr=0.001,
             data_path='./data/waveform_data/processed', window_size=2500)
    # run_test(data='simulation', e2e_lr=0.001, tcl_lr=0.001, cpc_lr=0.001, trip_lr=0.001,
    #          data_path='./data/simulated_data/', window_size=50)


