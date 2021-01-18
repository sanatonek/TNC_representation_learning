import os
import torch
import numpy as np
import pickle
import random
import argparse
import matplotlib.pyplot as plt

from tnc.models import RnnEncoder, StateClassifier, E2EStateClassifier, WFEncoder, WFClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
from sklearn.metrics import average_precision_score


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def epoch_run(model, dataloader, train=False, lr=0.01):
    if train:
        model.train()
    else:
        model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
        prediction_all.append(torch.nn.Softmax(-1)(prediction).detach().cpu().numpy())
        # prediction_all.append(prediction.detach().cpu().numpy())

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
    epoch_auprc = average_precision_score(y_onehot_all, prediction_all)
    c = confusion_matrix(y_all.astype(int), prediction_class_all)
    return epoch_loss / batch_count, epoch_acc / batch_count, epoch_auc, epoch_auprc, c


def epoch_run_encoder(encoder, classifier, dataloader, train=False, lr=0.01):
    # encoder.eval()
    if train:
        classifier.train()
        encoder.train()
    else:
        classifier.eval()
        encoder.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=1e-4)

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
        prediction_all.append(torch.nn.Softmax(-1)(prediction).detach().cpu().numpy())
        # prediction_all.append(prediction.detach().cpu().numpy())

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
    epoch_auprc = average_precision_score(y_onehot_all, prediction_all)
    c = confusion_matrix(y_all.astype(int), prediction_class_all)
    return epoch_loss / batch_count, epoch_acc / batch_count, epoch_auc, epoch_auprc, c


def train(train_loader, valid_loader, classifier, lr, data_type, encoder=None, n_epochs=100, type='e2e', cv=0):
    best_auc, best_acc, best_aupc, best_loss = 0, 0, 0, np.inf
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    for epoch in range(n_epochs):
        if type=='e2e':
            train_loss, train_acc, train_auc, train_auprc, _ = epoch_run(classifier, dataloader=train_loader, train=True, lr=lr)
            test_loss, test_acc, test_auc, test_auprc,  _ = epoch_run(classifier, dataloader=valid_loader, train=False)
        else:
            train_loss, train_acc, train_auc, train_auprc, _  = epoch_run_encoder(encoder=encoder, classifier=classifier, dataloader=train_loader, train=True, lr=lr)
            test_loss, test_acc, test_auc, test_auprc, _  = epoch_run_encoder(encoder=encoder, classifier=classifier, dataloader=valid_loader, train=False)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        if test_loss<best_loss:
            best_auc = test_auc
            best_acc = test_acc
            best_loss = test_loss
            best_aupc = test_auprc
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

    # Save performance plots
    plt.figure()
    plt.plot(np.arange(n_epochs), train_losses, label="train Loss")
    plt.plot(np.arange(n_epochs), test_losses, label="test Loss")

    plt.plot(np.arange(n_epochs), train_accs, label="train Acc")
    plt.plot(np.arange(n_epochs), test_accs, label="test Acc")
    plt.savefig(os.path.join("./plots/%s" % data_type, "classification_%s_%d.pdf"%(type, cv)))
    return best_acc, best_auc, best_aupc


def run_test(data, e2e_lr, tnc_lr, cpc_lr, trip_lr, data_path, window_size, n_cross_val):
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
    e2e_accs, e2e_aucs, e2e_auprcs = [], [], []
    tnc_accs, tnc_aucs, tnc_auprcs = [], [], []
    cpc_accs, cpc_aucs, cpc_auprcs = [], [], []
    trip_accs, trip_aucs, trip_auprcs = [], [], []
    for cv in range(n_cross_val):
        shuffled_inds = list(range(len(x_window)))
        random.shuffle(shuffled_inds)
        x_window = x_window[shuffled_inds]
        y_window = y_window[shuffled_inds]
        n_train = int(0.7*len(x_window))
        X_train, X_test = x_window[:n_train], x_window[n_train:]
        y_train, y_test = y_window[:n_train], y_window[n_train:]

        trainset = torch.utils.data.TensorDataset(X_train, y_train)
        validset = torch.utils.data.TensorDataset(X_test, y_test)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=200, shuffle=False)
        valid_loader = torch.utils.data.DataLoader(validset, batch_size=200, shuffle=False)

        # Define baseline models
        if data == 'waveform':
            encoding_size = 64
            n_classes = 4

            e2e_model = WFEncoder(encoding_size=encoding_size, classify=True, n_classes=n_classes).to(device)

            tnc_encoder = WFEncoder(encoding_size=encoding_size).to(device)
            if not os.path.exists('./ckpt/waveform/checkpoint_%d.pth.tar'%cv):
                RuntimeError('Checkpoint for TNC encoder does not exist!')
            # tnc_checkpoint = torch.load('./ckpt/waveform/checkpoint_%d.pth.tar'%cv)
            tnc_checkpoint = torch.load('/scratch/gobi1/sana/TNC_results/ckpt_paper/waveform/checkpoint_%d.pth.tar' % cv)
            tnc_encoder.load_state_dict(tnc_checkpoint['encoder_state_dict'])
            tnc_classifier = WFClassifier(encoding_size=encoding_size, output_size=4)
            tnc_model = torch.nn.Sequential(tnc_encoder, tnc_classifier).to(device)

            cpc_encoder = WFEncoder(encoding_size=encoding_size).to(device)
            if not os.path.exists('./ckpt/waveform_cpc/checkpoint_%d.pth.tar'%cv):
                RuntimeError('Checkpoint for CPC encoder does not exist!')
            # cpc_checkpoint = torch.load('./ckpt/waveform_cpc/checkpoint_%d.pth.tar'%cv)
            cpc_checkpoint = torch.load('/scratch/gobi1/sana/TNC_results/ckpt_paper/waveform_cpc/checkpoint_%d.pth.tar' % cv)
            cpc_encoder.load_state_dict(cpc_checkpoint['encoder_state_dict'])
            cpc_classifier = WFClassifier(encoding_size=encoding_size, output_size=4)
            cpc_model = torch.nn.Sequential(cpc_encoder, cpc_classifier).to(device)

            trip_encoder = WFEncoder(encoding_size=encoding_size).to(device)
            if not os.path.exists('./ckpt/waveform_trip/checkpoint_%d.pth.tar'%cv):
                RuntimeError('Checkpoint for Triplet Loss encoder does not exist!')
            # trip_checkpoint = torch.load('./ckpt/waveform_trip/checkpoint_%d.pth.tar'%cv)
            trip_checkpoint = torch.load('/scratch/gobi1/sana/TNC_results/ckpt_paper/waveform_trip/checkpoint_%d.pth.tar' % cv)
            trip_encoder.load_state_dict(trip_checkpoint['encoder_state_dict'])
            trip_classifier = WFClassifier(encoding_size=encoding_size, output_size=4)
            trip_model = torch.nn.Sequential(trip_encoder, trip_classifier).to(device)
            n_epochs = 8
            n_epoch_e2e = 8

        elif data == 'simulation':
            encoding_size = 10
            e2e_model = E2EStateClassifier(hidden_size=100, in_channel=3, encoding_size=encoding_size,
                                           output_size=4, device=device)

            tnc_encoder = RnnEncoder(hidden_size=100, in_channel=3, encoding_size=encoding_size, device=device)
            tnc_checkpoint = torch.load('./ckpt/simulation/checkpoint_%d.pth.tar'%cv)
            # tnc_checkpoint = torch.load('/scratch/gobi1/sana/TNC_results/ckpt_paper/simulation/checkpoint_%d.pth.tar' % cv)
            tnc_encoder.load_state_dict(tnc_checkpoint['encoder_state_dict'])
            tnc_classifier = StateClassifier(input_size=encoding_size, output_size=4).to(device)
            tnc_model = torch.nn.Sequential(tnc_encoder, tnc_classifier).to(device)

            cpc_encoder = RnnEncoder(hidden_size=100, in_channel=3, encoding_size=encoding_size, device=device)
            cpc_checkpoint = torch.load('./ckpt/simulation_cpc/checkpoint_%d.pth.tar'%cv)
            # cpc_checkpoint = torch.load('/scratch/gobi1/sana/TNC_results/ckpt_paper/simulation_cpc/checkpoint_%d.pth.tar' % cv)
            cpc_encoder.load_state_dict(cpc_checkpoint['encoder_state_dict'])
            cpc_classifier = StateClassifier(input_size=encoding_size, output_size=4).to(device)
            cpc_model = torch.nn.Sequential(cpc_encoder, cpc_classifier).to(device)

            trip_encoder = RnnEncoder(hidden_size=100, in_channel=3, encoding_size=encoding_size, device=device)
            trip_checkpoint = torch.load('./ckpt/simulation_trip/checkpoint_%d.pth.tar'%cv)
            # trip_checkpoint = torch.load('/scratch/gobi1/sana/TNC_results/ckpt_paper/simulation_trip/checkpoint_%d.pth.tar'%cv)
            trip_encoder.load_state_dict(trip_checkpoint['encoder_state_dict'])
            trip_classifier = StateClassifier(input_size=encoding_size, output_size=4).to(device)
            trip_model = torch.nn.Sequential(trip_encoder, trip_classifier).to(device)
            n_epochs = 30
            n_epoch_e2e = 100

        elif data == 'har':
            encoding_size = 10
            e2e_model = E2EStateClassifier(hidden_size=100, in_channel=561, encoding_size=encoding_size,
                                           output_size=6, device=device)

            tnc_encoder = RnnEncoder(hidden_size=100, in_channel=561, encoding_size=encoding_size, device=device)
            # tnc_checkpoint = torch.load('./ckpt/har/checkpoint_%d.pth.tar'%cv)
            tnc_checkpoint = torch.load('/scratch/gobi1/sana/TNC_results/ckpt_paper/har/checkpoint_%d.pth.tar' % cv)
            tnc_encoder.load_state_dict(tnc_checkpoint['encoder_state_dict'])
            tnc_classifier = StateClassifier(input_size=encoding_size, output_size=6).to(device)
            tnc_model = torch.nn.Sequential(tnc_encoder, tnc_classifier).to(device)

            cpc_encoder = RnnEncoder(hidden_size=100, in_channel=561, encoding_size=encoding_size, device=device)
            # cpc_checkpoint = torch.load('./ckpt/har_cpc/checkpoint_%d.pth.tar'%cv)
            cpc_checkpoint = torch.load('/scratch/gobi1/sana/TNC_results/ckpt_paper/har_cpc/checkpoint_%d.pth.tar' % cv)
            cpc_encoder.load_state_dict(cpc_checkpoint['encoder_state_dict'])
            cpc_classifier = StateClassifier(input_size=encoding_size, output_size=6).to(device)
            cpc_model = torch.nn.Sequential(cpc_encoder, cpc_classifier).to(device)

            trip_encoder = RnnEncoder(hidden_size=100, in_channel=561, encoding_size=encoding_size, device=device)
            # trip_checkpoint = torch.load('./ckpt/har_trip/checkpoint_%d.pth.tar'%cv)
            trip_checkpoint = torch.load('/scratch/gobi1/sana/TNC_results/ckpt_paper/har_trip/checkpoint_%d.pth.tar' % cv)
            trip_encoder.load_state_dict(trip_checkpoint['encoder_state_dict'])
            trip_classifier = StateClassifier(input_size=encoding_size, output_size=6).to(device)
            trip_model = torch.nn.Sequential(trip_encoder, trip_classifier).to(device)
            n_epochs = 50
            n_epoch_e2e = 100

        # Train the model
        # ***** E2E *****
        best_acc_e2e, best_auc_e2e, best_auprc_e2e = train(train_loader, valid_loader, e2e_model, e2e_lr,
                             data_type=data, n_epochs=n_epoch_e2e, type='e2e', cv=cv)
        print('E2E: ', best_acc_e2e*100, best_auc_e2e, best_auprc_e2e)
        # ***** TNC *****
        best_acc_tnc, best_auc_tnc, best_auprc_tnc = train(train_loader, valid_loader, tnc_classifier, tnc_lr,
                                           encoder=tnc_encoder, data_type=data, n_epochs=n_epochs, type='tnc', cv=cv)
        print('TNC: ', best_acc_tnc*100, best_auc_tnc, best_auprc_tnc)
        # ***** CPC *****
        best_acc_cpc, best_auc_cpc, best_auprc_cpc = train(train_loader, valid_loader, cpc_classifier, cpc_lr,
                                           encoder=cpc_encoder, data_type=data, n_epochs=n_epochs, type='cpc', cv=cv)
        print('CPC: ', best_acc_cpc*100, best_auc_cpc, best_auprc_cpc)
        # ***** Trip *****
        best_acc_trip, best_auc_trip, best_auprc_trip = train(train_loader, valid_loader, trip_classifier, trip_lr,
                                             encoder=trip_encoder, data_type=data, n_epochs=n_epochs, type='trip', cv=cv)
        print('TRIP: ', best_acc_trip*100, best_auc_trip, best_auprc_trip)

        print('TESTING!!!!!')
        # Validate the model
        # ***** E2E *****
        _, best_acc_e2e, best_auc_e2e, best_auprc_e2e, _ = epoch_run(e2e_model, dataloader=test_loader, train=False)
        print('E2E: ', best_acc_e2e * 100, best_auc_e2e, best_auprc_e2e)
        # ***** TNC *****
        _, best_acc_tnc, best_auc_tnc, best_auprc_tnc, _ = epoch_run_encoder(tnc_encoder, tnc_classifier, dataloader=test_loader, train=False)
        print('TNC: ', best_acc_tnc * 100, best_auc_tnc, best_auprc_tnc)
        # ***** CPC *****
        _, best_acc_cpc, best_auc_cpc, best_auprc_cpc, _ = epoch_run_encoder(cpc_encoder, cpc_classifier, dataloader=test_loader, train=False)
        print('CPC: ', best_acc_cpc * 100, best_auc_cpc, best_auprc_cpc)
        # ***** Trip *****
        _, best_acc_trip, best_auc_trip, best_auprc_trip, _ = epoch_run_encoder(trip_encoder, trip_classifier, dataloader=test_loader, train=False)
        print('TRIP: ', best_acc_trip * 100, best_auc_trip, best_auprc_trip)

        e2e_accs.append(best_acc_e2e)
        e2e_aucs.append(best_auc_e2e)
        e2e_auprcs.append(best_auprc_e2e)
        tnc_accs.append(best_acc_tnc)
        tnc_aucs.append(best_auc_tnc)
        tnc_auprcs.append(best_auprc_tnc)
        cpc_accs.append(best_acc_cpc)
        cpc_aucs.append(best_auc_cpc)
        cpc_auprcs.append(best_auprc_cpc)
        trip_accs.append(best_acc_trip)
        trip_aucs.append(best_auc_trip)
        trip_auprcs.append(best_auprc_trip)

        with open("./outputs/%s_classifiers.txt"%data, "a") as f:
            f.write("\n\nPerformance result for a fold" )
            f.write("End-to-End model: \t AUC: %s\t Accuracy: %s \n\n" % (str(best_auc_e2e), str(100*best_acc_e2e)))
            f.write("TNC model: \t AUC: %s\t Accuracy: %s \n\n" % (str(best_auc_tnc), str(100*best_acc_tnc)))
            f.write("CPC model: \t AUC: %s\t Accuracy: %s \n\n" % (str(best_auc_cpc), str(100*best_acc_cpc)))
            f.write("Triplet Loss model: \t AUC: %s\t Accuracy: %s \n\n" % (str(best_auc_trip), str(100*best_acc_trip)))

        torch.cuda.empty_cache()

    print('=======> Performance Summary:')
    print('E2E model: \t Accuracy: %.2f +- %.2f \t AUC: %.3f +- %.3f \t AUPRC: %.3f +- %.3f'%
          (100 * np.mean(e2e_accs), 100 * np.std(e2e_accs), np.mean(e2e_aucs), np.std(e2e_aucs),
           np.mean(e2e_auprcs), np.std(e2e_auprcs)))
    print('TNC model: \t Accuracy: %.2f +- %.2f \t AUC: %.3f +- %.3f \t AUPRC: %.3f +- %.3f'%
          (100 * np.mean(tnc_accs), 100 * np.std(tnc_accs), np.mean(tnc_aucs), np.std(tnc_aucs),
           np.mean(tnc_auprcs), np.std(tnc_auprcs)))
    print('CPC model: \t Accuracy: %.2f +- %.2f \t AUC: %.3f +- %.3f \t AUPRC: %.3f +- %.3f'%
          (100 * np.mean(cpc_accs), 100 * np.std(cpc_accs), np.mean(cpc_aucs), np.std(cpc_aucs),
           np.mean(cpc_auprcs), np.std(cpc_auprcs)))
    print('Trip model: \t Accuracy: %.2f +- %.2f \t AUC: %.3f +- %.3f \t AUPRC: %.3f +- %.3f'%
          (100 * np.mean(trip_accs), 100 * np.std(trip_accs), np.mean(trip_aucs), np.std(trip_aucs),
           np.mean(trip_auprcs), np.std(trip_auprcs)))


if __name__=='__main__':
    random.seed(1234)
    parser = argparse.ArgumentParser(description='Run classification test')
    parser.add_argument('--data', type=str, default='simulation')
    parser.add_argument('--cv', type=int, default=1)
    args = parser.parse_args()

    if not os.path.exists('./ckpt/classifier_test'):
        os.mkdir('./ckpt/classifier_test')

    f = open("./outputs/%s_classifiers.txt"%args.data, "w")
    f.close()
    if args.data=='simulation':
        run_test(data='simulation', e2e_lr=0.01, tnc_lr=0.01, cpc_lr=0.1, trip_lr=0.1,
                 data_path='./data/simulated_data/', window_size=50, n_cross_val=args.cv)
    elif args.data=='waveform':
        run_test(data='waveform', e2e_lr=0.0001, tnc_lr=0.01, cpc_lr=0.01, trip_lr=0.01,
                 data_path='./data/waveform_data/processed', window_size=2500, n_cross_val=args.cv)
    elif args.data=='har':
        run_test(data='har', e2e_lr=0.001, tnc_lr=0.1, cpc_lr=0.1, trip_lr=0.1,
                 data_path='./data/HAR_data/', window_size=4, n_cross_val=args.cv)
