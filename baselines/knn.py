import numpy as np
import random
import argparse

import os
import pickle

from tslearn.neighbors import KNeighborsTimeSeries
from tslearn.clustering import TimeSeriesKMeans
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from sklearn.metrics import silhouette_score, davies_bouldin_score


def main(args):
    if args.data=='simulation':
        window_size = 50
        path = './data/simulated_data/'
        n_cluster = 4
        augment = 5
    if args.data=='wf':
        window_size = 2500
        path = './data/waveform_data/processed'
        n_cluster = 4
        augment = 500
    if args.data=='har':
        window_size = 5
        path = './data/HAR_data/'
        n_cluster = 6
        augment = 100

    with open(os.path.join(path, 'x_train.pkl'), 'rb') as f:
        x = pickle.load(f)
    with open(os.path.join(path, 'state_train.pkl'), 'rb') as f:
        y = pickle.load(f)
    with open(os.path.join(path, 'x_test.pkl'), 'rb') as f:
        x_test = pickle.load(f)
    with open(os.path.join(path, 'state_test.pkl'), 'rb') as f:
        y_test = pickle.load(f)

    T = x.shape[-1]
    t = np.random.randint(window_size,  T- window_size, len(x)*augment)
    x_window = np.array( [x[i//augment, :, tt - window_size//2: tt + window_size//2] for i, tt in enumerate(t)] )
    y_window = np.round(np.mean( np.array([y[i//augment, tt - window_size//2: tt + window_size//2] for i, tt in enumerate(t)]), -1 ))
    if args.data == 'wf':
        minority_index = np.logical_or(y_window==1, y_window==2)
        rand_index = np.random.randint(0, len(y_window), 200)
        y_window = np.concatenate([y_window[minority_index], y_window[rand_index]], 0)
        x_window = np.concatenate([x_window[minority_index], x_window[rand_index]], 0)
        x_window = x_window.transpose((0,2,1)) # shape:[n_samples, t_len, d]
        x_window = x_window[:, ::2, :]  # Decimate measurements for efficiency
    else:
        x_window = x_window.transpose((0,2,1)) # shape:[n_samples, t_len, d]

    t = np.random.randint(window_size,  T- window_size, len(x_test)*augment)
    x_test_window = np.array( [x_test[i//augment, :, tt - window_size//2: tt + window_size//2] for i, tt in enumerate(t)] )
    y_test_window = np.round(np.mean( np.array([y_test[i//augment, tt - window_size//2: tt + window_size//2] for i, tt in enumerate(t)]), -1 ))
    if 0:#args.data =='wf':
        minority_index = np.logical_or(y_test_window==1, y_test_window==2)
        rand_index = np.random.randint(0, len(y_test_window), 150)
        y_test = np.concatenate([y_test_window[minority_index], y_test_window[rand_index]], 0)
        x_test = np.concatenate([x_test_window[minority_index], x_test_window[rand_index]], 0)
        x_test_window = x_test.transpose((0,2,1)) # shape:[n_samples, t_len, d]
        x_test = x_test_window[:, ::2, :]  # Decimate measurements for efficiency
    else:
        y_test = y_test_window
        x_test = x_test_window
        x_test = x_test.transpose((0,2,1)) # shape:[n_samples, t_len, d]

    accuracy, s_score, db_score, auc, auprc = [], [], [], [], []
    for cv in range(3):
        shuffled_inds = list(range(len(x_window)))
        random.shuffle(shuffled_inds)
        x_window = x_window[shuffled_inds]
        y_window = y_window[shuffled_inds]
        if args.data=='wf':
            n_train = int(0.7 * len(x_window))
            x_train = x_window[:n_train]
            y_train = y_window[:n_train]
            x_test = x_window[n_train:]
            y_test = y_window[n_train:]
        else:
            x_train = x_window
            y_train = y_window

        knn = KNeighborsTimeSeries(n_neighbors=args.K, metric='dtw').fit(x_train)
        kmeans = TimeSeriesKMeans(n_clusters=n_cluster, metric='dtw')
        cluster_labels = kmeans.fit_predict(x_test)

        dist, ind = knn.kneighbors(x_test, return_distance=True)
        predictions = np.array([y_train[np.bincount(preds).argmax()] for preds in ind])
        y_onehot = np.zeros((len(y_test), n_cluster))
        y_onehot[np.arange(len(y_onehot)), y_test.astype(int)] = 1
        prediction_onehot = np.zeros((len(y_test), n_cluster))
        prediction_onehot[np.arange(len(prediction_onehot)), predictions.astype(int)] = 1

        accuracy.append(accuracy_score(y_test, predictions))
        auc.append(roc_auc_score(y_onehot, prediction_onehot))
        auprc.append(average_precision_score(y_onehot, prediction_onehot))
        s_score.append(silhouette_score(x_test.reshape((len(x_test), -1)), cluster_labels))
        db_score.append(davies_bouldin_score(x_test.reshape((len(x_test), -1)), cluster_labels))

    print('\nSummary performance:')
    print('Accuracy: ', np.mean(accuracy)*100, '+-', np.std(accuracy)*100)
    print('AUC: ', np.mean(auc), '+-', np.std(auc))
    print('AUPRC: ', np.mean(auprc), '+-', np.std(auprc))
    print('Silhouette score: ', np.mean(s_score), '+-', np.std(s_score))
    print('Davies Bouldin score: ', np.mean(db_score), '+-', np.std(db_score))


if __name__=='__main__':
    random.seed(1234)
    parser = argparse.ArgumentParser(description='Run KNN')
    parser.add_argument('--data', type=str, default='simulation')
    parser.add_argument('--K', type=int, default=10)
    args = parser.parse_args()
    main(args)