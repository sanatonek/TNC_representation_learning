import numpy as np
import random
import argparse

import os
import pickle

from tslearn.neighbors import KNeighborsTimeSeries
from tslearn.clustering import TimeSeriesKMeans
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import silhouette_score, davies_bouldin_score


def main(args):
    if args.data=='simulation':
        window_size = 50
        path = './data/simulated_data/'
        with open(os.path.join(path, 'x_train.pkl'), 'rb') as f:
            x = pickle.load(f)
        with open(os.path.join(path, 'state_train.pkl'), 'rb') as f:
            y = pickle.load(f)
        T = x.shape[-1]
        t = np.random.randint(window_size,  T- window_size, len(x)*5)
        x_window = np.array( [x[i//5, :, tt - window_size//2: tt + window_size//2] for i, tt in enumerate(t)] )
        y_window = np.round(np.mean( np.array([y[i//5, tt - window_size//2: tt + window_size//2] for i, tt in enumerate(t)]), -1 ))
        x_window = x_window.transpose((0,2,1)) # shape:[n_samples, t_len, d]

        accuracy, s_score, db_score, auc = [], [], [], []
        for cv in range(3):
            shuffled_inds = list(range(len(x_window)))
            random.shuffle(shuffled_inds)
            x_window = x_window[shuffled_inds]
            y_window = y_window[shuffled_inds]
            n_train = int(0.8 * len(x_window))
            x_train, x_test = x_window[:n_train], x_window[n_train:]
            y_train, y_test = y_window[:n_train], y_window[n_train:]


            knn = KNeighborsTimeSeries(n_neighbors=args.K, metric='dtw').fit(x_train)
            kmeans = TimeSeriesKMeans(n_clusters=4, metric='dtw')
            cluster_labels = kmeans.fit_predict(x_test)

            dist, ind = knn.kneighbors(x_test, return_distance=True)
            predictions = np.array([y_train[np.bincount(preds).argmax()] for preds in ind])
            y_onehot = np.zeros((len(y_test), 4))
            y_onehot[np.arange(len(y_onehot)), y_test.astype(int)] = 1
            prediction_onehot = np.zeros((len(y_test), 4))
            prediction_onehot[np.arange(len(prediction_onehot)), predictions.astype(int)] = 1

            accuracy.append(accuracy_score(y_test, predictions))
            auc.append(roc_auc_score(y_onehot, prediction_onehot))
            s_score.append(silhouette_score(x_test.reshape((len(x_test), -1)), cluster_labels))
            db_score.append(davies_bouldin_score(x_test.reshape((len(x_test), -1)), cluster_labels))

    print('Test accuracy: ', np.mean(accuracy)*100, '+-', np.std(accuracy)*100)
    print('Silhouette score: ', np.mean(s_score), '+-', np.std(s_score))
    print('Davies Bouldin score: ', np.mean(db_score), '+-', np.std(db_score))


if __name__=='__main__':
    random.seed(1234)
    parser = argparse.ArgumentParser(description='Run DTW')
    parser.add_argument('--data', type=str, default='simulation')
    parser.add_argument('--K', type=int, default=10)
    args = parser.parse_args()
    main(args)