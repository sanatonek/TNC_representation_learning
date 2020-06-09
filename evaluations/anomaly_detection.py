import torch
import os
import pickle
import numpy as np
import random

from tnc.models import WFEncoder
from sklearn.metrics import roc_auc_score, average_precision_score
from baselines.dtw import DTWDistance, cluster

device = 'cuda' if torch.cuda.is_available() else 'cpu'

encoder = WFEncoder(encoding_size=64)
tcl_checkpoint = torch.load('./ckpt/waveform/checkpoint_0.pth.tar')
# tcl_checkpoint = torch.load('./ckpt/waveform_trip/checkpoint.pth.tar')
encoder.load_state_dict(tcl_checkpoint['encoder_state_dict'])
encoder.eval()
encoder.to(device)

window_size = 2500
path = './data/waveform_data/processed'


with open(os.path.join(path, 'x_test.pkl'), 'rb') as f:
    x_test = pickle.load(f)
with open(os.path.join(path, 'state_test.pkl'), 'rb') as f:
    y_test = pickle.load(f)

# with open(os.path.join(path, 'x_train.pkl'), 'rb') as f:
#     x_train = pickle.load(f)
# with open(os.path.join(path, 'state_train.pkl'), 'rb') as f:
#     y_train = pickle.load(f)

T = x_test.shape[-1]
x_window = np.split(x_test[:, :, :window_size * (T // window_size)], (T // window_size), -1)
x_window = np.concatenate(x_window, 0)
y_window = np.concatenate(np.split(y_test[:, :window_size * (T // window_size)], (T // window_size), -1),0).astype(int)
y_window = np.array([np.bincount(yy).argmax() for yy in y_window])

T = x_test.shape[-1]
# x_window_train = np.split(x_train[:, :, :window_size * (T // window_size)], (T // window_size), -1)
# x_window_train = np.concatenate(x_window_train, 0)
# y_window_train = np.concatenate(np.split(y_train[:, :window_size * (T // window_size)], (T // window_size), -1),0).astype(int)
# y_window_train = np.array([np.bincount(yy).argmax() for yy in y_window_train])
# shuffled_inds_train = list(range(len(x_window_train)))
# random.shuffle(shuffled_inds_train)
shuffled_inds_test = list(range(len(x_window)))
random.shuffle(shuffled_inds_test)

# print(x_window.shape, y_window.shape)
testset = torch.utils.data.TensorDataset(torch.Tensor(x_window), torch.Tensor(y_window))
test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=True)
# trainset = torch.utils.data.TensorDataset(torch.Tensor(x_window_train), torch.Tensor(y_window_train))
# train_loader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True)

is_anomaly = y_window.copy()
is_anomaly = np.logical_or(is_anomaly==1, is_anomaly==2).astype(int)

encodings_train = []
for x,_ in test_loader:
    encodings_train.append(encoder(x.to(device)).detach().cpu().numpy())
encodings_train = np.concatenate(encodings_train, 0)
print(encodings_train.shape)

# train the KNN detector
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.cblof import CBLOF
from pyod.models.auto_encoder import AutoEncoder

clf_knn = KNN()
clf_lof = LOF()
clf_cblof = CBLOF()
clf_ae = AutoEncoder(epochs=50)
clf_knn.fit(encodings_train)
clf_lof.fit(encodings_train)
clf_cblof.fit(encodings_train)
clf_ae.fit(encodings_train)

anomaly_scores_knn = clf_knn.predict_proba(encodings_train)
anomaly_scores_lof = clf_lof.predict_proba(encodings_train)
anomaly_scores_cblof = clf_cblof.predict_proba(encodings_train)
anomaly_scores_ae = clf_ae.predict_proba(encodings_train)


# y_test_scores = []
# for x,_ in test_loader:
#     encodings_test = encoder(torch.Tensor(x).to(device))
#     probs = clf.predict_proba(encodings_test.detach().cpu().numpy())
#     y_test_scores.extend(probs[:,0])
# y_test_scores = np.array(y_test_scores)

y_ind_1 = np.argwhere(y_window.reshape(-1, ) == 1)
y_ind_3 = np.argwhere(y_window.reshape(-1, ) == 3)

for i, anomaly_scores in enumerate([anomaly_scores_knn, anomaly_scores_lof, anomaly_scores_cblof, anomaly_scores_ae]):
    method = ['KNN', 'LOF', 'CBLOF', 'AE'][i]
    print('********** Results for ', method)
    auc = roc_auc_score(is_anomaly, anomaly_scores[:,0])
    auprc = average_precision_score(is_anomaly, anomaly_scores[:,0])
    print('Anomaly detection AUC: ', auc)
    print('Anomaly detection AUPRC: ', auprc)
    print('Label 1: ', np.mean(anomaly_scores[y_ind_1.reshape(-1,)]), '+-',
          np.std(anomaly_scores[y_ind_1.reshape(-1,)]))
    print('Label 3: ', np.mean(anomaly_scores[y_ind_3.reshape(-1,)]), '+-',
          np.std(anomaly_scores[y_ind_3.reshape(-1,)]))


# train_data = np.transpose(x_window, (0,2,1))

# Calculate distances using DTW
# dtw_distances = np.zeros((100,100))
# w = 2500
# for ind, i in enumerate(train_data[shuffled_inds[:100]]):
#     for c_ind, j in enumerate(train_data[shuffled_inds[:100]]):
#         cur_dist = 0.0
#         # Find sum of distances along each dimension
#         for z in range(np.shape(train_data)[2]):
#             cur_dist += DTWDistance(i[:, z], j[:, z], w)
#         print(cur_dist)
#         dtw_distances[ind, c_ind] = cur_dist
# clusters, curr_medoids = cluster(dtw_distances, 4)
#
# encoding_distances = torch.matmul(encodings, encodings.permute(1,0)).detach().cpu().numpy()
# encoding_distances = np.min(abs(encoding_distances), -1)
#
# # print('Distance distribution in DTW: ', np.mean(dtw_distances), ' += ', np.std(dtw_distances))
# print('Distance distribution in encoding space: ', np.mean(encoding_distances).item(), ' += ', np.std(encoding_distances).item())
#
# y_ind_1 = np.argwhere(y_window.reshape(-1,)==1)
# print(y_ind_1)