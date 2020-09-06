'''
Preprocess the HAR data to concatenate individual measurements
'''

import pandas as pd
import numpy as np
import pickle

trainX = pd.read_csv('./data/HAR_data/train/X_train.txt', delim_whitespace=True,header=None)
trainy = pd.read_csv('./data/HAR_data/train/y_train.txt',delim_whitespace=True,header=None)
train_subj = pd.read_csv('./data/HAR_data/train/subject_train.txt', delim_whitespace=True,header=None)
testX = pd.read_csv('./data/HAR_data/test/X_test.txt',delim_whitespace=True,header=None)
testy = pd.read_csv('./data/HAR_data/test/y_test.txt',delim_whitespace=True,header=None)
test_subj = pd.read_csv('./data/HAR_data/test/subject_test.txt', delim_whitespace=True,header=None)

train_ids = np.unique(train_subj)
x_train, y_train = [], []
lens = []
for i, ids in enumerate(train_ids):
    inds = np.where(train_subj == ids)[0]
    ts = np.take(trainX, inds, 0).to_numpy()
    ts_labels = np.repeat(np.take(trainy, inds, 0).to_numpy(), trainX.shape[1])
    ts = ts.reshape(-1, )
    lens.append(len(ts))
    x_train.append(ts)
    y_train.append(ts_labels.reshape(-1, ))

x_train = np.stack([x[:min(lens)] for x in x_train])
y_train = np.stack([y[:min(lens)] for y in y_train])

test_ids = np.unique(test_subj)
x_test, y_test = [], []
lens = []
for i, ids in enumerate(test_ids):
    inds = np.where(test_subj == ids)[0]
    ts = np.take(testX, inds, 0).to_numpy()
    ts_labels = np.repeat(np.take(testy, inds, 0).to_numpy(), testX.shape[1])
    ts = ts.reshape(-1, )
    lens.append(len(ts))
    x_test.append(ts)
    y_test.append(ts_labels.reshape(-1, ))

x_test = np.stack([x[:min(lens)] for x in x_test])
y_test = np.stack([y[:min(lens)] for y in y_test])

## Save signals to file
with open('./data/HAR_data/x_train.pkl', 'wb') as f:
    pickle.dump(x_train, f)
with open('./data/HAR_data/x_test.pkl', 'wb') as f:
    pickle.dump(x_test, f)
with open('./data/HAR_data/y_train.pkl', 'wb') as f:
    pickle.dump(y_train, f)
with open('./data/HAR_data/y_test.pkl', 'wb') as f:
    pickle.dump(y_test, f)