import numpy as np
import TimeSynth.timesynth as ts
import matplotlib.pyplot as plt

import os
import pickle

n_signals = 5
n_states = 4
transition_matrix = np.eye(n_states)*0.9
transition_matrix[0,1] = transition_matrix[1,0] = 0.1
# transition_matrix[0,2] = transition_matrix[2,0] = 0.1
# transition_matrix[0,3] = transition_matrix[3,0] = 0.1
# transition_matrix[0,4] = transition_matrix[4,0] = 0.1
transition_matrix[2,3] = transition_matrix[3,2] = 0.1


def main(n_samples, n_features, sig_len):
    all_signals = []
    all_states = []
    for _ in range(n_samples):
        sample_signal, sample_state = create_signal(n_features, sig_len)
        all_signals.append(sample_signal)
        all_states.append(sample_state)
    print(np.array(all_signals).shape)

    dataset = np.array(all_signals)
    states = np.array(all_states)
    n_train = int(len(dataset) * 0.8)
    train_data = dataset[:n_train]
    test_data = dataset[n_train:]
    train_data_n, test_data_n = normalize(train_data[:, np.newaxis, :], test_data[:, np.newaxis, :])
    train_state = states[:n_train]
    test_state = states[n_train:]
    # plt.plot(train_data_n[0,0,:])
    # plt.show()

    # Save signals to file
    if not os.path.exists('./simulated_data'):
        os.mkdir('./simulated_data')
    with open('./simulated_data/x_train.pkl', 'wb') as f:
        pickle.dump(train_data_n, f)
    with open('./simulated_data/x_test.pkl', 'wb') as f:
        pickle.dump(test_data_n, f)
    with open('./simulated_data/state_train.pkl', 'wb') as f:
        pickle.dump(train_state, f)
    with open('./simulated_data/state_test.pkl', 'wb') as f:
        pickle.dump(test_state, f)


def create_signal(n_features, sig_len, window_size=20):
    states = []
    signals = []
    pi = np.ones((1,n_states))/n_states

    for _ in range(sig_len//window_size):
        current_state = np.random.choice(n_states, 1, p=pi.reshape(-1))
        current_signal = ts_generator(current_state[0], window_size)
        states.extend(list(current_state)*window_size)
        signals.extend(current_signal)
        pi = transition_matrix[current_state]
    # print(states)
    # plt.plot(signals)
    # plt.show()
    return signals, states


def ts_generator(state, window_size):
    time_sampler = ts.TimeSampler(stop_time=window_size)
    sampler = time_sampler.sample_regular_time(num_points=window_size)
    white_noise = ts.noise.GaussianNoise(std=0.2)
    if state == 0:
        sig_type = ts.signals.GaussianProcess(kernel="Periodic", lengthscale=1., mean=0., variance=1., p=5) # Scale is too big
    elif state == 1:
        sig_type = ts.signals.GaussianProcess(kernel="SE", lengthscale=1., mean=0., variance=.1)
    elif state == 2:
        #sig_type = ts.signals.GaussianProcess(kernel="Linear", c=0.01, offset=-0.5) ## TODO scale appropiately
        sig_type = ts.signals.NARMA(order=5, initial_condition=[0.671, 0.682, 0.675, 0.687, 0.69])
    elif state == 3:
        sig_type = ts.signals.NARMA(order=3, initial_condition=[1, 0.97, 0.96])

    timeseries = ts.TimeSeries(sig_type, noise_generator=white_noise)
    samples, _, _ = timeseries.sample(sampler)
    return samples


def normalize(train_data, test_data, config='mean_normalized'):
    """ Calculate the mean and std of each feature from the training set
    """
    feature_size = train_data.shape[1]
    sig_len = train_data.shape[2]
    d = [x.T for x in train_data]
    d = np.stack(d, axis=0)
    if config == 'mean_normalized':
        feature_means = np.tile(np.mean(d.reshape(-1, feature_size), axis=0), (sig_len, 1)).T
        feature_std = np.tile(np.std(d.reshape(-1, feature_size), axis=0), (sig_len, 1)).T
        np.seterr(divide='ignore', invalid='ignore')
        train_data_n = np.array(
            [np.where(feature_std == 0, (x - feature_means), (x - feature_means) / feature_std) for
             x in train_data])
        test_data_n = np.array(
            [np.where(feature_std == 0, (x - feature_means), (x - feature_means) / feature_std) for
             x in test_data])
    elif config == 'zero_to_one':
        feature_max = np.tile(np.max(d.reshape(-1, feature_size), axis=0), (sig_len, 1)).T
        feature_min = np.tile(np.min(d.reshape(-1, feature_size), axis=0), (sig_len, 1)).T
        train_data_n = np.array([(x - feature_min) / (feature_max - feature_min) for x in train_data])
        test_data_n = np.array([(x - feature_min) / (feature_max - feature_min) for x in test_data])
    return train_data_n, test_data_n


if __name__ == '__main__':
    main(n_samples=500, n_features=1, sig_len=800)
