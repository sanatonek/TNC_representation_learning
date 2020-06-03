import numpy as np
import TimeSynth.timesynth as ts
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

import os
import pickle

n_signals = 5
n_states = 4
transition_matrix = np.eye(n_states)*0.85
transition_matrix[0,1] = transition_matrix[1,0] = 0.05
transition_matrix[0,2] = transition_matrix[2,0] = 0.05
transition_matrix[0,3] = transition_matrix[3,0] = 0.05
transition_matrix[2,3] = transition_matrix[3,2] = 0.05
transition_matrix[2,1] = transition_matrix[1,2] = 0.05
transition_matrix[3,1] = transition_matrix[1,3] = 0.05

# transition_matrix = np.eye(n_states)*0.7
# transition_matrix[0,1] = transition_matrix[1,0] = 0.1
# transition_matrix[0,2] = transition_matrix[2,0] = 0.1
# transition_matrix[0,3] = transition_matrix[3,0] = 0.1
# transition_matrix[2,3] = transition_matrix[3,2] = 0.1
# transition_matrix[2,1] = transition_matrix[1,2] = 0.1
# transition_matrix[3,1] = transition_matrix[1,3] = 0.1

def main(n_samples, sig_len):
    all_signals = []
    all_states = []
    for _ in range(n_samples):
        sample_signal, sample_state = create_signal(sig_len)
        all_signals.append(sample_signal)
        all_states.append(sample_state)

    dataset = np.array(all_signals)
    states = np.array(all_states)
    n_train = int(len(dataset) * 0.8)
    train_data = dataset[:n_train]
    test_data = dataset[n_train:]
    train_data_n, test_data_n = normalize(train_data, test_data)
    train_state = states[:n_train]
    test_state = states[n_train:]

    print("Dataset Shape ====> \tTrainset: ", train_data_n.shape, "\tTestset: ", test_data_n.shape)
    # f, axes = plt.subplots(3,1)
    # f.set_figheight(3)
    # f.set_figwidth(10)
    # # print(train_state[0])
    # color = [[0.6350, 0.0780, 0.1840], [0.4660, 0.6740, 0.1880], [0, 0.4470, 0.7410]]
    # for i,ax in enumerate(axes):
    #     ax.plot(train_data_n[0,i,:],  c=color[i])
    #     for t in range(train_data_n[0,i,:].shape[-1]):
    #         ax.axvspan(t, min(t+1, train_state.shape[-1]-1), facecolor=['y', 'g', 'b', 'r'][train_state[0,t]], alpha=0.3)
    # f.set_figheight(6)
    # f.set_figwidth(12)
    # plt.savefig('./simulation_sample.pdf')

    # Save signals to file
    if not os.path.exists('./data/simulated_data'):
        os.mkdir('./data/simulated_data')
    with open('./data/simulated_data/x_train.pkl', 'wb') as f:
        pickle.dump(train_data_n, f)
    with open('./data/simulated_data/x_test.pkl', 'wb') as f:
        pickle.dump(test_data_n, f)
    with open('./data/simulated_data/state_train.pkl', 'wb') as f:
        pickle.dump(train_state, f)
    with open('./data/simulated_data/state_test.pkl', 'wb') as f:
        pickle.dump(test_state, f)


def create_signal(sig_len, window_size=50):
    states = []
    sig_1 = []
    sig_2 = []
    sig_3 = []
    pi = np.ones((1,n_states))/n_states

    for _ in range(sig_len//window_size):
        current_state = np.random.choice(n_states, 1, p=pi.reshape(-1))
        states.extend(list(current_state)*window_size)

        current_signal = ts_generator(current_state[0], window_size)
        sig_1.extend(current_signal)
        correlated_signal = current_signal*0.9 + .03 + np.random.randn(len(current_signal))*0.4
        sig_2.extend(correlated_signal)
        uncorrelated_signal = ts_generator((current_state[0]+2)%4, window_size)
        sig_3.extend(uncorrelated_signal)

        pi = transition_matrix[current_state]
    signals = np.stack([sig_1, sig_2, sig_3])
    # print(states)
    # f, axes = plt.subplots(3,1)
    # for i,ax in enumerate(axes):
    #     ax.plot(signals[i])
    # plt.show()
    return signals, states


def ts_generator(state, window_size):
    time_sampler = ts.TimeSampler(stop_time=window_size)
    sampler = time_sampler.sample_regular_time(num_points=window_size)
    white_noise = ts.noise.GaussianNoise(std=0.3)
    if state == 0:
        sig_type = ts.signals.GaussianProcess(kernel="Periodic", lengthscale=1., mean=0., variance=.1, p=5) # Scale is too big
    elif state == 1:
        sig_type = ts.signals.NARMA(order=5, initial_condition=[0.671, 0.682, 0.675, 0.687, 0.69])
    elif state == 2:
        sig_type = ts.signals.GaussianProcess(kernel="SE", lengthscale=1., mean=0., variance=.1)
    elif state == 3:
        sig_type = ts.signals.NARMA(order=3, coefficients=[0.1, 0.25, 2.5, -0.005], initial_condition=[1, 0.97, 0.96])

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
        # feature_means = np.tile(np.mean(d.reshape(-1, feature_size), axis=0), (sig_len, 1)).T
        feature_means = np.mean(train_data, axis=(0,2))
        # feature_std = np.tile(np.std(d.reshape(-1, feature_size), axis=0), (sig_len, 1)).T
        feature_std = np.std(train_data, axis=(0, 2))
        np.seterr(divide='ignore', invalid='ignore')
        train_data_n = train_data - feature_means[np.newaxis,:,np.newaxis]/\
                       np.where(feature_std == 0, 1, feature_std)[np.newaxis,:,np.newaxis]
        test_data_n = test_data - feature_means[np.newaxis, :, np.newaxis] / \
                       np.where(feature_std == 0, 1, feature_std)[np.newaxis, :, np.newaxis]
        # train_data_n = np.array(
        #     [np.where(feature_std == 0, (x - feature_means), (x - feature_means) / feature_std) for
        #      x in train_data])
        # test_data_n = np.array(
        #     [np.where(feature_std == 0, (x - feature_means), (x - feature_means) / feature_std) for
        #      x in test_data])
    elif config == 'zero_to_one':
        feature_max = np.tile(np.max(d.reshape(-1, feature_size), axis=0), (sig_len, 1)).T
        feature_min = np.tile(np.min(d.reshape(-1, feature_size), axis=0), (sig_len, 1)).T
        train_data_n = np.array([(x - feature_min) / (feature_max - feature_min) for x in train_data])
        test_data_n = np.array([(x - feature_min) / (feature_max - feature_min) for x in test_data])
    return train_data_n, test_data_n


if __name__ == '__main__':
    main(n_samples=500, sig_len=800)
