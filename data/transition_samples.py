"""
Create a simulation sample that gradually transitions from on hidden state to another
The transtion is from one narma model to another and the parameters gradually change to model a smooth transition.
"""

import numpy as np
import TimeSynth.timesynth as ts
import matplotlib.pyplot as plt
import pickle
import os


coefficients_s1 = [0.3, 0.05, 1.5, 0.1]
coefficients_s2 = [0.1, 0.25, 2.5, -0.005]
sig_1, sig_2, sig_3 = [], [], []
window_size = 50

for i in range(20):
    time_sampler = ts.TimeSampler(stop_time=window_size)
    sampler = time_sampler.sample_regular_time(num_points=window_size)
    white_noise = ts.noise.GaussianNoise(std=0.3)
    if len(sig_1)>=0:
        sig_type = ts.signals.NARMA(order=5, coefficients=coefficients_s1, initial_condition=[0.671, 0.682, 0.675, 0.687, 0.69])
    else:
        sig_type = ts.signals.NARMA(order=5, coefficients=coefficients_s1, initial_condition=sig_1[-5:])
    timeseries = ts.TimeSeries(sig_type, noise_generator=white_noise)
    samples, _, _ = timeseries.sample(sampler)
    sig_1.extend(samples)
    if i>3 and i <14:
        coefficients_s1[0] -= 0.02
        coefficients_s1[1] += 0.01
    correlated_signal = samples * 0.9 + .03 + np.random.randn(len(samples)) * 0.4
    sig_2.extend(correlated_signal)
    if len(sig_3)>=0:
        sig_type = ts.signals.NARMA(order=5, coefficients=[0.3, 0.05, 2.5, -0.005], initial_condition=[1, 0.97, 0.96])
    else:
        sig_type = ts.signals.NARMA(order=5, coefficients=coefficients_s2, initial_condition=sig_3[-5:])
    timeseries = ts.TimeSeries(sig_type, noise_generator=white_noise)
    samples, _, _ = timeseries.sample(sampler)
    sig_3.extend(samples)
    if i > 3 and i < 14:
        coefficients_s2[0] += 0.02
        coefficients_s2[1] -= 0.01


f, axes = plt.subplots(3,1)
f.set_figheight(3)
f.set_figwidth(10)
axes[0].plot(sig_1)
axes[1].plot(sig_2)
axes[2].plot(sig_3)
plt.savefig('./plots/transition.pdf')

time_series = np.stack([sig_1, sig_2, sig_3])
if not os.path.exists('./data/simulated_data'):
    os.mkdir('./data/simulated_data')
with open('./data/simulated_data/x_transition.pkl', 'wb') as f:
    pickle.dump(time_series, f)