# Unsupervised Representation Learning for TimeSeries with Temporal Neighborhood Coding

### To train the TNC encoder model, simply run:
```
python -m tnc.tnc --data <YOUR DATASET>
```
We have used 2 datasets for our experiments. You can create the simulated dataset using the following script:
```
python data/simulated_data.py
```
For the waveform dataset, you need to download the raw recordings from Physionet website. The module  data/afib_data.py will preprocess the data and annotations for you.
