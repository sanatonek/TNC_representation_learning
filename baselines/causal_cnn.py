
import torch
import numpy as np
import numpy
import os
import random
import pickle
import matplotlib.pyplot as plt

from tcl.models import RnnEncoder, WFEncoder
from tcl.utils import plot_distribution
from tcl.evaluations import ClassificationPerformanceExperiment, WFClassificationExperiment

class Chomp1d(torch.nn.Module):
    """
    Removes the last elements of a time series.
    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`, `L - s`) where `s`
    is the number of elements to remove.
    @param chomp_size Number of elements to remove.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]


class SqueezeChannels(torch.nn.Module):
    """
    Squeezes, in a three-dimensional tensor, the third dimension.
    """
    def __init__(self):
        super(SqueezeChannels, self).__init__()

    def forward(self, x):
        return x.squeeze(2)


class CausalConvolutionBlock(torch.nn.Module):
    """
    Causal convolution block, composed sequentially of two causal convolutions
    (with leaky ReLU activation functions), and a parallel residual connection.
    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`, `L`).
    @param in_channels Number of input channels.
    @param out_channels Number of output channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    @param padding Zero-padding applied to the left of the input of the
           non-residual convolutions.
    @param final Disables, if True, the last activation function.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation,
                 final=False):
        super(CausalConvolutionBlock, self).__init__()

        # Computes left padding so that the applied convolutions are causal
        padding = (kernel_size - 1) * dilation

        # First causal convolution
        conv1 = torch.nn.utils.weight_norm(torch.nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        ))
        # The truncation makes the convolution causal
        chomp1 = Chomp1d(padding)
        relu1 = torch.nn.LeakyReLU()

        # Second causal convolution
        conv2 = torch.nn.utils.weight_norm(torch.nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        ))
        chomp2 = Chomp1d(padding)
        relu2 = torch.nn.LeakyReLU()

        # Causal network
        self.causal = torch.nn.Sequential(
            conv1, chomp1, relu1, conv2, chomp2, relu2
        )

        # Residual connection
        self.upordownsample = torch.nn.Conv1d(
            in_channels, out_channels, 1
        ) if in_channels != out_channels else None

        # Final activation function
        self.relu = torch.nn.LeakyReLU() if final else None

    def forward(self, x):
        out_causal = self.causal(x)
        res = x if self.upordownsample is None else self.upordownsample(x)
        if self.relu is None:
            return out_causal + res
        else:
            return self.relu(out_causal + res)


class CausalCNN(torch.nn.Module):
    """
    Causal CNN, composed of a sequence of causal convolution blocks.
    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C_out`, `L`).
    @param in_channels Number of input channels.
    @param channels Number of channels processed in the network and of output
           channels.
    @param depth Depth of the network.
    @param out_channels Number of output channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    """
    def __init__(self, in_channels, channels, depth, out_channels,
                 kernel_size):
        super(CausalCNN, self).__init__()

        layers = []  # List of causal convolution blocks
        dilation_size = 1  # Initial dilation size

        for i in range(depth):
            in_channels_block = in_channels if i == 0 else channels
            layers += [CausalConvolutionBlock(
                in_channels_block, channels, kernel_size, dilation_size
            )]
            dilation_size *= 2  # Doubles the dilation size at each step

        # Last layer
        layers += [CausalConvolutionBlock(
            channels, out_channels, kernel_size, dilation_size
        )]

        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class CausalCNNEncoder(torch.nn.Module):
    """
    Encoder of a time series using a causal CNN: the computed representation is
    the output of a fully connected layer applied to the output of an adaptive
    max pooling layer applied on top of the causal CNN, which reduces the
    length of the time series to a fixed size.
    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`).
    @param in_channels Number of input channels.
    @param channels Number of channels manipulated in the causal CNN.
    @param depth Depth of the causal CNN.
    @param reduced_size Fixed length to which the output time series of the
           causal CNN is reduced.
    @param out_channels Number of output channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    """
    def __init__(self, in_channels, channels, depth, reduced_size,
                 out_channels, kernel_size):
        super(CausalCNNEncoder, self).__init__()
        self.encoding_size = out_channels
        causal_cnn = CausalCNN(
            in_channels, channels, depth, reduced_size, kernel_size
        )
        reduce_size = torch.nn.AdaptiveMaxPool1d(1)
        squeeze = SqueezeChannels()  # Squeezes the third dimension (time)
        linear = torch.nn.Linear(reduced_size, out_channels)
        self.network = torch.nn.Sequential(
            causal_cnn, reduce_size, squeeze, linear
        )

    def forward(self, x):
        return self.network(x)


class TripletLoss(torch.nn.modules.loss._Loss):
    """
    Triplet loss for representations of time series. Optimized for training
    sets where all time series have the same length.
    Takes as input a tensor as the chosen batch to compute the loss,
    a PyTorch module as the encoder, a 3D tensor (`B`, `C`, `L`) containing
    the training set, where `B` is the batch size, `C` is the number of
    channels and `L` is the length of the time series, as well as a boolean
    which, if True, enables to save GPU memory by propagating gradients after
    each loss term, instead of doing it after computing the whole loss.
    The triplets are chosen in the following manner. First the size of the
    positive and negative samples are randomly chosen in the range of lengths
    of time series in the dataset. The size of the anchor time series is
    randomly chosen with the same length upper bound but the the length of the
    positive samples as lower bound. An anchor of this length is then chosen
    randomly in the given time series of the train set, and positive samples
    are randomly chosen among subseries of the anchor. Finally, negative
    samples of the chosen length are randomly chosen in random time series of
    the train set.
    @param compared_length Maximum length of randomly chosen time series. If
           None, this parameter is ignored.
    @param nb_random_samples Number of negative samples per batch example.
    @param negative_penalty Multiplicative coefficient for the negative sample
           loss.
    """
    def __init__(self, compared_length, nb_random_samples, negative_penalty):
        super(TripletLoss, self).__init__()
        self.compared_length = compared_length
        if self.compared_length is None:
            self.compared_length = np.inf
        self.nb_random_samples = nb_random_samples
        self.negative_penalty = negative_penalty

    def forward(self, batch, encoder, train, save_memory=False):
        batch_size = batch.size(0)
        train_size = train.size(0)
        length = min(self.compared_length, train.size(2))

        # For each batch element, we pick nb_random_samples possible random
        # time series in the training set (choice of batches from where the
        # negative examples will be sampled)
        samples = np.random.choice(
            train_size, size=(self.nb_random_samples, batch_size)
        )
        samples = torch.LongTensor(samples)

        # Choice of length of positive and negative samples
        length_pos_neg = self.compared_length
        # length_pos_neg = np.random.randint(1, high=length + 1)


        # We choose for each batch example a random interval in the time
        # series, which is the 'anchor'
        random_length = self.compared_length
        # random_length = np.random.randint(
        #     length_pos_neg, high=length + 1
        # )  # Length of anchors

        beginning_batches = np.random.randint(
            0, high=length - random_length + 1, size=batch_size
        )  # Start of anchors

        # The positive samples are chosen at random in the chosen anchors
        beginning_samples_pos = np.random.randint(
            0, high=random_length - length_pos_neg + 1, size=batch_size
        )  # Start of positive samples in the anchors
        # Start of positive samples in the batch examples
        beginning_positive = beginning_batches + beginning_samples_pos
        # End of positive samples in the batch examples
        end_positive = beginning_positive + length_pos_neg + np.random.randint(0,window_size)

        # We randomly choose nb_random_samples potential negative samples for
        # each batch example
        beginning_samples_neg = np.random.randint(
            0, high=length - length_pos_neg + 1,
            size=(self.nb_random_samples, batch_size)
        )

        # print('Actual ...............', torch.cat(
        #     [batch[
        #         j: j + 1, :,
        #         beginning_batches[j]: beginning_batches[j] + random_length
        #     ] for j in range(batch_size)]).shape)

        representation = encoder(torch.cat(
            [batch[
                j: j + 1, :,
                beginning_batches[j]: beginning_batches[j] + random_length
            ] for j in range(batch_size)]
        ))  # Anchors representations


        # print('Positive ......', torch.cat([batch[
        #                  j: j + 1, :, end_positive[j] - length_pos_neg: end_positive[j]
        #                  ] for j in range(batch_size)]).shape)

        positive_representation = encoder(torch.cat(
            [batch[
                j: j + 1, :, end_positive[j] - length_pos_neg: end_positive[j]
            ] for j in range(batch_size)]
        ))  # Positive samples representations

        size_representation = representation.size(1)
        # Positive loss: -logsigmoid of dot product between anchor and positive
        # representations
        loss = -torch.mean(torch.nn.functional.logsigmoid(torch.bmm(
            representation.view(batch_size, 1, size_representation),
            positive_representation.view(batch_size, size_representation, 1)
        )))

        # If required, backward through the first computed term of the loss and
        # free from the graph everything related to the positive sample
        if save_memory:
            loss.backward(retain_graph=True)
            loss = 0
            del positive_representation
            torch.cuda.empty_cache()

        multiplicative_ratio = self.negative_penalty / self.nb_random_samples
        for i in range(self.nb_random_samples):
            # Negative loss: -logsigmoid of minus the dot product between
            # anchor and negative representations

            # print('Negative .....', torch.cat([train[samples[i, j]: samples[i, j] + 1][
            #         :, :,
            #         beginning_samples_neg[i, j]:
            #         beginning_samples_neg[i, j] + length_pos_neg
            #     ] for j in range(batch_size)]).shape)

            negative_representation = encoder(
                torch.cat([train[samples[i, j]: samples[i, j] + 1][
                    :, :,
                    beginning_samples_neg[i, j]:
                    beginning_samples_neg[i, j] + length_pos_neg
                ] for j in range(batch_size)])
            )
            loss += multiplicative_ratio * -torch.mean(
                torch.nn.functional.logsigmoid(-torch.bmm(
                    representation.view(batch_size, 1, size_representation),
                    negative_representation.view(
                        batch_size, size_representation, 1
                    )
                ))
            )
            # If required, backward through the first computed term of the loss
            # and free from the graph everything related to the negative sample
            # Leaves the last backward pass to the training procedure
            if save_memory and i != self.nb_random_samples - 1:
                loss.backward(retain_graph=True)
                loss = 0
                del negative_representation
                torch.cuda.empty_cache()

        return loss


# class TripletLoss(torch.nn.modules.loss._Loss):
#     """
#     Triplet loss for representations of time series. Optimized for training
#     sets where all time series have the same length.
#     Takes as input a tensor as the chosen batch to compute the loss,
#     a PyTorch module as the encoder, a 3D tensor (`B`, `C`, `L`) containing
#     the training set, where `B` is the batch size, `C` is the number of
#     channels and `L` is the length of the time series, as well as a boolean
#     which, if True, enables to save GPU memory by propagating gradients after
#     each loss term, instead of doing it after computing the whole loss.
#     The triplets are chosen in the following manner. First the size of the
#     positive and negative samples are randomly chosen in the range of lengths
#     of time series in the dataset. The size of the anchor time series is
#     randomly chosen with the same length upper bound but the the length of the
#     positive samples as lower bound. An anchor of this length is then chosen
#     randomly in the given time series of the train set, and positive samples
#     are randomly chosen among subseries of the anchor. Finally, negative
#     samples of the chosen length are randomly chosen in random time series of
#     the train set.
#     @param compared_length Maximum length of randomly chosen time series. If
#            None, this parameter is ignored.
#     @param nb_random_samples Number of negative samples per batch example.
#     @param negative_penalty Multiplicative coefficient for the negative sample
#            loss.
#     """
#     def __init__(self, compared_length, nb_random_samples, negative_penalty):
#         super(TripletLoss, self).__init__()
#         self.compared_length = compared_length
#         if self.compared_length is None:
#             self.compared_length = numpy.inf
#         self.nb_random_samples = nb_random_samples
#         self.negative_penalty = negative_penalty
#
#     def forward(self, batch, encoder, train, save_memory=False):
#         batch_size = batch.size(0)
#         train_size = train.size(0)
#         length = min(self.compared_length, train.size(2))
#
#         # For each batch element, we pick nb_random_samples possible random
#         # time series in the training set (choice of batches from where the
#         # negative examples will be sampled)
#         samples = numpy.random.choice(
#             train_size, size=(self.nb_random_samples, batch_size)
#         )
#         samples = torch.LongTensor(samples)
#
#         # Choice of length of positive and negative samples
#         length_pos_neg = numpy.random.randint(1, high=length + 1)
#
#         # We choose for each batch example a random interval in the time
#         # series, which is the 'anchor'
#         random_length = numpy.random.randint(
#             length_pos_neg, high=length + 1
#         )  # Length of anchors
#         beginning_batches = numpy.random.randint(
#             0, high=length - random_length + 1, size=batch_size
#         )  # Start of anchors
#
#         # The positive samples are chosen at random in the chosen anchors
#         beginning_samples_pos = numpy.random.randint(
#             0, high=random_length - length_pos_neg + 1, size=batch_size
#         )  # Start of positive samples in the anchors
#         # Start of positive samples in the batch examples
#         beginning_positive = beginning_batches + beginning_samples_pos
#         # End of positive samples in the batch examples
#         end_positive = beginning_positive + length_pos_neg
#
#         # We randomly choose nb_random_samples potential negative samples for
#         # each batch example
#         beginning_samples_neg = numpy.random.randint(
#             0, high=length - length_pos_neg + 1,
#             size=(self.nb_random_samples, batch_size)
#         )
#
#         representation = encoder(torch.cat(
#             [batch[
#                 j: j + 1, :,
#                 beginning_batches[j]: beginning_batches[j] + random_length
#             ] for j in range(batch_size)]
#         ))  # Anchors representations
#
#         positive_representation = encoder(torch.cat(
#             [batch[
#                 j: j + 1, :, end_positive[j] - length_pos_neg: end_positive[j]
#             ] for j in range(batch_size)]
#         ))  # Positive samples representations
#
#         size_representation = representation.size(1)
#         # Positive loss: -logsigmoid of dot product between anchor and positive
#         # representations
#         loss = -torch.mean(torch.nn.functional.logsigmoid(torch.bmm(
#             representation.view(batch_size, 1, size_representation),
#             positive_representation.view(batch_size, size_representation, 1)
#         )))
#
#         # If required, backward through the first computed term of the loss and
#         # free from the graph everything related to the positive sample
#         if save_memory:
#             loss.backward(retain_graph=True)
#             loss = 0
#             del positive_representation
#             torch.cuda.empty_cache()
#
#         multiplicative_ratio = self.negative_penalty / self.nb_random_samples
#         for i in range(self.nb_random_samples):
#             # Negative loss: -logsigmoid of minus the dot product between
#             # anchor and negative representations
#             negative_representation = encoder(
#                 torch.cat([train[samples[i, j]: samples[i, j] + 1][
#                     :, :,
#                     beginning_samples_neg[i, j]:
#                     beginning_samples_neg[i, j] + length_pos_neg
#                 ] for j in range(batch_size)])
#             )
#             loss += multiplicative_ratio * -torch.mean(
#                 torch.nn.functional.logsigmoid(-torch.bmm(
#                     representation.view(batch_size, 1, size_representation),
#                     negative_representation.view(
#                         batch_size, size_representation, 1
#                     )
#                 ))
#             )
#             # If required, backward through the first computed term of the loss
#             # and free from the graph everything related to the negative sample
#             # Leaves the last backward pass to the training procedure
#             if save_memory and i != self.nb_random_samples - 1:
#                 loss.backward(retain_graph=True)
#                 loss = 0
#                 del negative_representation
#                 torch.cuda.empty_cache()
#
#         return loss


def epoch_run(data, encoder, device, window_size, optimizer=None, train=True):
    if train:
        encoder.train()
    else:
        encoder.eval()
    encoder.to(device)
    loss_criterion = TripletLoss(compared_length=window_size, nb_random_samples=25, negative_penalty=1)

    epoch_loss = 0
    acc = 0
    dataset = torch.utils.data.TensorDataset(torch.Tensor(data), torch.zeros((len(data),1)))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=20, shuffle=True, sampler=None, batch_sampler=None,
                    num_workers=0, collate_fn=None, pin_memory=False, drop_last=False)
    for batch,y in data_loader:
        loss = loss_criterion(batch, encoder, torch.Tensor(data))
        epoch_loss += loss.item()
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return epoch_loss / len(x), acc/len(x)


def learn_encoder(x, encoder, window_size, data, lr=0.001, decay=0, n_epochs=100, device='cpu'):
    n_train = int(len(x)*0.8)
    inds = list(range(len(x)))
    random.shuffle(inds)
    x = x[inds]
    params = encoder.parameters()
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=decay)

    train_loss, test_loss = [], []
    for epoch in range(n_epochs):
        epoch_loss, acc = epoch_run(x[:n_train], encoder, device, window_size, optimizer=optimizer, train=True)
        epoch_loss_test, acc_test = epoch_run(x[n_train:], encoder, device, window_size, optimizer=optimizer, train=False)
        print('\nEpoch ', epoch)
        print('Train ===> Loss: ', epoch_loss, '\t Accuracy: ', acc)
        print('Test ===> Loss: ', epoch_loss_test, '\t Accuracy: ', acc_test)
        train_loss.append(epoch_loss)
        test_loss.append(epoch_loss_test)
    plt.figure()
    plt.plot(np.arange(n_epochs), train_loss, label="Train")
    plt.plot(np.arange(n_epochs), test_loss, label="Test")
    plt.title("Loss")
    plt.legend()
    plt.savefig(os.path.join("./plots/%s_trip/loss.pdf"%data))
    state = {
        'epoch': epoch,
        'encoder_state_dict': encoder.state_dict()
    }
    torch.save(state, './ckpt/%s_trip/checkpoint.pth.tar'%data)



window_size = 50
data = 'waveform'
device = 'cuda' if torch.cuda.is_available() else 'cpu'


if data =='waveform':
    path = './data/waveform_data/processed'
    encoding_size = 64
    window_size = 2500
    encoder = WFEncoder(encoding_size=64).to(device)
    with open(os.path.join(path, 'x_train.pkl'), 'rb') as f:
        x = pickle.load(f)
    T = x.shape[-1]
    x = np.concatenate(np.split(x[:,:,:T//20*20], 20, -1), 0)
    learn_encoder(x, encoder, window_size, n_epochs=150, lr=1e-5, decay=1e-3, data=data)
    with open(os.path.join(path, 'x_test.pkl'), 'rb') as f:
        x_test = pickle.load(f)
    with open(os.path.join(path, 'state_test.pkl'), 'rb') as f:
        y_test = pickle.load(f)
    x_test = np.concatenate(np.split(x_test[:,:,:T//100*100], 100, -1), 0)
    y_test = np.concatenate(np.split(y_test[:, :T // 100 * 100], 100, -1), 0)
    plot_distribution(x_test, y_test, encoder, window_size=window_size, path='%s_trip' % data, device=device)
    exp = WFClassificationExperiment(window_size=window_size)
    exp.run(data='waveform_trip', n_epochs=15, lr_e2e=0.001, lr_cls=0.001)


else:
    path = './data/simulated_data/'
    window_size = 50
    encoder = RnnEncoder(hidden_size=100, in_channel=3, encoding_size=10, device=device)
    with open(os.path.join(path, 'x_train.pkl'), 'rb') as f:
        x = pickle.load(f)
    learn_encoder(x, encoder, window_size, lr=1e-5, decay=0.1, data=data)
    with open(os.path.join(path, 'x_test.pkl'), 'rb') as f:
        x_test = pickle.load(f)
    with open(os.path.join(path, 'state_test.pkl'), 'rb') as f:
        y_test = pickle.load(f)
    plot_distribution(x_test, y_test, encoder, window_size=window_size, path='%s_trip' % data, device=device)
    exp = WFClassificationExperiment(window_size=window_size)
    exp.run(data='waveform_trip', n_epochs=15, lr_e2e=0.001, lr_cls=0.001)
