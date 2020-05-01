import torch
import torch.nn as nn
import time
import numpy as np
import math

def mu_law_expansion(data, mu):
    s = np.sign(data) * (np.exp(np.abs(data) * np.log(mu + 1)) - 1) / mu
    return s

def dilate(x, dilation, init_dilation=1, pad_start=True):
    """
    :param x: Tensor of size (N, C, L), where N is the input dilation, C is the number of channels, and L is the input length
    :param dilation: Target dilation. Will be the size of the first dimension of the output tensor.
    :param pad_start: If the input length is not compatible with the specified dilation, zero padding is used. This parameter determines wether the zeros are added at the start or at the end.
    :return: The dilated tensor of size (dilation, C, L*N / dilation). The output might be zero padded at the start
    """

    [n, c, l] = x.size()
    dilation_factor = dilation / init_dilation
    if dilation_factor == 1:
        return x

    # zero padding for reshaping
    new_l = int(np.ceil(l / dilation_factor) * dilation_factor)
    if new_l != l:
        l = new_l
        x = constant_pad_1d(x, new_l, dimension=2, pad_start=pad_start)

    l_old = int(round(l / dilation_factor))
    n_old = int(round(n * dilation_factor))
    l = math.ceil(l * init_dilation / dilation)
    n = math.ceil(n * dilation / init_dilation)

    # reshape according to dilation
    x = x.permute(1, 2, 0).contiguous()  # (n, c, l) -> (c, l, n)
    x = x.view(c, l, n)
    x = x.permute(2, 0, 1).contiguous()  # (c, l, n) -> (n, c, l)

    return x


class DilatedQueue:
    def __init__(self, max_length, data=None, dilation=1, num_deq=1, num_channels=1, dtype=torch.FloatTensor):
        self.in_pos = 0
        self.out_pos = 0
        self.num_deq = num_deq
        self.num_channels = num_channels
        self.dilation = dilation
        self.max_length = max_length
        self.data = data
        self.dtype = dtype
        if data == None:
            self.data = torch.autograd.Variable(dtype(num_channels, max_length).zero_())

    def enqueue(self, input):
        self.data[:, self.in_pos] = input
        self.in_pos = (self.in_pos + 1) % self.max_length

    def dequeue(self, num_deq=1, dilation=1):
        #       |
        #  |6|7|8|1|2|3|4|5|
        #         |
        start = self.out_pos - ((num_deq - 1) * dilation)
        if start < 0:
            t1 = self.data[:, start::dilation]
            t2 = self.data[:, self.out_pos % dilation:self.out_pos + 1:dilation]
            t = torch.cat((t1, t2), 1)
        else:
            t = self.data[:, start:self.out_pos + 1:dilation]

        self.out_pos = (self.out_pos + 1) % self.max_length
        return t

    def reset(self):
        self.data = torch.autograd.Variable(self.dtype(self.num_channels, self.max_length).zero_())
        self.in_pos = 0
        self.out_pos = 0


class ConstantPad1d(torch.autograd.Function):
    def __init__(self, target_size, dimension=0, value=0, pad_start=False):
        super(ConstantPad1d, self).__init__()
        self.target_size = target_size
        self.dimension = dimension
        self.value = value
        self.pad_start = pad_start

    def forward(self, input):
        self.num_pad = self.target_size - input.size(self.dimension)
        assert self.num_pad >= 0, 'target size has to be greater than input size'

        self.input_size = input.size()

        size = list(input.size())
        size[self.dimension] = self.target_size
        output = input.new(*tuple(size)).fill_(self.value)
        c_output = output

        # crop output
        if self.pad_start:
            c_output = c_output.narrow(self.dimension, self.num_pad, c_output.size(self.dimension) - self.num_pad)
        else:
            c_output = c_output.narrow(self.dimension, 0, c_output.size(self.dimension) - self.num_pad)

        c_output.copy_(input)
        return output

    def backward(self, grad_output):
        grad_input = grad_output.new(*self.input_size).zero_()
        cg_output = grad_output

        # crop grad_output
        if self.pad_start:
            cg_output = cg_output.narrow(self.dimension, self.num_pad, cg_output.size(self.dimension) - self.num_pad)
        else:
            cg_output = cg_output.narrow(self.dimension, 0, cg_output.size(self.dimension) - self.num_pad)

        grad_input.copy_(cg_output)
        return grad_input


def constant_pad_1d(input,
                    target_size,
                    dimension=0,
                    value=0,
                    pad_start=False):
    return ConstantPad1d(target_size, dimension, value, pad_start)(input)



class RnnEncoder(torch.nn.Module):
    def __init__(self, hidden_size, in_channel, encoding_size, cell_type='GRU', num_layers=1, device='cpu', dropout=0, bidirectional=True):
        super(RnnEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.in_channel = in_channel
        self.num_layers = num_layers
        self.cell_type = cell_type
        self.encoding_size = encoding_size
        self.bidirectional = bidirectional
        self.device = device

        self.nn = torch.nn.Sequential(torch.nn.Linear(self.hidden_size*(int(self.bidirectional) + 1), self.encoding_size))
        if cell_type=='GRU':
            self.rnn = torch.nn.GRU(input_size=self.in_channel, hidden_size=self.hidden_size, num_layers=num_layers,
                                    batch_first=False, dropout=dropout, bidirectional=bidirectional)
        elif cell_type=='LSTM':
            self.rnn = torch.nn.LSTM(input_size=self.in_channel, hidden_size=self.hidden_size, num_layers=num_layers,
                                    batch_first=False, dropout=dropout, bidirectional=bidirectional)
        else:
            raise ValueError('Cell type not defined, must be one of the following {GRU, LSTM, RNN}')

    def forward(self, x):
        x = x.permute(2,0,1)
        if self.cell_type=='GRU':
            past = torch.zeros(self.num_layers * (int(self.bidirectional) + 1), x.shape[1], self.hidden_size).to(self.device)
        elif self.cell_type=='LSTM':
            h_0 = torch.zeros(self.num_layers * (int(self.bidirectional) + 1), (x.shape[1]), self.hidden_size).to(self.device)
            c_0 = torch.zeros(self.num_layers * (int(self.bidirectional) + 1), (x.shape[1]), self.hidden_size).to(self.device)
            past = (h_0, c_0).to(self.device)
        out, _ = self.rnn(x, past)  # out shape = [seq_len, batch_size, num_directions*hidden_size]
        encodings = self.nn(out[-1].squeeze(0))
        return encodings


class StateClassifier(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(StateClassifier, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.nn = torch.nn.Sequential(torch.nn.Linear(self.input_size, self.output_size))#,
                                      # torch.nn.ReLU(),
                                      # torch.nn.Dropout(),
                                      # torch.nn.Linear(64, self.output_size))

    def forward(self, x):
        logits = self.nn(x)
        return torch.nn.Softmax(-1)(logits)


class E2EStateClassifier(torch.nn.Module):
    def __init__(self, hidden_size, in_channel, encoding_size, output_size, cell_type='GRU', num_layers=1, dropout=0, bidirectional=True):
        super(E2EStateClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.in_channel = in_channel
        self.num_layers = num_layers
        self.cell_type = cell_type
        self.encoding_size = encoding_size
        self.bidirectional = bidirectional
        self.output_size = output_size

        self.fc = torch.nn.Sequential(torch.nn.Linear(self.hidden_size*(int(self.bidirectional) + 1), self.encoding_size))
        self.nn = torch.nn.Sequential(torch.nn.Linear(self.encoding_size, self.output_size))#,
                                      # torch.nn.ReLU(),
                                      # torch.nn.Dropout(),
                                      # torch.nn.Linear(64, self.output_size))
        if cell_type=='GRU':
            self.rnn = torch.nn.GRU(input_size=self.in_channel, hidden_size=self.hidden_size, num_layers=num_layers,
                                    batch_first=False, dropout=dropout, bidirectional=bidirectional)
        elif cell_type=='LSTM':
            self.rnn = torch.nn.LSTM(input_size=self.in_channel, hidden_size=self.hidden_size, num_layers=num_layers,
                                    batch_first=False, dropout=dropout, bidirectional=bidirectional)
        else:
            raise ValueError('Cell type not defined, must be one of the following {GRU, LSTM, RNN}')

    def forward(self, x):
        x = x.permute(2,0,1)
        if self.cell_type=='GRU':
            past = torch.zeros(self.num_layers * (int(self.bidirectional) + 1), x.shape[1], self.hidden_size)
        elif self.cell_type=='LSTM':
            h_0 = torch.zeros(self.num_layers * (int(self.bidirectional) + 1), (x.shape[1]), self.hidden_size)
            c_0 = torch.zeros(self.num_layers * (int(self.bidirectional) + 1), (x.shape[1]), self.hidden_size)
            past = (h_0, c_0)
        out, _ = self.rnn(x, past)  # out shape = [seq_len, batch_size, num_directions*hidden_size]
        encodings = self.fc(out[-1].squeeze(0))
        return torch.nn.Softmax(-1)(self.nn(encodings))


class MimicEncoder(torch.nn.Module):
    def __init__(self, input_size, in_channel, encoding_size):
        super(MimicEncoder, self).__init__()
        self.input_size = input_size
        self.in_channel = in_channel
        self.encoding_size = encoding_size

        self.nn = torch.nn.Sequential(torch.nn.Linear(input_size, 64),
                                      torch.nn.Dropout(),
                                      torch.nn.ReLU(),
                                      torch.nn.Linear(64, encoding_size))

    def forward(self, x):
        x = torch.mean(x, dim=1)
        encodings = self.nn(x)
        return encodings


class WFEncoder(nn.Module):
    def __init__(self, encoding_size, classify=False, n_classes=None):
        # Input x is (batch, 2, 256)
        super(WFEncoder, self).__init__()

        self.encoding_size = encoding_size
        self.n_classes = n_classes
        self.classify = classify
        self.classifier =None
        if self.classify:
            if self.n_classes is None:
                raise ValueError('Need to specify the number of output classes for te encoder')
            else:
                self.classifier = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(self.encoding_size, 2048),
                    nn.ELU(inplace=True),
                    nn.BatchNorm1d(2048, eps=0.001),
                    nn.Linear(2048, self.n_classes),
                    torch.nn.Softmax(-1)
                )
                nn.init.xavier_uniform_(self.classifier[1].weight)
                nn.init.xavier_uniform_(self.classifier[4].weight)

        self.features = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=4, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(64, eps=0.001),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(64, eps=0.001),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(128, eps=0.001),
            # nn.Dropout(),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(128, eps=0.001),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # nn.Dropout(0.5),
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(256, eps=0.001),
            # nn.Dropout(0.5),
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(256, eps=0.001),
            nn.MaxPool1d(kernel_size=2, stride=2)
            )
        # self.features = torch.nn.DataParallel(self.features)

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(79872, 2048),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(2048, eps=0.001),
            nn.Linear(2048, self.encoding_size)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        encoding = self.fc(x)
        if self.classify:
            c = self.classifier(encoding)
            return c
        else:
            return encoding


class WaveNetModel(nn.Module):
    """
    A Complete Wavenet Model
    Args:
        layers (Int):               Number of layers in each block
        blocks (Int):               Number of wavenet blocks of this model
        dilation_channels (Int):    Number of channels for the dilated convolution
        residual_channels (Int):    Number of channels for the residual connection
        skip_channels (Int):        Number of channels for the skip connections
        classes (Int):              Number of possible values each sample can have
        output_length (Int):        Number of samples that are generated for each input
        kernel_size (Int):          Size of the dilation kernel
        dtype:                      Parameter type of this model
    Shape:
        - Input: :math:`(N, C_{in}, L_{in})`
        - Output: :math:`()`
        L should be the length of the receptive field
    """
    def __init__(self,
                 layers=10,
                 blocks=4,
                 dilation_channels=32,
                 residual_channels=32,
                 skip_channels=256,
                 end_channels=256,
                 classes=256,
                 output_length=32,
                 kernel_size=2,
                 dtype=torch.FloatTensor,
                 bias=False):

        super(WaveNetModel, self).__init__()

        self.layers = layers
        self.blocks = blocks
        self.dilation_channels = dilation_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.classes = classes
        self.kernel_size = kernel_size
        self.dtype = dtype

        # build model
        receptive_field = 1
        init_dilation = 1

        self.dilations = []
        self.dilated_queues = []
        # self.main_convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        # 1x1 convolution to create channels
        self.start_conv = nn.Conv1d(in_channels=self.classes,
                                    out_channels=residual_channels,
                                    kernel_size=1,
                                    bias=bias)

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilations of this layer
                self.dilations.append((new_dilation, init_dilation))

                # dilated queues for fast generation
                self.dilated_queues.append(DilatedQueue(max_length=(kernel_size - 1) * new_dilation + 1,
                                                        num_channels=residual_channels,
                                                        dilation=new_dilation,
                                                        dtype=dtype))

                # dilated convolutions
                self.filter_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=kernel_size,
                                                   bias=bias))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=kernel_size,
                                                 bias=bias))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=1,
                                                     bias=bias))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=1,
                                                 bias=bias))

                receptive_field += additional_scope
                additional_scope *= 2
                init_dilation = new_dilation
                new_dilation *= 2

        self.end_conv_1 = nn.Conv1d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=1,
                                  bias=True)

        self.end_conv_2 = nn.Conv1d(in_channels=end_channels,
                                    out_channels=classes,
                                    kernel_size=1,
                                    bias=True)

        # self.output_length = 2 ** (layers - 1)
        self.output_length = output_length
        self.receptive_field = receptive_field

    def wavenet(self, input, dilation_func):

        x = self.start_conv(input)
        skip = 0

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            (dilation, init_dilation) = self.dilations[i]

            residual = dilation_func(x, dilation, init_dilation, i)

            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.nn.functional.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.nn.functional.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection
            s = x
            print(x.shape, '******************')
            if x.size(2) != 1:
                 s = dilate(x, 1, init_dilation=dilation)
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, -s.size(2):]
            except:
                skip = 0
            skip = s + skip

            x = self.residual_convs[i](x)
            x = x + residual[:, :, (self.kernel_size - 1):]

        x = torch.nn.functional.relu(skip)
        x = torch.nn.functional.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)

        return x

    def wavenet_dilate(self, input, dilation, init_dilation, i):
        x = dilate(input, dilation, init_dilation)
        return x

    def queue_dilate(self, input, dilation, init_dilation, i):
        queue = self.dilated_queues[i]
        queue.enqueue(input.data[0])
        x = queue.dequeue(num_deq=self.kernel_size,
                          dilation=dilation)
        x = x.unsqueeze(0)

        return x

    def forward(self, input):
        x = self.wavenet(input,
                         dilation_func=self.wavenet_dilate)

        # reshape output
        [n, c, l] = x.size()
        l = self.output_length
        x = x[:, :, -l:]
        x = x.transpose(1, 2).contiguous()
        x = x.view(n * l, c)
        return x

    def generate(self,
                 num_samples,
                 first_samples=None,
                 temperature=1.):
        self.eval()
        if first_samples is None:
            first_samples = self.dtype(1).zero_()
        generated = torch.autograd.Variable(first_samples, volatile=True)

        num_pad = self.receptive_field - generated.size(0)
        if num_pad > 0:
            generated = constant_pad_1d(generated, self.scope, pad_start=True)
            print("pad zero")

        for i in range(num_samples):
            input = torch.autograd.Variable(torch.FloatTensor(1, self.classes, self.receptive_field).zero_())
            input = input.scatter_(1, generated[-self.receptive_field:].view(1, -1, self.receptive_field), 1.)

            x = self.wavenet(input,
                             dilation_func=self.wavenet_dilate)[:, :, -1].squeeze()

            if temperature > 0:
                x /= temperature
                prob = torch.nn.functional.softmax(x, dim=0)
                prob = prob.cpu()
                np_prob = prob.data.numpy()
                x = np.random.choice(self.classes, p=np_prob)
                x = torch.autograd.Variable(torch.LongTensor([x]))#np.array([x])
            else:
                x = torch.max(x, 0)[1].float()

            generated = torch.cat((generated, x), 0)

        generated = (generated / self.classes) * 2. - 1
        mu_gen = mu_law_expansion(generated, self.classes)

        self.train()
        return mu_gen

    def generate_fast(self,
                      num_samples,
                      first_samples=None,
                      temperature=1.,
                      regularize=0.,
                      progress_callback=None,
                      progress_interval=100):
        self.eval()
        if first_samples is None:
            first_samples = torch.LongTensor(1).zero_() + (self.classes // 2)
        first_samples = torch.autograd.Variable(first_samples)

        # reset queues
        for queue in self.dilated_queues:
            queue.reset()

        num_given_samples = first_samples.size(0)
        total_samples = num_given_samples + num_samples

        input = torch.autograd.Variable(torch.FloatTensor(1, self.classes, 1).zero_())
        input = input.scatter_(1, first_samples[0:1].view(1, -1, 1), 1.)

        # fill queues with given samples
        for i in range(num_given_samples - 1):
            x = self.wavenet(input,
                             dilation_func=self.queue_dilate)
            input.zero_()
            input = input.scatter_(1, first_samples[i + 1:i + 2].view(1, -1, 1), 1.).view(1, self.classes, 1)

            # progress feedback
            if i % progress_interval == 0:
                if progress_callback is not None:
                    progress_callback(i, total_samples)

        # generate new samples
        generated = np.array([])
        regularizer = torch.pow(torch.autograd.Variable(torch.arange(self.classes)) - self.classes / 2., 2)
        regularizer = regularizer.squeeze() * regularize
        tic = time.time()
        for i in range(num_samples):
            x = self.wavenet(input,
                             dilation_func=self.queue_dilate).squeeze()

            x -= regularizer

            if temperature > 0:
                # sample from softmax distribution
                x /= temperature
                prob = torch.nn.functional.softmax(x, dim=0)
                prob = prob.cpu()
                np_prob = prob.data.numpy()
                x = np.random.choice(self.classes, p=np_prob)
                x = np.array([x])
            else:
                # convert to sample value
                x = torch.max(x, 0)[1][0]
                x = x.cpu()
                x = x.data.numpy()

            o = (x / self.classes) * 2. - 1
            generated = np.append(generated, o)

            # set new input
            x = torch.autograd.Variable(torch.from_numpy(x).type(torch.LongTensor))
            input.zero_()
            input = input.scatter_(1, x.view(1, -1, 1), 1.).view(1, self.classes, 1)

            if (i+1) == 100:
                toc = time.time()
                print("one generating step does take approximately " + str((toc - tic) * 0.01) + " seconds)")

            # progress feedback
            if (i + num_given_samples) % progress_interval == 0:
                if progress_callback is not None:
                    progress_callback(i + num_given_samples, total_samples)

        self.train()
        mu_gen = mu_law_expansion(generated, self.classes)
        return mu_gen


    def parameter_count(self):
        par = list(self.parameters())
        s = sum([np.prod(list(d.size())) for d in par])
        return s

    def cpu(self, type=torch.FloatTensor):
        self.dtype = type
        for q in self.dilated_queues:
            q.dtype = self.dtype
        super().cpu()