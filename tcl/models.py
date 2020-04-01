import torch
import torch.nn as nn


class RnnEncoder(torch.nn.Module):
    def __init__(self, hidden_size, in_channel, encoding_size, cell_type='GRU', num_layers=1, dropout=0, bidirectional=True):
        super(RnnEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.in_channel = in_channel
        self.num_layers = num_layers
        self.cell_type = cell_type
        self.encoding_size = encoding_size
        self.bidirectional = bidirectional

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
            past = torch.zeros(self.num_layers * (int(self.bidirectional) + 1), x.shape[1], self.hidden_size)
        elif self.cell_type=='LSTM':
            h_0 = torch.zeros(self.num_layers * (int(self.bidirectional) + 1), (x.shape[1]), self.hidden_size)
            c_0 = torch.zeros(self.num_layers * (int(self.bidirectional) + 1), (x.shape[1]), self.hidden_size)
            past = (h_0, c_0)
        out, _ = self.rnn(x, past)  # out shape = [seq_len, batch_size, num_directions*hidden_size]
        encodings = self.nn(out[-1].squeeze(0))
        return encodings


class StateClassifier(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(StateClassifier, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.nn = torch.nn.Sequential(torch.nn.Linear(self.input_size, 64),
                                      torch.nn.ReLU(),
                                      torch.nn.Dropout(),
                                      torch.nn.Linear(64, self.output_size))

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
        self.nn = torch.nn.Sequential(torch.nn.Linear(self.encoding_size, 64),
                                      torch.nn.ReLU(),
                                      torch.nn.Dropout(),
                                      torch.nn.Linear(64, self.output_size))
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