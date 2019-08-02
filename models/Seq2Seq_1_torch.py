import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cude" if torch.cuda.is_available() else "cpu")


class EncoderRnn(nn.Module):
    """
    encoder
    """

    def __init__(self, config):
        super(EncoderRNN, self).__init__()
        self.hidden_size = config["n_hidden"]
        self.input_size = config["maxlen_e"]

        self.embedding = nn.Embedding(self.input_size, self.hidden_size)
        self.lstm = nn.LSTM(
            self.input_size, self.hidden_size, batch_first=True)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.lstm(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class DecoderRnn(nn.Module):
    """
    decoder
    """

    def __init__(self, config):
        super(DecoderRnn, self).__init__()
        self.hidden_size = config["n_hidden"]
        self.output_size = config["maxlen_dec"]

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
