from unicodedata import bidirectional
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence
import torch.nn.functional as F
import torch

class BiLSTM(nn.Module):
    def __init__(self, device = "cuda", input_size = 300, hidden_size = 128, output_size = 2):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first = True, bidirectional=True, num_layers = 2)
        self.fc = nn.Sequential(nn.ReLU(),
                                nn.BatchNorm1d(hidden_size*2, eps = 1e-08),
                                nn.Dropout(0.3),
                                nn.Linear(hidden_size*2, output_size)
                                )


    # function taken from https://discuss.pytorch.org/t/how-to-use-pack-sequence-if-we-are-going-to-use-word-embedding-and-bilstm/28184/4
    def simple_elementwise_apply(fn, packed_sequence):
        """applies a pointwise function fn to each element in packed_sequence"""
        return torch.nn.utils.rnn.PackedSequence(fn(packed_sequence.data), packed_sequence.batch_sizes)

    def init_hidden(self, batch_size):
        if self.cuda:
            return (torch.zeros(4, batch_size, self.hidden_size).to(self.device),
                    torch.zeros(4, batch_size, self.hidden_size).to(self.device),)

    def forward(self, x):
        batch_size = x.batch_sizes[0].item()
        hidden = self.init_hidden(batch_size)

        # output: batch_size, sequence_length, hidden_size * 2 (since is bilstm)
        out, _ = self.lstm(x, hidden)
        out, input_sizes = pad_packed_sequence(out, batch_first=True)
        # Interested only in the last layer
        out = out[list(range(batch_size)), input_sizes - 1, :]
        out = self.fc(out)

        return out