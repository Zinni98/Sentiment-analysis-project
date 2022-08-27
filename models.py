from turtle import forward
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence
import torch.nn.functional as F
import torch

class BiLSTM(nn.Module):
    def __init__(self, embedding_matrix, device = "cuda", input_size = 300, hidden_size = 128, output_size = 2):
        super(BiLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.device = device
        self.embedding = self.create_embedding_layer(embedding_matrix)
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first = True, bidirectional=True)
        self.fc = nn.Sequential(nn.ReLU(),
                                nn.BatchNorm1d(hidden_size*2, eps = 1e-08),
                                nn.Dropout(0.3),
                                nn.Linear(hidden_size*2, output_size)
                                )

    def create_embedding_layer(self, embedding_matrix):
        num_embeddings, embedding_dim = embedding_matrix.shape
        emb_layer = nn.Embedding(num_embeddings, embedding_dim, -1)
        emb_layer.load_state_dict({"weight": embedding_matrix})
        return emb_layer

    # function taken from https://discuss.pytorch.org/t/how-to-use-pack-sequence-if-we-are-going-to-use-word-embedding-and-bilstm/28184/4
    def simple_elementwise_apply(self, fn, packed_sequence):
        """applies a pointwise function fn to each element in packed_sequence"""
        return torch.nn.utils.rnn.PackedSequence(fn(packed_sequence.data), packed_sequence.batch_sizes)

    def init_hidden(self, batch_size):
        if self.cuda:
            return (torch.zeros(2, batch_size, self.hidden_size).to(self.device),
                    torch.zeros(2, batch_size, self.hidden_size).to(self.device),)

    def forward(self, x):
        batch_size = x.batch_sizes[0].item()
        hidden = self.init_hidden(batch_size)

        x = self.simple_elementwise_apply(self.embedding, x)

        # output: batch_size, sequence_length, hidden_size * 2 (since is bilstm)
        out, _ = self.lstm(x, hidden)
        out, input_sizes = pad_packed_sequence(out, batch_first=True)
        # Interested only in the last layer
        out = out[list(range(batch_size)), input_sizes - 1, :]
        out = self.fc(out)

        return out

class BiLSTMAttention(BiLSTM):
    # BiLSTM with attention inspired by the following paper: https://aclanthology.org/S18-1040.pdf
    def __init__(self, embedding_matrix, device="cuda", input_size=300, hidden_size=128, output_size=2):
        super(BiLSTMAttention, self).__init__(embedding_matrix, device, input_size, hidden_size, output_size)
        # Not self attention :)
        self.attention = nn.Linear(self.hidden_size * 2, 1)
    
    def forward(self, x):
        batch_size = x.batch_sizes[0].item()
        hidden = self.init_hidden(batch_size)

        x = self.simple_elementwise_apply(self.embedding, x)

        # output: batch_size, sequence_length, hidden_size * 2 (since is bilstm)
        out, _ = self.lstm(x, hidden)
        out, input_sizes = pad_packed_sequence(out, batch_first=True)

        # reshape to (batch_size * seq_length, hidden)

        attention_values = torch.tanh(self.attention(out)).squeeze()
        attention_weights = torch.softmax(attention_values, dim = 1).unsqueeze(1)
        out = torch.sum(attention_weights.matmul(out), dim = 1)
        out = self.fc(out)

        return out