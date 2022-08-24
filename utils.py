import torch
import numpy as np
from typing import List
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import operator
from tqdm import tqdm
import torch

def pad(batch, max_size):
    pad = torch.zeros(batch[0].size(dim=1))
    for idx in range(len(batch)):
        remaining = max_size - batch[idx].size(dim = 0)
        batch[idx] = torch.cat((batch[idx], pad.repeat((remaining, 1))), dim = 0)
    return batch

def batch_to_tensor(X: List[torch.tensor], max_size):
    X_tensor = torch.zeros(len(X), max_size, X[0].size(dim = 1))
    for i, embed in enumerate(X):
        X_tensor[i] = embed
    return X_tensor

def sort_ds(X, Y):
    """
    Sort inputs by document lengths
    """
    document_lengths = np.array([tens.size(dim = 0) for tens in X])
    indexes = np.argsort(document_lengths)

    X_sorted = X[indexes][::-1]
    Y_sorted = Y[indexes][::-1]
    document_lengths = torch.from_numpy(document_lengths[indexes][::-1].copy())

    return X_sorted, Y_sorted, document_lengths



def collate(batch):
    X, Y = list(zip(*batch))
    Y = np.array(list(Y))
    X = np.array(list(X))

    # Sort dataset
    X, Y, document_lengths = sort_ds(X, Y)

    # Get tensor sizes
    max_size = torch.max(document_lengths).item()

    # Pad tensor each element
    X = pad(X, max_size)

    # Transform the batch to a tensor
    X_tensor = batch_to_tensor(X, max_size)
    Y_tensor = torch.from_numpy(Y.copy())
    # Return the padded sequence object
    X_final = pack_padded_sequence(X_tensor, document_lengths, batch_first=True)
    return X_final, Y_tensor



# function inspired by https://www.kaggle.com/code/christofhenkel/how-to-preprocessing-when-using-embeddings/notebook
def check_coverage(vocab,embeddings_index):
    a = {}
    oov = {}
    k = 0
    i = 0
    null_embedding = torch.tensor([0.0]*300)
    for word in tqdm(vocab):
        try:
          if torch.equal(embeddings_index.get_vecs_by_tokens(word), null_embedding):
            raise KeyError
          a[word] = embeddings_index.get_vecs_by_tokens(word)
          k += vocab[word]
        except:

            oov[word] = vocab[word]
            i += vocab[word]
            pass

    print()
    print(f'Found embeddings for {len(a) / len(vocab):.2%} of vocab')
    print(f'Found embeddings for  {k / (k + i):.2%} of all text')
    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]

    return sorted_x
