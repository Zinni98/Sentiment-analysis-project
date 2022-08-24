from torch.utils.data import Dataset
from torchtext.vocab import GloVe
import torch
import numpy as np

class MovieReviewsDataset(Dataset):
  def __init__(self, raw_dataset):
    super(MovieReviewsDataset, self).__init__()
    self.corpus = np.array(raw_dataset[0], dtype = object)
    self.labels = np.array(raw_dataset[1], dtype = object)
    self.max_element = len(max(self.corpus, key=lambda x: len(x)))

  def __len__(self):
    return len(self.corpus)
  
  def elements_to_tensor(self):
    global_vectors = GloVe(name='840B', dim=300)
    for idx, item in enumerate(self.corpus):
      item_tensor = torch.empty(len(item), 300)
      for i in range(len(item)):
        token = item[i]
        item_tensor[i] = global_vectors.get_vecs_by_tokens(token)
      self.corpus[idx] = item_tensor
  
  def __getitem__(self, index):
    item = self.corpus[index]
    label = self.labels[index]
    return (item, label)