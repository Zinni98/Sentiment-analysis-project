from torch.utils.data import Dataset
from torchtext.vocab import GloVe
import torch
import numpy as np
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

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


def get_data(batch_size: int, dataset, collate_fn, random_state = 42):

  max_element = dataset.max_element

  # Random Split
  train_indexes, test_indexes = train_test_split(list(range(len(dataset.labels))), test_size = 0.2,
                                                  stratify = dataset.labels, random_state = random_state)

  train_ds = Subset(dataset, train_indexes)
  test_ds = Subset(dataset, test_indexes)

  train_loader = DataLoader(train_ds, batch_size = batch_size, collate_fn = collate_fn, pin_memory=True)
  test_loader = DataLoader(test_ds, batch_size = batch_size, collate_fn = collate_fn, pin_memory=True)

  return train_loader, test_loader