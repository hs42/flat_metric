import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class custom_dataset(Dataset):
    """
    Class for loading data from a file.

    Requires that the specified file is a text file and can be read with np.loadtxt()

    The class is used in building the data loaders (in lnets/data/load_data.py using PyTorch DataLoader class), so needs to implement the __len__() and __getitem__() routines
    """
    def __init__(self, path):
        try:
            self.data = np.loadtxt(path) #should be a ndarray
        except:
            print('Could not load data at ', path, '\nDoes it exist and is an ndarray?')
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]