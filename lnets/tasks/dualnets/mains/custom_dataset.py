import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class custom_text_dataset_for_single_cell_data(Dataset):
    """Class for loading data from a file.
    It is used in building the data loaders (in lnets/data/load_data.py using PyTorch DataLoader class), so needs to implement the __len__() and __getitem__() routines"""
    def __init__(self, path_to_cell_cluster):
        try:
            self.data = np.loadtxt(path_to_cell_cluster) #should be a ndarray
        except:
            print('Could not load data at ', path_to_cell_cluster, '\nDoes it exist and is an ndarray?')
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]