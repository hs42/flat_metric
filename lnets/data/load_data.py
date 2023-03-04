from cmath import exp
import os
import numpy as np

from torch.utils.data import Subset, DataLoader
import torchvision.datasets as datasets

from lnets.data.data_transforms import get_data_transforms
from lnets.data.utils import load_indices

from lnets.tasks.dualnets.mains.custom_dataset import *
from lnets.tasks.dualnets.distrib.load_distrib import DistribLoader


class linked_samples(DistribLoader):
    """
    The linked_samples class provides a method to 
    """

    def __init__(self, config, dataloader1, dataloader2, mode="train"): #config,
        #super().__init__(self, mode="train") #config
        assert mode == "train" or mode == "test", "Mode must be either 'train' or 'test'."
        
        self.config = config
        self.mode = mode
        self.sampled_so_far = 0

        self.dataloader1 = dataloader1
        self.dataloader2 = dataloader2
       

    def __reset_iters(self):
        return iter(self.dataloader1), iter(self.dataloader2)

    def __next__(self):
        if self.sampled_so_far < self.config.optim.epoch_len:
            self.sampled_so_far += 1
            
            # In the end, we need a way, to extract batch_size (encoded in the Dataloaders) many samples from dataloader1 and dataloader2
            # 
            try:
                distrib1_samples = next(self.dataloader1)
                distrib2_samples = next(self.dataloader2)
            except: #if iterator objects of dataloader1 or dataloader2 are exhausted. Want to begin all over again in this case
                # Note that this is an artefact of having dataloaders within the linked_samples::__next__() method
                # Later on, we will iterate in trainers/trainer.py as follows:
                # for sample in instance_of_linked_samples:
                #     Do some training

                iter1, iter2 = self.__reset_iters()
                distrib1_samples = next(iter1)
                distrib2_samples = next(iter2)

            # if the samples are already PyTorch tensors, don't touch them.
            if not isinstance(distrib1_samples, torch.Tensor):
                distrib1_samples = torch.from_numpy(distrib1_samples).float()

            if not isinstance(distrib2_samples, torch.Tensor):
                distrib2_samples = torch.from_numpy(distrib2_samples).float()

            return (distrib1_samples.float(),
                    distrib2_samples.float())
        else:
            raise StopIteration



def build_loaders(cfg, train_data, val_data, test_data):
    """
    data_name = config['data']['name'].lower()
    batch_size = config['optim']['batch_size']
    num_workers = config['data']['num_workers']
    """

    loaders_distrib = [dict(), dict()]


    for i in range(2):
        """
        use all available data in a single batch. For once, this is because we dont have
        many data points to begin with, but also because different sizes of the data sets
        encode different normalizations of the corresponding distributions. So, we need to
        make sure, that the ratio of the data sets 1 and 2 is respected when we return sample
        points. The easiest way to do so is to just return the full data set 1 and 2 respectively.
        """
        trainsize, testsize, valsize = 1, 1, 1
        if train_data[i] is not None:
            trainsize = train_data[i].__len__()
        if val_data[i] is not None:
            valsize = val_data[i].__len__()
        if test_data[i] is not None:
            testsize = test_data[i].__len__()

        batch_size = {'train' : trainsize, 'validation' : valsize, 'test' : testsize}


        #DataLoader(None) gives no exception, so need to catch those manually. Suffices if we do so for i=2, i.e. data2 distribution
        loaders_distrib[i] = {
            'train': DataLoader(train_data[i], batch_size=batch_size['train'], shuffle=True),
            'validation': DataLoader(val_data[i], batch_size=batch_size['validation']),
            'test': DataLoader(test_data[i], batch_size=batch_size['test'])
        }

    final_linked_loader = {}

    if train_data[1] is not None:
        final_linked_loader['train'] = linked_samples(cfg, loaders_distrib[0]['train'], loaders_distrib[1]['train'])
    else:
        final_linked_loader['train'] = None

    if val_data[1] is not None:
        final_linked_loader['validation'] = linked_samples(cfg, loaders_distrib[0]['validation'], loaders_distrib[1]['validation'])
    else:
        final_linked_loader['validation'] = None

    if test_data[1] is not None:
        final_linked_loader['test'] = linked_samples(cfg, loaders_distrib[0]['test'], loaders_distrib[1]['test'])
    else:
        final_linked_loader['test'] = None

    return final_linked_loader

def load_data(cfg):
    """
    The load_data methods
    """

    if 'path_train' in cfg.distrib1 and os.path.exists(cfg.distrib1.path_train):
        train_data1 = custom_text_dataset_for_single_cell_data(cfg.distrib1.path_train)
        train_data2 = custom_text_dataset_for_single_cell_data(cfg.distrib2.path_train)

    if 'path_val' in cfg.distrib1 and os.path.exists(cfg.distrib1.path_val):
        val_data1 = custom_text_dataset_for_single_cell_data(cfg.distrib1.path_val)
        val_data2 = custom_text_dataset_for_single_cell_data(cfg.distrib2.path_val)
    else:
        val_data1 = None
        val_data2 = None

    if 'path_test' in cfg.distrib1 and os.path.exists(cfg.distrib1.path_test):
        test_data1 = custom_text_dataset_for_single_cell_data(cfg.distrib1.path_test)
        test_data2 = custom_text_dataset_for_single_cell_data(cfg.distrib2.path_test)
    else:
        test_data1 = None
        test_data2 = None


    #return build_loaders(config, train_data, val_data, test_data)
    return build_loaders(cfg, [train_data1, train_data2], [val_data1, val_data2], [test_data1, test_data2])
