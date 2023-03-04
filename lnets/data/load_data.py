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
    The linked_samples class provides a method to concatenate a dataloader for distribution 1 and for distribution 2 into a 
    common dataloader. A linked_samples will return a tuple (samples from distribution 1, samples from distribution 2), which is 
    the format needed in the actual training specified in lnets/tasks/dualnets/mains/train_dual.py
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
                #fetch the next samples from the data file
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
    The build_loaders() method returns a dictionary of linked_samples objects, each of which will yield tuples of the form 
    (samples from distribution 1, samples from distribution 2).

    In a first step, a dictionary of PyTorch Dataloader objects for the training, testing, and validation dataset is constructed for
    each distribution. Afterwads, the dataloaders of each distribution are put together in linked_samples object, which will return the
    aforementioned tuples of samples. Lastly, those linked_samples objects are assembled in a dictionary.
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
    The load_data method provides a method to return a tuple of PyTorches dataloader objects corresponding to 
    training, test, and validation data for both distributions.

    Returns a dictionary of linked_samples objects, each of which will yield tuples of the form 
    (samples from distribution 1, samples from distribution 2). This is consistent with the usage of load_distrib() in 
    lnets/tasks/dualnets/distrib/load_distrib.py such that both methods can be used without changing the syntax in
    training routine lnets/tasks/dualnets/main/train_dual.py 

    In contrast to the load_distrib() method, the load_data() function relies on a text file of data being specified.
    New samples of both distributions are read from this data file rather than generated according to a law (which is what load_distrib() does).
    """

    if 'path_train' in cfg.distrib1 and os.path.exists(cfg.distrib1.path_train):
        train_data1 = custom_dataset(cfg.distrib1.path_train)
        train_data2 = custom_dataset(cfg.distrib2.path_train)

    if 'path_val' in cfg.distrib1 and os.path.exists(cfg.distrib1.path_val):
        val_data1 = custom_dataset(cfg.distrib1.path_val)
        val_data2 = custom_dataset(cfg.distrib2.path_val)
    else:
        val_data1 = None
        val_data2 = None

    if 'path_test' in cfg.distrib1 and os.path.exists(cfg.distrib1.path_test):
        test_data1 = custom_dataset(cfg.distrib1.path_test)
        test_data2 = custom_dataset(cfg.distrib2.path_test)
    else:
        test_data1 = None
        test_data2 = None


    return build_loaders(cfg, [train_data1, train_data2], [val_data1, val_data2], [test_data1, test_data2])
