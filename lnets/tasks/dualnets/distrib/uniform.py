import numpy as np

from lnets.tasks.dualnets.distrib.base_distrib import BaseDistrib


class Uniform(BaseDistrib):
    def __init__(self, config):
        super(Uniform, self).__init__(config)

        self.dim = config.dim
        self.center = config.center
        self.supportlength = config.supportlength

        assert self.dim > 0, "Dimensionality must be larger than 0. " 
        assert self.supportlength > 0, "Standard deviation must be strictly larger than 0. "

    def __call__(self, size):
        samples = self.supportlength * np.random.uniform(size=(size,self.dim)) + self.center - 0.5*self.supportlength
            
        return samples
