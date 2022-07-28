import numpy as np

from lnets.tasks.dualnets.distrib.base_distrib import BaseDistrib


class Dirac(BaseDistrib):
  
    def __init__(self, config):
        super(Dirac, self).__init__(config)

        self.dim = len(config.center_x)
        self.center = config.center_x
        self.supportlength = config.supportlength

        assert self.dim > 0, "Dimensionality must be larger than 0. " 

    def __call__(self, size):
        samples = self.supportlength * np.random.random_sample((size,self.dim)) + self.center - 0.5*self.supportlength
            
        return samples
