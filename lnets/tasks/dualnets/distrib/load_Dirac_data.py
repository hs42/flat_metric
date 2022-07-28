import numpy as np

from lnets.tasks.dualnets.distrib.base_distrib import BaseDistrib


class load_sum_of_Diracs(BaseDistrib):
  
    def __init__(self, config):
        super(load_sum_of_Diracs, self).__init__(config)

        self.dim = len(config.center_x)
        self.x = config.center_x

        assert self.dim > 0, "Dimensionality must be larger than 0. " 

    def __call__(self, size):
        
        samples = np.loadtxt('lnets/tasks/dualnets/distrib/sample_points_sum_of_Diracs')
            
        return samples
