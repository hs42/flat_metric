import numpy as np

from lnets.tasks.dualnets.distrib.base_distrib import BaseDistrib


class Dirac(BaseDistrib):
    '''
    This is a generator for samples of a Dirac distribution at x. It returns samples
    of \delta(x-self.x_x)*\delta(y-self.x_y)*\delta(z-self.x_z)
    Note that in general, we rather want a Dirac distribution in spherical coordinates
    concerning the radius around x, so \delta(radius - self.radius). This is done
    using the multi_spherical_shell generator.
    '''
    def __init__(self, config):
        super(Dirac, self).__init__(config)

        self.dim = len(config.x)
        self.x = config.x

        assert self.dim > 0, "Dimensionality must be larger than 0. " 

    def __call__(self, size):
        samples = np.repeat(self.x, size)
            
        return samples
