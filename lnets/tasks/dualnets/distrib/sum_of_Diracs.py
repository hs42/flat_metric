from unittest import mock
import numpy as np
from munch import Munch
from lnets.tasks.dualnets.distrib.base_distrib import BaseDistrib

import multi_spherical_shell

class Sum_of_Diracs_with_different_radii(BaseDistrib):
    def __init__(self, config):
        super(Sum_of_Diracs_with_different_radii, self).__init__(config)

        self.dim = config.dim
        self.center_x = config.center_x
        self.l = config.l
        self.l_fraction = config.l_fraction
        self.m = 0
        self.r_i = np.zeros(self.l)

        assert self.dim > 0, "Dimensionality must be larger than 0. " 

    def __call__(self, size):

        self.m = size
        #print("In call fct der Distrb: ",  size - self.l)
        self.r_i = np.random.uniform(0.0, 2.0, self.l)
        r_i_outside = np.random.uniform(2.0, 200.0, size - self.l)

        mock_config = Munch({"empty_dim": 0, "reshape_to_grid" : False, "radius": 1.0, "dim": self.dim, "center_x": self.center_x})

        generator_of_shell_samples = multi_spherical_shell.MultiSphericalShell(mock_config)

        samples_within = generator_of_shell_samples(self.l)
        samples_within = self.r_i[:, None] * samples_within #rescale such that each radius is different

        samples_without = generator_of_shell_samples(size - self.l)
        samples_without = r_i_outside[:, None] * samples_without #rescale such that each radius is different

        samples = np.concatenate((samples_within, samples_without))
        np.random.shuffle(samples)

        return samples

    def get_groundtruth(self, size_other_distr):
        radii_to_sum_up = np.sort(self.r_i)
        up_to = min(self.l, size_other_distr)

        term1 = np.sum(radii_to_sum_up[0:up_to])
        term2 = np.abs(self.l - size_other_distr)
        term3 = self.m - self.l

        return (term1 + term2 + term3) / min(size_other_distr, self.m) #to make comparable with our estimate which normalizes by min(n,m)