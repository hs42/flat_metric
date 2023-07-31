from unittest import mock
import numpy as np
from munch import Munch
from lnets.tasks.dualnets.distrib.base_distrib import BaseDistrib

import multi_spherical_shell

class Sum_of_Diracs_with_different_radii(BaseDistrib):
    """
    This class provides a way to generate an uniform distribution as used in the "Testing unequal masses and dropping the
    assumptions on the support" experiment. More specifically,
    the data points will first be generated on a dim-dimensional sphere, after which each data point will be scaled
    radially with a factor r_i or r_o_outside such that they lie in a ball of radius 200 around the origin. In doing so,
    it is ensured that a fraction of l_fraction of those will lie within a radius of 2, such that one can analyze the effects
    of transport vs deletion/creation of probability mass in the unbalanced optimal transport problem.
    """
    def __init__(self, config, size):
        super(Sum_of_Diracs_with_different_radii, self).__init__(config)

        self.dim = config.dim
        self.center_x = config.center_x
        self.l = config.l
        self.l_fraction = config.l_fraction
        self.m = size
        self.r_i = np.zeros(self.l)

        assert self.dim > 0, "Dimensionality must be larger than 0. " 

    def __call__(self, size):
        """
        The __call__ method will produce size many samples of the distribution. l_fraction*mass (rounded) many of these will lie in a radius of 2, the other
        ones will lie in a radius of at maximum 200
        """

        if size != self.m:
            raise RuntimeError('The given sample size does not equal the sample size with which this instance was initialized. This will yield errors in the ground truth computation')
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
        """
        Computes the ground truth for this particular distribution (set of data points) according to eq (5.3) in the paper.
        For this, we need to know the masses of mu, nu and the actual value for l (not the fraction, but the absolute number of samples within radius 2)
        """
        radii_to_sum_up = np.sort(self.r_i)
        up_to = min(self.l, size_other_distr)

        term1 = np.sum(radii_to_sum_up[0:up_to])
        term2 = np.abs(self.l - size_other_distr)
        term3 = self.m - self.l


        if term1 == 0:
            #When the radii are not yet set because the __call__ method wasnt called yet, cant compute the ground truth
            raise RuntimeError('Wanted to compute the ground truth before the necessary data points were sampled.')
        

        return (term1 + term2 + term3) / min(size_other_distr, self.m) #to make comparable with our estimate which normalizes by min(n,m)