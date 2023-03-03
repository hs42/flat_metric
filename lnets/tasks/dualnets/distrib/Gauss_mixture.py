import numpy as np

from lnets.tasks.dualnets.distrib.base_distrib import BaseDistrib


class Gauss_mix(BaseDistrib):
    def __init__(self, config):
        super(Gauss_mix, self).__init__(config)

        self.dim = config.dim
        self.mu1 = config.mu1
        self.sigma1 = config.sigma1
        self.mu2 = config.mu2
        self.sigma2 = config.sigma2

        assert self.dim > 0, "Dimensionality must be larger than 0. " 
        assert self.sigma1 > 0 and self.sigma2 > 0, "Standard deviation must be strictly larger than 0. "

    def __call__(self, size):
        samples1 = np.random.multivariate_normal(mean=(self.mu1) * np.ones(shape=self.dim), cov=self.sigma1**2 * np.eye(self.dim),
                                                    size=size//2)

        samples2 = np.random.multivariate_normal(mean=(self.mu2) * np.ones(shape=self.dim), cov=self.sigma2**2 * np.eye(self.dim),
                                                    size=size//2)                           
            
        samples = np.concatenate((samples1, samples2))
        np.random.shuffle(samples)
        return samples
