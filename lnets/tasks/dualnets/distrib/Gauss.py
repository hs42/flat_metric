import numpy as np

from lnets.tasks.dualnets.distrib.base_distrib import BaseDistrib


class Gauss(BaseDistrib):
    """
    A class that yields samples from a Gaussian distribution
    """
    def __init__(self, config):
        super(Gauss, self).__init__(config)

        self.dim = config.dim
        self.mu = config.mu
        self.sigma = config.sigma

        assert self.dim > 0, "Dimensionality must be larger than 0. " 
        assert self.sigma > 0, "Standard deviation must be strictly larger than 0. "

    def __call__(self, size):
        """
        The __call__ method returns samples of Normal(mu, sigma), where sigma is a diagonal matrix (no correlation betweem) coordinates
        """
        samples = np.random.multivariate_normal(mean=self.mu * np.ones(shape=self.dim), cov=self.sigma**2 * np.eye(self.dim),
                                                    size=size)
            
        return samples
