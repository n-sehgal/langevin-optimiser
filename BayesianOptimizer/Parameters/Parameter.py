import torch
from abc import ABC, abstractmethod


class Parameter(ABC):
    """ Parameter class for a distribution

        Must implement ``log_prior``, ``__call__`` and ``num_coefficients``
    """
    @abstractmethod
    def log_prior(self, beta):
        """ Log prior """
        pass

    @abstractmethod
    def __call__(self, beta):
        pass

    @abstractmethod
    def num_coefficients(self):
        """ Returns the number of coefficients """
        pass

    def initializer(self, num_particles):
        return torch.randn(num_particles, self.num_coefficients(), dtype=torch.float64)
