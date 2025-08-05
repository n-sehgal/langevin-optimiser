import torch
from .Parameter import Parameter


class ConstantParameter(Parameter):
    """ Constant parameter class

        Parameters
        ----------
        val : float
            The constant value to be returned when __call__ is called
    """
    def __init__(self, val, num_data_points):
        self.val = val
        self.num_data_points = num_data_points

    @staticmethod
    def log_prior(beta):
        return 0

    def __call__(self, beta):
        return self.val * torch.ones(len(beta), 1, dtype=beta.dtype, device=beta.device)

    def num_coefficients(self):
        return 0
