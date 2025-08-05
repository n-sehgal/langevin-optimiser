import torch
from .BaseDistribution import BaseDistribution

class GeneralizedExtreme(BaseDistribution):
    def __init__(self, Y, mu, sigma, xi):
        super().__init__(Y, mu=mu, sigma=sigma, xi=xi)

    @staticmethod
    def t(Y, mu, sigma, xi):
        """
        Computes the helper function :math:`t(x)` of GEV distribution defined above.

        Return
        ------
        output : torch.Tensor of double
            Returns a tensor consisting of all individual probability .
        """
        with torch.no_grad():
            is_gumbell = abs(xi) < 1e-11
            is_not_gumbell = ~is_gumbell
            if is_gumbell.shape[1] == 1:
                reshaped_is_gumbell = is_gumbell * torch.ones(is_gumbell.shape[0], len(Y), dtype=bool)
                reshaped_is_not_gumbell = (~is_gumbell) * torch.ones(is_gumbell.shape[0], len(Y), dtype=bool)
            else:
                reshaped_is_gumbell = is_gumbell
                reshaped_is_not_gumbell = is_not_gumbell

        std_Y = (Y - mu) / sigma
        out = torch.zeros_like(std_Y)

        if is_gumbell.any():
            out[reshaped_is_gumbell] = torch.exp(-std_Y[reshaped_is_gumbell])
        if is_not_gumbell.any():
            temp = (1 + xi * std_Y) ** (-1 / xi)
            out[reshaped_is_not_gumbell] = temp[reshaped_is_not_gumbell]
        return out

    @staticmethod
    def log_pdf(Y, mu, sigma, xi):
        t_vals = GeneralizedExtreme.t(Y, mu, sigma, xi)
        return (xi + 1) * torch.log(t_vals) - t_vals - torch.log(sigma)

    @staticmethod
    def get_valid_samples(Y, mu, sigma, xi):
        # ToDo: when mu, sigma or xi is nan or infinity, this does not work correctly
        return ((xi * (Y[None, :] - mu) / sigma > -1) & (sigma > 0)).all(axis=-1)