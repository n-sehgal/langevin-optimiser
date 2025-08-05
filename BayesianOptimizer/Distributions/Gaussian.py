import torch
import numpy as np

from .BaseDistribution import BaseDistribution


class Gaussian(BaseDistribution):
    def __init__(self, Y, mu, sigma):
        super().__init__(Y, mu=mu, sigma=sigma)

    @staticmethod
    def log_pdf(Y, mu, sigma):
        return -0.5 * torch.log(2 * np.pi * sigma ** 2) - 0.5 * ((Y - mu) / sigma) ** 2

    @staticmethod
    def get_valid_samples(Y, mu, sigma):
        return (sigma > 0).all(axis=-1)
