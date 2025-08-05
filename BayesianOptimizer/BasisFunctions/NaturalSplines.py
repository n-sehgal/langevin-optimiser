import torch
import numpy as np

import patsy


def natural_splines(X, df=None):
    """ Returns basis functions for natural splines.

        Note natural splines do not have a bias term

        Parameters
        ----------
        X : (N,) np.ndarray or torch.Tensor
            Covariate to build the natural spline on
        df : int, default=None
            Number of degrees of freedom
        include_bias : bool, default=True
            Whether to include bias or not

        Returns
        -------
        torch.Tensor
    """
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()

    return torch.tensor(patsy.cr(X, df=df))


def b_splines(X, df=None, include_bias=True, mean_zero=False):
    """ Returns basis functions for b-splines of degree 3

        Parameters
        ----------
        X : (N,) np.ndarray or torch.Tensor
            Covariate to build the natural spline on
        df : int, default=None
            Number of degrees of freedom
        include_bias : bool, default=True
            Whether to include bias or not

        Returns
        -------
        torch.Tensor
    """
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()

    if include_bias:
        design_matrix = np.concatenate([np.ones((len(X), 1)),
                                        patsy.bs(X, df=df, degree=3, include_intercept=False)], axis=1)
    else:
        design_matrix = patsy.bs(X, df=df, degree=3, include_intercept=False)

    if mean_zero and include_bias:
        mean = design_matrix.mean(axis=0)
        mean[0] = 0

        return torch.tensor(design_matrix - mean[None, :], dtype=torch.float64)

    return torch.tensor(design_matrix)
