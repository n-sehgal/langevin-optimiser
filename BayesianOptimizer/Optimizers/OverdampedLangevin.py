import torch
import numpy as np
from .utils import replace_invalid_samples

class OverdampedLangevin:
    def __init__(self, h, beta, num_particles, dim, potential_func, is_valid=None,
                 which_discretization='euler', dtype=torch.float64, initializer=None):
        """ Initialize the OverdampedLangevin optimizer

            Parameters
            ----------
            h : float
                The step size
            num_particles : int
                The number of simultanous particles to run
            dim : int
                The dimension of the problem
            potential_func : function
                The potential
            which_discretization : str, optional (default 'euler')
                Runs the discretization with an Euler-Maruyama or Leimkuhler-Matthews method
            dtype : torch.dtype, optional (default torch.float64)
                dtype to use for the computations
        """
        self.h = h
        self.beta = beta
        self.num_particles = num_particles
        self.dim = dim

        self.which_discretization = which_discretization

        self.potential_func = potential_func
        if is_valid is None:
            self.is_valid = lambda particles: torch.ones(len(particles)).to(bool)
        else:
            self.is_valid = is_valid

        if initializer is None:
            self.particles = torch.randn(num_particles, dim, dtype=dtype, requires_grad=True)
        else:
            self.particles = initializer(num_particles)

        self.old_particles = self.particles.detach().clone()
        self.noise = torch.randn_like(self.particles)

        self.dtype = dtype

    def run(self, num_iters, beta=None, h=None):
        """ Runs the overdamped Langevin and only returns the best solution

        Parameters
        ----------
        num_iters : int
            The number of iterations to run
        beta : float, optional (default `self.beta`)
            The temperature of the SDE, if `beta` is not specified, then `beta` is assumed to be `self.beta`
        h : float, optional (default `self.h`)
            The step size to run the simulation, if `h` is not specified, then `h` is assumed to be `self.h`

        Returns
        -------
        torch.Tensor, float
            The best solution and the corresponding value
        """
        if h is None:
            h = self.h
        if beta is None:
            beta = self.beta

        min_val = np.inf
        best_particle = None
        best_iteration = None
        for i in range(num_iters):
            vals = self._step(h, beta, self.which_discretization)
            min_arg = torch.argmin(vals)
            if min_val > vals[min_arg].item():
                min_val = vals[min_arg]
                best_particle = self.old_particles[min_arg, :].detach().clone()
                best_iteration = i

        return best_particle, min_val, best_iteration

    def run_with_history(self, num_iters, beta=None, h=None):
        """ Runs the overdamped Langevin and returns the trajectories and best solution

        Parameters
        ----------
        num_iters : int
            The number of iterations to run
        beta : float, optional (default `self.beta`)
            The temperature of the SDE, if `beta` is not specified, then `beta` is assumed to be `self.beta`
        h : float, optional (default `self.h`)
            The step size to run the simulation, if `h` is not specified, then `h` is assumed to be `self.h`

        Returns
        -------
        torch.Tensor, torch.Tensor, float
            The history, the best solution, and the corresponding value
        """
        if h is None:
            h = self.h
        if beta is None:
            beta = self.beta

        min_val = np.inf
        best_particle = None
        best_iteration = None
        history = torch.zeros(num_iters, self.num_particles, self.dim, dtype=self.dtype)
        for i in range(num_iters):
            vals = self._step(h, beta, self.which_discretization)
            min_arg = torch.argmin(vals)
            if min_val > vals[min_arg].item():
                min_val = vals[min_arg]
                best_particle = self.old_particles[min_arg, :].detach().clone()
                best_iteration = i

            history[i, :, :] = self.old_particles.detach().clone()

        return history, best_particle, min_val, best_iteration

    def _step(self, h, beta, which_discretization):
        """ Performs one overdamped Langevin step. Note that this function occurs in-place so nothing
            is returned.

            Note: Eventually, we would want to change this code such that each step has `h` and `beta` that
            diminishes as the number of iterations tend to infinity. It is easiest to first start with taking
            `h` and `beta` as constant, then changing the code so that they decay is an easy change. Essentially,
            this is performed by passing different values of `h` and `beta` into this method. That's why we want
            `h` and `beta` to be arguments to this method.
        """
        vals = self.potential_func(self.particles)
        vals.sum().backward()
        if which_discretization == "euler":
            # Performs the update in-place
            with torch.no_grad():
                self.old_particles[:, :] = self.particles[:, :]  # Copies particles in old_particles
                self.particles -= h * self.particles.grad + np.sqrt(2 * h * beta) * self.noise
        else:
            raise NotImplementedError

        with torch.no_grad():
            valid_samples = self.is_valid(self.particles)
            is_nan = ~torch.isnan(self.particles).any(axis=-1)
            valid_samples *= is_nan

        replace_invalid_samples(self.particles, valid_samples)
        self.particles.grad.zero_()
        self.noise = torch.randn_like(self.particles)
        return vals.detach().clone()

    def reset(self):
        self.particles = torch.randn(self.num_particles, self.dim, dtype=self.dtype, requires_grad=True)
