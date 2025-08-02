import torch
import numpy as np

class OverdampedLangevin:
    def __init__(self,
                 step_size,
                 beta,
                 num_particles,
                 dimension,
                 potential_function,
                 is_valid = None,
                 which_discretization = 'euler',
                 dtype = torch.float64,
                 initializer = None):
        
        self.step_size = step_size
        self.beta = beta
        self.num_particles = num_particles
        self.dimension = dimension
        self.which_discretization = which_discretization
        self.potential_function = potential_function

        if is_valid is None:
            self.is_valid = lambda particles: torch.ones(len(particles)).to(bool)
        else:
            self.is_valid = is_valid

        if initializer is None:
            self.particles = torch.randn(num_particles, dimension, dtype = dtype, requires_grad = True)
