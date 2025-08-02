import torch
import numpy as np

# in overdamped we do not need to worry about velocities as particles moving in an overdamped system are analogous to particles moving through a thick viscous fluid, where the intertia/momentum is negligible, slow moving particles and so we do not care for the velocities.

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

        # is_valid is a function, which if it is not provided, defaults to a function that returns True for all particles, lambda is a method that helps you write a function in a single line.

        if initializer is None:
            self.particles = torch.randn(num_particles, dimension, dtype = dtype, requires_grad = True)
        else:
            self.particles = initializer(num_particles)
        # so a tensor argument takes in (dim1, dim2, dim3, dim4...) and so the self.particles above is a 2d tensor, with num_partices rows and dimension columns, these dimension columns will eventually be the coordinates of the particles, remember that when we are dealing with functions, there is special syntaxs involved for arguments where the keyword has to specified, and not - so we ignore the ones we don't want to specify etc etc...

        self.old_particles = self.particles.detatch().close()
        
        self.noise = torch.randn_like(self.particles)
        