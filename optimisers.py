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
                 is_valid=None,
                 which_discretization='euler',
                 dtype=torch.float64,
                 initializer=None):
        
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

        self.old_particles = self.particles.detach().clone()
        # the detach command is useful to detatch the process from the computational graph, so we don't mess with the gradient tracking, and the clone creates a copy at a new point in the memory.
        
        self.noise = torch.randn_like(self.particles)
        # noise is a tensor of the same shape as particles, with random values, this is used to add noise to the particles during the optimization process.

        self.dtype = dtype

        # the function above defines the __init__ method, which is called when the class is initialized, it sets the initial values of the class attributes, which are used in the optimization process.

    # w holds the weights of the valid samples, if it is not provided, then a uniform distribution is assumed.
    def replace_invalid_samples(self, valid_samples, w=None):
        if w is None:
            w = valid_samples.detach().clone().numpy()
            w = w / w.sum()
        
        num_valid = valid_samples.sum().item()
        num_invalid = len(valid_samples) - num_valid
        if num_valid == 0:
            raise ValueError("No valid samples")
        elif num_valid == len(valid_samples):
            return

        sampled_index = np.random.choice(np.arange(len(self.particles)), size=num_invalid, replace=True, p=w)
        with torch.no_grad():
            self.particles[~valid_samples] = self.particles[sampled_index]
            

    def _step(self, step_size, beta, which_discretization):
        # the underscore before the step function indicates that this is a private method, which is not meant to be called outside the class, also beta and step_size are included as arguments to allow for further flexibility later on, where the code can be adjusted to lower the step size and or the temperature once the system approaches an equilibrium.

        vals = self.potential_function(self.particles)
        vals.sum().backward()
        if which_discretization == 'euler':
            with torch.no_grad():
                self.old_particles[:, :] = self.particles[:, :]
                self.particles -= step_size * self.particles.grad + np.sqrt(2 * step_size / beta) * self.noise
        else:
            raise NotImplementedError("Only Euler-Maruyama discretization is implemented for Overdamped Langevin")
        
        # the nan stands for not a number, and checks the simulation to ensure that it does not become unstable
        # tilde ~ is responsible for inverting the result of the function
        with torch.no_grad():
            is_nan = ~torch.isnan(self.particles).any(axis=1) # this checks for any coordinates that are NaN
            valid_samples = self.is_valid(self.particles)
            valid_samples = valid_samples * is_nan

            self.replace_invalid_samples(valid_samples)
            self.particles.grad.zero_()
            self.noise = torch.randn_like(self.particles)
            return vals.detach().clone()
        

        # self.old_particles is only a temporary variable that stores information about the prevoius state of the particles.
    
    def reset(self):
        self.particles = torch.randn(self.num_particles, self.dimension, dtype=self.dtype, requires_grad=True)


    def run(self, num_iterations, beta = None, step_size = None):
        # here is the step size is not specified, then it is assumed to be self, similarly for beta, this gives flexibility to how the optimizer is run in the future.
        # we expect this function to return a tensor with the best values that the optimizer has discovered

        if step_size is None:
            step_size = self.step_size
        if beta is None:
            beta = self.beta
        
        min_val = np.inf
        # this initilized a variable to track the minumum potential energy value found so far. It is set to infinity so that any potential found after the 1st iteration will be lower

        best_particle = None
        # best particle initalized
        best_iteration = None
        # iteration number

        for i in range(num_iterations):
            vals = self._step(step_size, beta, self.which_discretization)
            min_arg = torch.argmin(vals)
            if min_val > vals[min_arg].item():
                min_val = vals[min_arg].item()
            # this in effect stores what the minimum potential energy value that we found is
                best_particle = self.old_particles[min_arg, :].detach().clone()
                best_iteration = i
        
        return best_particle, min_val, best_iteration
    def run_with_history(self, num_iterations, beta = None, step_size = None):
        if step_size is None:
            step_size = self.step_size
        if beta is None:
            beta = self.beta
        
        min_val = np.inf
        best_particle = None
        best_iteration = None
        history = torch.zeros(num_iterations, self.num_particles, self.dimension, dtype=self.dtype)
        for i in range(num_iterations):
            vals = self._step(step_size, beta, self.which_discretization)
            min_arg = torch.argmin(vals)
            if min_val > vals[min_arg].item():
                min_val = vals[min_arg]
                best_particle = self.old_particles[min_arg, :].detach().clone()
                best_iteration = i

            history[i, :, :] = self.old_particles.detach().clone()

        return history, best_particle, min_val, best_iteration
