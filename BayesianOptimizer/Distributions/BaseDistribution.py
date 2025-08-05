from functools import reduce
import torch

from ..Parameters import Parameter
from ..Parameters import ConstantParameter


class BaseDistribution:
    def __init__(self, Y, **param_classes):
        self.Y = Y

        self._param_classes = param_classes

        self.param_classes = {}
        self.constant_classes = {}

        for key, val in param_classes.items():
            if isinstance(val, Parameter):
                self.param_classes[key] = val
            else:
                self.constant_classes[key] = ConstantParameter(val, len(Y))

        parameter_lengths = [param.num_coefficients() for param in self.param_classes.values()]
        self.slices = [slice(sum(parameter_lengths[:i]), sum(parameter_lengths[:i + 1]))
                       for i in range(len(parameter_lengths))]

    @staticmethod
    def log_pdf(Y, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def get_valid_samples(Y, *args, **kwargs):
        raise NotImplementedError

    def is_valid(self, X):
        out = {key: param(X[:, s]) for (s, (key, param)) in zip(self.slices, self.param_classes.items())}
        out.update({key: param(X) for (key, param) in self.constant_classes.items()})
        return self.get_valid_samples(self.Y, **out)

    def initializer(self, num_particles):
        out = [param.initializer(num_particles) for param in self.param_classes.values()]
        return torch.concat(out, dim=-1).requires_grad_()

    @property
    def num_parameters(self):
        return sum([val.num_coefficients() for val in self.param_classes.values()])

    def __call__(self, X):
        # Minimizing the negative log-likelihood
        out = {key: param(X[:, s]) for (s, (key, param)) in zip(self.slices, self.param_classes.items())}
        out.update({key: param(X) for (key, param) in self.constant_classes.items()})

        log_likelihood = self.log_pdf(self.Y, **out).sum(axis=-1)
        log_priors = reduce(lambda x, y: x + y, [param.log_prior(X[:, s])
                                                 for (s, param) in zip(self.slices, self.param_classes.values())])

        return -(log_likelihood + log_priors)

    def predict(self, X):
        """ X expected to be a torch.tensor of shape (n_dim) """
        assert len(X.shape) == 1
        assert X.shape[0] == self.num_parameters
        X = X.reshape(1, -1)

        out = {key: param(X[:, s]).flatten()
               for (s, (key, param)) in zip(self.slices, self.param_classes.items())}
        # out.update({key: param(X) for (key, param) in self.constant_classes.items()})
        return out
