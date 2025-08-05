import numpy as np
import torch


def replace_invalid_samples(particles, valid_samples, w=None):
    # Replacement performs inplace
    if w is None:
        w = valid_samples.detach().clone().numpy()
        w = w / w.sum()

    num_valid = valid_samples.sum().item()
    num_invalid = len(valid_samples) - num_valid
    if num_valid == 0:
        raise ValueError("No valid samples")
    elif num_valid == len(valid_samples):
        return

    sampled_index = np.random.choice(np.arange(len(particles)), size=num_invalid, replace=True, p=w)
    with torch.no_grad():
        particles[~valid_samples] = particles[sampled_index]
