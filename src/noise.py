
import copy
import random

import numpy as np


class OUNoise:
    """Ornstein-Uhlenbeck mechanism for noise"""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2, decay=0.995, minp=.1):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.init_theta = theta
        self.init_sigma = sigma
        self.seed = random.seed(seed)
        np.random.seed(seed)
        self.decay = decay
        self.theta = theta
        self.sigma = sigma
        self.minp = minp
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def decay_noise_params(self, iteration):
        decay_apply = np.min([self.decay**iteration, self.minp])
        self.theta = self.init_theta * decay_apply
        self.sigma = self.init_sigma * decay_apply

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        rand_vals = 2 * np.random.rand(len(x)) - 1
        dx = self.theta * (self.mu - x) + self.sigma * rand_vals
        self.state += dx
        return self.state
