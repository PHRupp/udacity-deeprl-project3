from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return -lim, lim


class ActorNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        seed: int,
        layers: List[int] = [64, 128, 64],
    ):
        """ Create instance of model
        :param state_size: (int): size of state
        :param action_size: (int) size of action
        :param seed: (int) random seed
        :param layers: List[int] hidden layers sizes
        """
        super(ActorNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, layers[0])
        self.fc2 = nn.Linear(layers[0], layers[1])
        self.fc3 = nn.Linear(layers[1], layers[2])
        #self.fc3 = nn.Linear(layers[1], action_size)
        self.fc4 = nn.Linear(layers[2], action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """ Network that calculates actions from states
        sigmoid, relu, tanh, leaky_relu
        :param state: Iterable[float] represents state
        :return: action: Iterable[int] represents possible actions
        """
        x = state
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        return x


class CriticNetwork(nn.Module):
    """Critic (Value) Model."""

    def __init__(
        self,
        state_size,
        action_size,
        seed,
        layers: List[int] = [256, 256, 128],
    ):
        """ Create instance of model
        :param state_size: (int): size of state
        :param action_size: (int) size of action
        :param seed: (int) random seed
        :param layers: List[int] hidden layers sizes
        """
        super(CriticNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, layers[0])
        self.fc2 = nn.Linear(layers[0] + action_size, layers[1])
        #self.fc3 = nn.Linear(layers[1], layers[2])
        #self.fc4 = nn.Linear(layers[2], 1)
        self.fc3 = nn.Linear(layers[1], 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        #self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        #self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """ Critic Network that calculates state-value to action pairs
        :param state: Iterable[float] represents state
        :param: action: Iterable[float] represents actions
        :return: action: Iterable[int] represents possible actions
        """
        x = F.relu(self.fc1(state))
        x = torch.cat((x, action), dim=1)
        x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        x = self.fc3(x)
        #x = self.fc4(x)
        return x
