"""
Author: Wbert Adrián Castro Vera (dobleuber)
Source: https://github.com/dobleuber/DeepReinforcementLearningUdacity
License: <unspecified>
"""

import torch.nn as nn
import torch.nn.functional as F

from dobleuber.utils import hidden_init


class Actor(nn.Module):
    """
    Actor (Policy) Model.
    """

    def __init__(self, state_size, action_size, seed, fc1_units=400, fc2_units=300):
        """
        Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        # self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset the network weights
        """
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """
        Build an actor (policy) network that maps states -> actions.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))
