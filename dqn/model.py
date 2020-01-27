# -*- coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        # TODO: Build a Q-network
        # okay we're really going to just use the size = (8,) state
        # that env.step(action) returns as the input for our QNetwork agent.
        # no im = env.render(mode="rgb_array") business. It's cool that
        # I'm learning to do that but no it's not a part of this exercise.

        self.state_size = state_size
        self.action_size = action_size

        self.fc1 = nn.Linear(self.state_size, 6)
        self.fc2 = nn.Linear(6, 6)
        self.fc3 = nn.Linear(6, self.action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.relu(self.fc3(x))
