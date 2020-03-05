import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import load_cfg

# Load configuration from YAML
cfg = load_cfg()

# Define global configuration variables
fc1_size = cfg["Model"]["fc1_size"]
fc2_size = cfg["Model"]["fc2_size"]

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(
        self,
        state_size,
        action_size,
        seed,
        fc1_size=fc1_size,
        fc2_size=fc2_size
        ):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)


        self.state_size = state_size
        self.action_size = action_size

        self.fc1 = nn.Linear(self.state_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, self.action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # We don't use an activation function for the last layer because the function we want to
        # approximate is not bound below by zero. Q(s,a) can have negative values, so don't rectify!
        return self.fc3(x)
