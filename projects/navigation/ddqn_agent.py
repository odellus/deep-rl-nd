import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_cfg

# Load configuration from YAML
cfg = load_cfg()

# Define global configuration variables
BUFFER_SIZE = cfg["Agent"]["Buffer_size"]
BATCH_SIZE = cfg["Agent"]["Batch_size"]
GAMMA = cfg["Agent"]["Gamma"]
TAU = cfg["Agent"]["Tau"]
LR = cfg["Agent"]["Lr"]
UPDATE_EVERY = cfg["Agent"]["Update_every"]


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        ## TODO: compute and minimize the loss

        # Formula is:
        # L(theta_i) = ExpectedValue_(s,a,r,s') [ (r + gamma * max_a' Q(s',a')) - Q(s,a) ]

        # First let's calculate max_a' Q(s',a')
        # next_states is s'
        # detatch tensor from computational graph.
        # Find max value for Q(s',:) according to qnetwork_target.
        # unsqueeze so it has appropriate shape.

        # To implement double dqn with the target network we need to change this line.
        # q_next_state = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)


        # We randomly select whether we evaluate qnetwork_target actions with qnetwork_local
        if np.random.rand() >= 0.5:
            # So we pick our next actions according to qnetwork_target
            next_actions = self.qnetwork_target(next_states).detach().max(1)[1].unsqueeze(1)
            # Now evaluate those choices with qnetwork_local
            q_next_state = self.qnetwork_local(next_states).gather(1,next_actions)
        # Or if we evaluate qnetwork_local action with qnetwork_target
        else:
            # Pick the actions with qnetwork_local
            next_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
            # Evaluate the state-action pairs with qnetwork_target
            q_next_state = self.qnetwork_target(next_states).gather(1, next_actions)

        # Okay now let's do the first half of the formula
        # When the rollout is complete, then 1 - dones means q_target is just r
        # which makes sense because there's no next state if rollout is done.
        q_target = rewards + gamma * q_next_state * (1 - dones)

        # This is just the value of Q(s,a) where s is contained in states
        # and a is contained in actions.

        # REMEMBER!!! THERE'S NO SOFTMAX AT THE END OF qnetwork_*. It's not a
        # categorical classifier but a state-action value function approximator!
        q_current = self.qnetwork_local(states).gather(1, actions)

        # All the squares and summations are inside mse_loss
        # The expected value is just an average because we use replay buffer to
        # ensure the samples are i.i.d.
        loss = F.mse_loss(q_current, q_target)

        # Zero out the gradient buffers of the optimizer
        self.optimizer.zero_grad()

        # Compute the gradient of loss wrt qnetwork_local.parameters()
        loss.backward()

        # This updates the qnetwork_local.parameters()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
