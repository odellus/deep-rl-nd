# coding: utf-8
import gym
env = gym.make("LunarLander-v2")
env.reset()
for j in range(200):
    action = np.random.randint(0,3)
    env.render()
    state, reward, done, _ = env.step(action)
    if done:
        break
        
import numpy as np
for j in range(200):
    action = np.random.randint(0,3)
    env.render()
    state, reward, done, _ = env.step(action)
    if done:
        break
        
u = env.render()
u
state
state.shape
for j in range(200):
    action = np.random.randint(0,3)
    env.render()
    state, reward, done, info = env.step(action)
    if done:
        break
        
env.reset()
for j in range(200):
    action = np.random.randint(0,3)
    env.render()
    state, reward, done, info = env.step(action)
    if done:
        break
        
info
env.reset()
dir(env)
env.observation_space
dir(env)
dir(env.env)
dir(env)
help(env.render)
help(env.render)
im = env.render(mode="rgb_array")
im
im.shape
from PIL import Image
img = Image.fromarray(im)
img.show()
import cv2
help(cv2.resize)
img[:,:,1]
im[:,:,1]
img = Image.fromarray(im[:,:,1])
img.show()
im2 = cv2.resize(im[:,:,1], (84,84))
img = Image.fromarray(im2)
img.show()
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
img = Image.fromarray(gray)
img.show()
gray = cv2.resize(gray, (84,84))
img = Image.fromarray(gray)
img.show()
help(env.render)
im = env.render(mode="rgb_array")
im.shape
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(gray, (84,84))
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
        
from torch import nn
import torch.nn.functional as F
import torch.nn as nn
import torch
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
        
net = Net()
dir(net)
net.parameters
net.parameters()
[x for x in net.parameters()]
len([x for x in net.parameters()])
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
        
qnet = QNetwork(8,4)
qnet = QNetwork(8,4,0)
import numpy as np
qnet(torch.rand(8,))
qnet.backwards()
help(qnet.train)
next_states = torch.randn(32,8)
qnet(next_states)
qnet(next_states).shape
qnet(next_states).detatch()
qnet(next_states).detach()
type(qnet(next_states).detach())
qnet(next_states).detach().max(1)
qnet(next_states).detach().max(1)[0]
qnet(next_states).detach().max(1)[0].unsqueeze(1)
qnet(next_states).detach().max(1)[0].unsqueeze(1).shape
qnet(next_states).detach().max(1)[0].shape
qnet(next_states).detach().max(1)[0].unsqueeze(1).shape
qnet(next_states).detach().max(0)
qnet(next_states).detach().max(1)
qnet(next_states).detach().max(1)[0]
qnet(next_states).detach().max(1)[0].unsqueeze(1)
qnet(next_states).detach().max(1)[0].unsqueeze(1).shape
states = torch.randn(32,8)
states
states.shape
next_states.shape
Q_targets_next = self.qnet(next_states).detach().max(1)[0].unsqueeze(1)
Q_targets_next = qnet(next_states).detach().max(1)[0].unsqueeze(1)
rewards = torch.randn(32,1)
rewards.shape
Q_targets = rewards + (0.99 *Q_targets_next)
Q_targets
Q_targets.shape
help(torch.randint)
actions = torch.randint(3, shape=(32,1))
actions = torch.randint(3, size=(32,1))
actions
actions = torch.randint(4, size=(32,1))
actions
Q_expected = qnet(states).gather(1, actions)
Q_expected
Q_expected.shape
loss = F.mse_loss(Q_expected, Q_targets)
import torch.optim as optim
optimizer = optim.Adam(qnet.parameters(), lr=5e-4)
optimizer.zero_grad()
loss.backward()
optimizer.step()
tau = 1e-3
get_ipython().run_line_magic('save', 'dqn_reply 1-112')
