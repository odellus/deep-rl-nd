# -*- coding: utf-8
"""
File:
    td_learning.py

Description:
    A file containing the contents of my solution to the temporal difference
    notebook in the Udacity Deep Reinforcement Learning Nanodegree.
"""
import sys
import gym
import numpy as np
from collections import defaultdict, deque
import matplotlib.pyplot as plt
%matplotlib inline

import check_test
from plot_utils import plot_values

env = gym.make('CliffWalking-v0')

print(env.action_space)
print(env.observation_space)

# define the optimal state-value function
V_opt = np.zeros((4,12))
V_opt[0:13][0] = -np.arange(3, 15)[::-1]
V_opt[0:13][1] = -np.arange(3, 15)[::-1] + 1
V_opt[0:13][2] = -np.arange(3, 15)[::-1] + 2
V_opt[3][0] = -13

plot_values(V_opt)

def get_probs(Q_s, epsilon, nA):
    """ obtains the action probabilities corresponding to epsilon-greedy policy """
    policy_s = np.ones(nA) * epsilon / nA
    best_a = np.argmax(Q_s)
    policy_s[best_a] = 1 - epsilon + (epsilon / nA)
    return policy_s

def select_action_epsilon_greedy(Q, state, epsilon, nA):
    """select the argmax of Q[state] in epsilon greedy fashion."""
    return np.random.choice(np.arange(nA), p=get_probs(Q[state], epsilon, nA)) \
        if state in Q else env.action_space.sample()

def sarsa(env, num_episodes, alpha, gamma=1.0, epsilon=0.2, epsilon_decay=0.999):
    # initialize action-value function (empty dictionary of arrays)
    Q = defaultdict(lambda: np.zeros(env.nA))
    # initialize performance monitor
    # loop over episodes
    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        epsilon *= epsilon_decay
        ## TODO: complete the function
        state = env.reset()
        action = select_action_epsilon_greedy(Q, state, epsilon, env.nA)
        while True:
            next_state, reward, done, info = env.step(action)
            next_action = select_action_epsilon_greedy(Q, next_state, epsilon, env.nA)
            # This is Q-learning. Oops. LOL.
            # Q[state][action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action])

            # Okay THIS is SARSA.
            Q[state][action] += alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])
            state = next_state
            action = next_action
            if done:
                break

    return Q


# obtain the estimated optimal policy and corresponding action-value function
Q_sarsa = sarsa(env, 5000, .01)

# print the estimated optimal policy
policy_sarsa = np.array([np.argmax(Q_sarsa[key]) if key in Q_sarsa else -1 for key in np.arange(48)]).reshape(4,12)
check_test.run_check('td_control_check', policy_sarsa)
print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
print(policy_sarsa)

# plot the estimated optimal state-value function
V_sarsa = ([np.max(Q_sarsa[key]) if key in Q_sarsa else 0 for key in np.arange(48)])
plot_values(V_sarsa)

def get_probs(Q_s, epsilon, nA):
    """ obtains the action probabilities corresponding to epsilon-greedy policy """
    policy_s = np.ones(nA) * epsilon / nA
    best_a = np.argmax(Q_s)
    policy_s[best_a] = 1 - epsilon + (epsilon / nA)
    return policy_s

def select_action_epsilon_greedy(Q, state, epsilon, nA):
    """select the argmax of Q[state] in epsilon greedy fashion."""
    return np.random.choice(np.arange(nA), p=get_probs(Q[state], epsilon, nA)) \
        if state in Q else env.action_space.sample()

def q_learning(env, num_episodes, alpha, gamma=1.0, epsilon=0.2, epsilon_decay=0.999):
    # initialize action-value function (empty dictionary of arrays)
    Q = defaultdict(lambda: np.zeros(env.nA))
    # initialize performance monitor
    # loop over episodes
    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        epsilon *= epsilon_decay
        ## TODO: complete the function
        state = env.reset()
        action = select_action_epsilon_greedy(Q, state, epsilon, env.nA)
        while True:
            next_state, reward, done, info = env.step(action)
            next_action = select_action_epsilon_greedy(Q, next_state, epsilon, env.nA)
            # This is Q-learning.
            Q[state][action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action])

            # This is SARSA.
            # Q[state][action] += alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])
            state = next_state
            action = next_action
            if done:
                break

    return Q


# obtain the estimated optimal policy and corresponding action-value function
Q_sarsamax = q_learning(env, 5000, .01)

# print the estimated optimal policy
policy_sarsamax = np.array([np.argmax(Q_sarsamax[key]) if key in Q_sarsamax else -1 for key in np.arange(48)]).reshape((4,12))
check_test.run_check('td_control_check', policy_sarsamax)
print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
print(policy_sarsamax)

# plot the estimated optimal state-value function
plot_values([np.max(Q_sarsamax[key]) if key in Q_sarsamax else 0 for key in np.arange(48)])


def get_probs(Q_s, epsilon, nA):
    """ obtains the action probabilities corresponding to epsilon-greedy policy """
    policy_s = np.ones(nA) * epsilon / nA
    best_a = np.argmax(Q_s)
    policy_s[best_a] = 1 - epsilon + (epsilon / nA)
    return policy_s

def select_action_epsilon_greedy(Q, state, epsilon, nA):
    """select the argmax of Q[state] in epsilon greedy fashion."""
    return np.random.choice(np.arange(nA), p=get_probs(Q[state], epsilon, nA)) \
        if state in Q else env.action_space.sample()

def expected_sarsa(env, num_episodes, alpha, gamma=1.0, epsilon=0.4, epsilon_decay=0.999):
    # initialize action-value function (empty dictionary of arrays)
    Q = defaultdict(lambda: np.zeros(env.nA))
    # initialize performance monitor
    # loop over episodes
    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        epsilon *= epsilon_decay
        ## TODO: complete the function
        state = env.reset()
        action = select_action_epsilon_greedy(Q, state, epsilon, env.nA)
        while True:
            next_state, reward, done, info = env.step(action)
            next_action = select_action_epsilon_greedy(Q, next_state, epsilon, env.nA)
            # This is Q-learning.
            # Q[state][action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action])

            # This is SARSA.
            # Q[state][action] += alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])

            # This is SARSA expected
            Q[state][action] += alpha * (reward + \
                                         gamma * np.dot(Q[next_state], get_probs(Q[next_state], epsilon, env.nA )) \
                                         - Q[state][action])


            state = next_state
            action = next_action
            if done:
                break

    return Q


# obtain the estimated optimal policy and corresponding action-value function
Q_expsarsa = expected_sarsa(env, 10000, 0.01)

# print the estimated optimal policy
policy_expsarsa = np.array([np.argmax(Q_expsarsa[key]) if key in Q_expsarsa else -1 for key in np.arange(48)]).reshape(4,12)
check_test.run_check('td_control_check', policy_expsarsa)
print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
print(policy_expsarsa)

# plot the estimated optimal state-value function
plot_values([np.max(Q_expsarsa[key]) if key in Q_expsarsa else 0 for key in np.arange(48)])
