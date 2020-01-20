# -*- coding: utf-8
"""
File:
    monte_carlo.py

Description:
    A file containing code from the IPython notebook on Monte Carlo
    reinforcement learning methods.
"""
import sys
import gym
import numpy as np
from collections import defaultdict

from plot_utils import plot_blackjack_values, plot_policy


env = gym.make('Blackjack-v0')

print(env.observation_space)
print(env.action_space)

for i_episode in range(3):
    state = env.reset()
    while True:
        print(state)
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        if done:
            print('End game! Reward: ', reward)
            print('You won :)\n') if reward > 0 else print('You lost :(\n')
            break
def generate_episode_from_limit_stochastic(bj_env):
    episode = []
    state = bj_env.reset()
    while True:
        probs = [0.8, 0.2] if state[0] > 17 else [0.2, 0.8]
        action = np.random.choice(np.arange(2), p=probs)
        print("The action is {}".format(action))
        next_state, reward, done, info = bj_env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode


for i in range(3):
    print(generate_episode_from_limit_stochastic(env))


from scipy.signal import lfilter

def discount(x, gamma):
    return lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def split_episode(episode):
    states, actions, rewards = [], [], []
    for x in episode:
        states.append(x[0])
        actions.append(x[1])
        rewards.append(x[2])
    return states, actions, rewards

def mc_prediction_q(env, num_episodes, generate_episode, gamma=1.0, first_visit_flag=False):
    # initialize empty dictionaries of arrays
    returns_sum = defaultdict(lambda: np.zeros(env.action_space.n))
    N = defaultdict(lambda: np.zeros(env.action_space.n))
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    # loop over episodes

    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        ## TODO: complete the function
        episode = generate_episode(env)
        states, actions, rewards = split_episode(episode)
        returns = discount(rewards, gamma)
        length_episode = len(returns)
        if first_visit_flag:
            for k in range(length_episode):
                state = states[k]
                action = actions[k]
                _return = returns[k]
                if state not in Q:
                    Q[state][action] = _return
        else:
            for k in range(length_episode):
                state = states[k]
                action = actions[k]
                _return = returns[k]
                returns_sum[state][action] += _return
                N[state][action] += 1
                Q[state][action] = returns_sum[state][action] / N[state][action]
    return Q


# obtain the action-value function
Q = mc_prediction_q(env, 500000, generate_episode_from_limit_stochastic)


# obtain the corresponding state-value function
V_to_plot = dict((k,(k[0]>18)*(np.dot([0.8, 0.2],v)) + (k[0]<=18)*(np.dot([0.2, 0.8],v))) \
         for k, v in Q.items())

# plot the state-value function
plot_blackjack_values(V_to_plot)


from scipy.signal import lfilter

def discount(x, gamma):
    return lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def generate_episode_from_policy(bj_env, policy):
    episode = []
    state = bj_env.reset()
    while True:
        if state in policy:
            action = policy[state]
        else:
            action = np.random.randint(bj_env.action_space.n)
        next_state, reward, done, info = bj_env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode

def get_policy(Q, epsilon, high_pass=0.5, low_pass=0.01):
    if epsilon <= high_pass:
        epsilon = 2.0
    elif epsilon >= low_pass:
        epsilon = -1.0
    policy = defaultdict()
    for k, v in Q.items():
        coin_toss = np.random.rand()
        if coin_toss <= epsilon:
            policy[k] = np.random.randint(len(k) - 1)
        policy[k] = np.argmax(v)
    return policy

def mc_control(env, num_episodes, alpha, gamma=1.0):
    nA = env.action_space.n
    # initialize empty dictionary of arrays
    Q = defaultdict(lambda: np.zeros(nA))
    policy = defaultdict(lambda: 0)
    # loop over episodes
    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        ## TODO: complete the function
        episode = generate_episode_from_policy(env, policy)
        epsilon = 0.05 + 1.0 / (2*i_episode + 1)
        length_episode = len(episode)
        # Yeah, I checked the homework solution for Part 1. This way looks better.
        states, actions, rewards = zip(*episode)
        returns = discount(rewards, gamma)
        for t in range(length_episode):
            state = states[t]
            action = actions[t]
            _return = returns[t]
            Q[state][action] += alpha*(_return - Q[state][action])
        policy = get_policy(Q, epsilon)
    return policy, Q


# obtain the estimated optimal policy and action-value function
policy, Q = mc_control(env, 500000, 0.01, gamma=0.9)

# obtain the corresponding state-value function
V = dict((k,np.max(v)) for k, v in Q.items())

# plot the state-value function
plot_blackjack_values(V)

# plot the policy
plot_policy(policy)
