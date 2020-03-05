# -*- coding: utf-8

# import gym
import random
import torch
import numpy as np
from collections import deque
from unityagents import UnityEnvironment
from utils import load_cfg

# Load configuration from YAML
cfg = load_cfg()

# Define global configuration variables
success = cfg["Environment"]["Success"]
brain_index = cfg["Agent"]["Brain_index"]
n_episodes = cfg["Training"]["Number_episodes"]
max_t = cfg["Training"]["Max_timesteps"]
eps_start = cfg["Training"]["Eps_start"]
eps_end = cfg["Training"]["Eps_end"]
eps_decay = cfg["Training"]["Eps_decay"]
train_mode = cfg["Training"]["Train_mode"]
score_window = cfg["Training"]["Score_window"]

def step_unity(
    env,
    action,
    brain_index=brain_index,
    brain_name=brain_name
    ):
    """Step Unity environment forward one timestep

    Params
    ======
        env (UnityEnvironment): The Unity environment to step forwards
        action (int): The action index to take during this timestep
        brain_index (int): The brain index of the agent we wish to act
        brain_name (str): The name of the brain we wish to act
    """
    env_info = env.step(action)[brain_name]
    state = env_info.vector_observations[brain_index]
    reward = env_info.rewards[brain_index]
    done = env_info.local_done[brain_index]
    return state, reward, done, env_info


def ddqn(
    env,
    agent,
    success=success
    brain_index=brain_index,
    n_episodes=n_episodes,
    max_t=max_t,
    eps_start=eps_start,
    eps_end=eps_end,
    eps_decay=eps_decay,
    train_mode=train_mode
    ):
    """Double Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=score_window)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    brain_name = env.brain_names[brain_index]
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=train_mode)[brain_name]
        state = env_info.vector_observations[brain_index]
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = step_unity(
                env,
                action,
                brain_index=brain_index,
                brain_name=brain_name
                )
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % score_window == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=success:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    return scores
