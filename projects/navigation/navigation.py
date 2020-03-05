from unityagents import UnityEnvironment
import numpy as np
from utils import load_cfg
from ddqn import ddqn
from ddqn_agent import Agent
import matplotlib.pyplot as plt

# Load configuration from YAML
cfg = load_cfg()

# Define global configuration variables
env_path = cfg["Environment"]["Filepath"]
brain_index = cfg["Agent"]["Brain_index"]

def load_environment(env_path=env_path, brain_index=brain_index):
    """Load Unity environment

    Params
    ======
    env_path (str): The path of the Unity executable
    brain_index (int): The index of the agent we want to act
    """
    env = UnityEnvironment(file_name=env_path)

    # get the default brain
    brain_name = env.brain_names[brain_index]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents in the environment
    print('Number of agents:', len(env_info.agents))

    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)

    # examine the state space
    state = env_info.vector_observations[0]
    print('States look like:', state)
    state_size = len(state)
    print('States have length:', state_size)
    return env, state_size, action_size, brain_name

def save_weights(agent, fname="checkpoint.pth"):
    """Save weights of PyTorch model

    Params
    ======
    agent (Agent class): A Double-DQN Agent
    fname (str): file name of weights to write to disk
    """
    torch.save(agent.qnetwork_local.state_dict(), fname)

def plot_scores(scores):
    """Plot scores from training episodes

    Params
    ======
    scores (list<float>): episode scores during training
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

def main():
    """Main function that ties it all together

    Description
    ===========
    Loads environment, instantiates agent, trains agent, saves weights,
    and plots the scores.
    """
    # Load the UnityMLAgents environment
    env, state_size, action_size, brain_name = load_environment()
    # Instantiate the agent
    agent = Agent(state_size=state_size, action_size=action_size, seed=0)
    # Use Double-DQN to train the agent
    scores = ddqn(env, agent)
    # Persist the weights of the learned model
    save_weights(agent)
    # Plot the scores from the training episodes.
    plot_scores(scores)
    # Clean up the workspace once finished.
    env.close()

if __name__ == "__main__":
    main()
