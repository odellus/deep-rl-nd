# Udacity Deep Reinforcement Learning Nano Degree
## Navigation Project


### Project Environment Details

<!-- The README describes the the project environment details (i.e., the state and action spaces, and when the environment is considered solved). -->

The dimensionality of the state space is 37 and the action space has dimensionality 4. The task in the environment is considered solved in our project whenever the score is greater than or equal to 13.

### Getting Started

<!-- The README has instructions for installing dependencies or downloading needed files. -->
The project depends on the software packages [UnityMLAgents](https://github.com/Unity-Technologies/ml-agents) and [PyTorch](https://pytorch.org/).

Instructions for installing UnityMLAgents can be found [here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md). We consider learning how to develop and build new environments in UnityMLAgents to be outside the scope of this project, but instructions on how to do so are located [here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Create-New.md).

Instructions for installing PyTorch can be found [here](https://pytorch.org/get-started/locally/).


### Instructions

<!-- The README describes how to run the code in the repository, to train the agent. For additional resources on creating READMEs or using Markdown, see here and here. -->
To train an agent simply run `python3.X navigation.py` where `X` is the subversion of python3 you have installed on your system. This will load the Unity environment, train your PyTorch agent, persist the learned model to disk at `checkpoint.pth`, and create a visualization of your learning curve with `matplotlib`.
