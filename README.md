# Robotic-Arm-Controller

## authors
- **Marie-Luise Korn** [GitHub](https://github.com/markorn1612)  
  - Brandenburg University of Applied Sciences
- **Darek Petersen** [GitHub](https://github.com/BlueCl0wn) 
  - University of Hamburg  
- **Leonard Prince**  [Github](??)
  - ??

## Problem
https://robotics.farama.org/envs/fetch/reach/
The task in the environment is for a manipulator to move the end effector to a randomly selected position in the robot’s workspace and maintain it for an indefinite period of time.

The **Observation Space** is a dictionary consisting of 3 keys with information about the robot’s end effector state and goal: 
- 'observation': ndarray of shape (10,)
- 'desired_goal': ndarray, (3,)
- 'achieved_goal': ndarray, (3,)

The **Action Space** is a Box(-1.0, 1.0, (4,), float32) and thereby continuos.

## Solution
Our first approach was to use the DQN Algorithm and Prioritized Experience Replay (PER):
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
https://paperswithcode.com/method/prioritized-experience-replay 

The code included in the src directory depicts this approch. However, we run into some problems due to the fact that DQN cannot directly handle discrete action spaces, so we researched for an algorithm that is better suited for continuos action spaces. We found the Deep Deterministic Policy Gradient (DDPG) and Truncated Quantile Critics (TQC) algorithms: 
https://www.findingtheta.com/blog/mastering-robotic-manipulation-with-reinforcement-learning-tqc-and-ddpg-for-fetch-environments

Instead of one neural network that outputs the q-values for every possible action this approach uses two networks: Actor and Critic. The Actor Network takes a state as an input and outputs only one optimal action. The Critic Network takes state and action as the input and outputs the q-value. 

The Structure of the Neural Networks as well as the TQC algorithm are implemented in the stable_baselines3 library: 
https://sb3-contrib.readthedocs.io/en/master/modules/tqc.html

The File *main.py* uses this library to train and evaluate the Robotic Arm for the FetchReach-v4 Environment. It is crucial to install gymnasium robotics directly from Github to be able to access v4. 

Currently we are working on adapting the following repository code for our project in order to further understand the implementation of the TQC algorithm:
https://github.com/SamsungLabs/tqc_pytorch/tree/master 

## Tensorboard
To use the Tensorboard run:
'tensorboard --logdir=logs'

## Citations

https://github.com/Lord225/stochastic-gym-solver

@article{QN_Learning_paper,
author = {Watkins, Christopher and Dayan, Peter},
year = {1992},
month = {05},
pages = {279-292},
title = {Technical Note: Q-Learning},
volume = {8},
journal = {Machine Learning},
doi = {10.1007/BF00992698}
}

example implementation of q-learning with torch
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

@misc{fetch_envs,
  Author = {Matthias Plappert and Marcin Andrychowicz and Alex Ray and Bob McGrew and Bowen Baker and Glenn Powell and Jonas Schneider and Josh Tobin and Maciek Chociej and Peter Welinder and Vikash Kumar and Wojciech Zaremba},
  Title = {Multi-Goal Reinforcement Learning: Challenging Robotics Environments and Request for Research},
  Year = {2018},
  Eprint = {arXiv:1802.09464},
}
