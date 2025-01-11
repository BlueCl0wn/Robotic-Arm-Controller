import torch
import random
from models import NeuralNetworkModel
from .ReplayMemory import ReplayMemory, Transition
import torch.optim as optim
import argparse
import gymnasium as gym
import numpy as np
import math
import torch.nn as nn


"""
source: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""



def initiate_stuff(params: argparse.Namespace, random=True):
    """
    TODO add docstring
    :param params:
    :return:
    """
    #policy_net = DQN(n_observations, n_actions).to(params.device)
    policy_net = NeuralNetworkModel(params.input_size, params.output_size, params.hidden_layers).to(params.device)
    #target_net = DQN(n_observations, n_actions).to(params.device)
    target_net = NeuralNetworkModel(params.input_size, params.output_size, params.hidden_layers).to(params.device)
    target_net.load_state_dict(policy_net.state_dict())

    # Initializing the policy and target networks with random parameters
    # I chose to initialize them with different sets of random parameters to increase the exploration.
    # If random is set to False all parameters are zero.
    if random:
        # policy network
        flat = policy_net.get_parameters()
        # new = np.random.random(len(flat))
        new = np.random.normal(loc=0, scale=0.5, size=len(flat)) # sigma=1.5 is chosen by gut-feeling
        policy_net.set_parameters(new)

        # target network
        flat = target_net.get_parameters()
        # new = np.random.random(len(flat))
        new = np.random.normal(loc=0, scale=0.5, size=len(flat)) # sigma=1.5 is chosen by gut-feeling
        target_net.set_parameters(new)


    optimizer = optim.AdamW(policy_net.parameters(), lr=params.LR, amsgrad=True)
    memory = ReplayMemory(params.replay_mem_size)

    return policy_net, target_net, optimizer, memory


#steps_done = 0



def select_action(env: gym.Env, state, policy_net, target_net, optimizer, memory, i, params: argparse.Namespace, logger=None):
    """
    TODO add better comments and type hinting
    TODO: remove unused parameters? might make parsing a bit more annoying
    :param env: The environment to interact with.
    :param state: Current state of the environment.
    :param policy_net: The Q-network used to select actions.
    :param target_net: Target Q-network for stability in training.
    :param optimizer: Optimizer used for updating the policy_net.
    :param memory: Replay memory (supports Prioritized Experience Replay).
    :param i: Current step or episode index.
    :param params: Hyperparameters for epsilon decay and replay settings.
    :param logger: Logger object for monitoring metrics.
    
    Returns: 
    torch.Tensor: Selected action for the current state.
    """
    sample = random.random()
    eps_threshold = params.EPS_END + (params.EPS_START - params.EPS_END) * \
        math.exp(-1. * i / params.EPS_DECAY)
    i += 1

    if logger is not None and (i % 10 == 0):
        logger.add_scalar("epsilon_threshold", eps_threshold, i)
    
    # Epsilon-greedy action selection
    if sample > eps_threshold:
        with (torch.no_grad()):
            q_values = policy_net(state)
            action = q_values.max(1)[1].view(1, 1) # not sure if we need .view(1,1) to reshape the action

    else:
        # Select a random action
        action = torch.tensor(env.action_space.sample(), device=params.device, dtype=torch.float32)

    return action

def optimize_model(policy_net, target_net, optimizer, memory, gamma, params: argparse.Namespace):
    """
    TODO: add better comments and type hinting
    :param policy_net:
    :param target_net:
    :param optimizer:
    :param memory:
    :param params:
    :return:
    """
    if len(memory) < params.BATCH_SIZE:
        return


    transitions, indices, weights = memory.sample(params.BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=params.device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    # action_batch = torch.cat(batch.action)
    action_batch = torch.cat(batch.action).long().unsqueeze(1)

    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    current_q_values = policy_net(state_batch).gather(1, action_batch.long()) 

    
    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_q_values = torch.zeros(params.BATCH_SIZE, device=params.device)
    with torch.no_grad():
        next_q_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    target_q_values = (next_q_values * params.GAMMA) + reward_batch
    
    # Compute weighted loss
    weights = torch.tensor(weights, dtype=torch.float32, device=params.device)
    loss = (weights * nn.functional.smooth_l1_loss(current_q_values, target_q_values.unsqueeze(1), reduction='none')).mean()

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)    # TODO maybe adjust the clipping range
    optimizer.step()

    # Update priorities in Replay Memory
    td_errors = torch.abs(current_q_values - target_q_values.unsqueeze(1)).detach().cpu().numpy()
    memory.update_priorities(indices, td_errors.flatten())