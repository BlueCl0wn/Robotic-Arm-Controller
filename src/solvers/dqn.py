import torch
import random
from src.models import NeuralNetworkModel
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
        new = np.random.normal(loc=-0.4, scale=0.2, size=len(flat)) # mu and sigma chosen by gut-feeling
        policy_net.set_parameters(new)

        # target network
        flat = target_net.get_parameters()
        # new = np.random.random(len(flat))
        new = np.random.normal(loc=-0.4, scale=0.2, size=len(flat)) # mu and sigma chosen by gut-feeling
        target_net.set_parameters(new)


    optimizer = optim.AdamW(policy_net.parameters(), lr=params.LR, amsgrad=True)
    memory = ReplayMemory(params.replay_mem_size)

    return policy_net, target_net, optimizer, memory


#steps_done = 0



def select_action(env: gym.Env, state, policy_net, target_net, optimizer, memory, i, params: argparse.Namespace, logger=None):
    """
    TODO add better comments and type hinting
    TODO: remove unused parameters? might make parsing a bit more annoying
    :param env:
    :param state:
    :param policy_net:
    :param target_net:
    :param optimizer:
    :param memory:
    :param i:
    :param params:
    :param logger:
    :return:
    """
    sample = random.random()
    eps_threshold = params.EPS_END + (params.EPS_START - params.EPS_END) * \
        math.exp(-1. * i / params.EPS_DECAY)
    i += 1

    if logger is not None and (i % 10 == 0):
        logger.add_scalar("epsilon_threshold", eps_threshold, i)

    if sample > eps_threshold:
        with (torch.no_grad()):
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            # print("policy")
            return policy_net(state).squeeze()

            #return o.max(1).indices.view(1, 1)

    else:
        #print("random")
        return torch.tensor(env.action_space.sample(), device=params.device, dtype=torch.float32)
        #return env.action_space.sample()

def optimize_model(policy_net, target_net, optimizer, memory, params: argparse.Namespace):
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
    transitions = memory.sample(params.BATCH_SIZE)
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
    action_batch = torch.stack(batch.action).view(params.BATCH_SIZE, -1)

    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    #print("shape(state_batch) =", state_batch.shape)
    #print("shape(action_batch) = ", action_batch.shape)
    state_action_values = policy_net(state_batch).gather(1, action_batch.long()) # This is from the example project. Here it makes a mess. Is it okay to just remove?

    #print("shape(policy_net(state_batch)) = ", policy_net(state_batch).shape)
    #print("shape(state_action_values) = ", state_action_values.shape)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(params.BATCH_SIZE, device=params.device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * params.GAMMA) + reward_batch
    #print("shape(expected_state_action_values) = ", expected_state_action_values.shape)

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    #print("type of loss : " , loss.shape)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()