import numpy as np
from collections import namedtuple, deque
import random
from itertools import chain

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class PrioritizedReplayMemory(object):

    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_decay=100000, eps=1e-5):
        """
        :param alpha: Priority exponent. Determines how much prioritization is used (0 = uniform, 1 = full prioritization).
        :param beta_start: Initial value for beta. Controls the strength of bias correction for the priorities.
        :param beta_decay: How quickly beta increases during training. Ensures that bias correction becomes stronger over time.
        :param eps: Small value to avoid assigning zero priority to any experience.
        """
        self.memory = deque([], maxlen=capacity)
        self.priorities = []
        self.alpha = alpha
        self.beta = beta_start
        self.beta_increment = (1.0 - beta_start) / beta_decay
        self.eps = eps
        
    def push(self, *args):
        """Save a transition"""
        max_priority = max(self.priorities, default=1.0)  # New transitions get max priority
        self.memory.append(Transition(*args))
        self.priorities.append(max_priority)

    def sample(self, batch_size):
        """
        Sample a batch of experiences based on their priorities.

        :param batch_size: Number of experiences to sample.
        :return: A batch of samples, their indices, and the corresponding importance-sampling weights.
        """

        # Compute probabilities (P_i = p_i^α / Σ p_k^α)
        priorities = np.array(self.priorities, dtype=np.float32)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)

        samples = [self.memory[i] for i in indices]

        # Calculate Importance-Sampling weights to correct bias during training:
        # (bias due to over-sampling of high-priority transitions)
        weights = (len(self.memory) * probabilities[indices]) ** -self.beta
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32) # not sure if needed
        self.beta = min(1.0, self.beta + self.beta_increment)

        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        """
        Update the priorities for the given experiences.

        :param indices: Indices of the experiences whose priorities need to be updated.
        :param priorities: The new priorities for the experiences at the specified indices.
        """
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio + self.eps  

    def __len__(self):
        """ Return the number of experiences currently stored in memory."""
        return len(self.memory)

    def get_list(self) -> list:
        """
        Returns the deque as a flattened list. Maybe useful in order to log the state of the ReplayMemory in the Tensorboard.
        :return: List of all transitions stored in memory.
        """
        #print("deque as list:")
        #return list(chain.from_iterable(self.memory))
        raise NotImplementedError
