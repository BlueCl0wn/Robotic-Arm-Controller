from collections import namedtuple, deque
import random
from itertools import chain

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def get_list(self) -> list:
        """
        Returns the deque as a flattened list. Maybe useful in order to log the state of the ReplayMemory in the Tensorboard.
        :return:
        """
        #print("deque as list:")
        #return list(chain.from_iterable(self.memory))
        raise NotImplementedError
