import numpy as np
from .base_model import Model

class RandomModel(Model):
    def make_decision(self, size) -> np.ndarray:
        return np.random.uniform(-1, 1, size)