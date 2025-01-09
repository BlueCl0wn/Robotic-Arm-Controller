import numpy as np


class Model:
    """
    A base model with all the functions needed for our implementation. Supposed to be overridden in actual model.

    """
    def make_decision(self, observation: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def get_parameters(self) -> dict:
        raise NotImplementedError()

    def set_parameters(self, parameters: dict):
        raise NotImplementedError()

    def get_model_penalty(self) -> float:
        return 0.0