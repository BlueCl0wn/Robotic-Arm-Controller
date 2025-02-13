import numpy as np
from .base_model import Model
import torch
import torch.nn as nn

class NeuralNetworkModel(Model, nn.Module):
    """
    Class representing a neural network model. Uses torch.
    """
    def __init__(self, input_size: int, output_size: int, hidden_layers: list|None = None):
        super(NeuralNetworkModel, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers

        # List of layers and actions functions.
        if hidden_layers is None:
            layers = [nn.Linear(input_size, output_size)]
            act_funcs = [nn.Softmax()]
        else:
            layers = ([nn.Linear(input_size, hidden_layers[0])]
                      +[nn.Linear(hidden_layers[i], hidden_layers[i+1]) for i in range(len(hidden_layers)-1)]
                      +[nn.Linear(hidden_layers[-1], output_size)])
            act_funcs = [nn.LeakyReLU()] * len(hidden_layers) + [nn.Softmax(dim=-1)] # List of activation functions.

        zipped = [elem for pair in zip(layers, act_funcs) for elem in pair] # Creates list with all layers and activation functions.
        #for i in range(0, len(zipped), 2):
            #torch.nn.init.xavier_uniform_(zipped[i].weight.T)
        self.linear = nn.Sequential(*zipped)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network.
        :param x: Input to the neural network in the form of a torch.Tensor.
        :return: Output of the neural network in the form of a torch.Tensor.
        """
        return self.linear(x)

    def make_decision(self, observation: np.ndarray) -> np.ndarray:
        """
        Main decision-making function of the neural network. Basically the same as self.forward but uses np.argmax on return of self.forward().

        :param observation: np.ndarray as input vector representing the observation state.
        :return: np.ndarray as output vector representing the decision.
        """

        # Creates torch.Tensor from passed np.ndarray.
        observation_tensor = torch.tensor(observation, dtype=torch.float32)

        output = self.forward(observation_tensor)
        #return np.array(np.argmax(output.detach().numpy()))
        p = output.detach().numpy()
        return np.random.choice(np.arange(4), p=p/np.sum(p))

    def get_parameters(self) -> np.ndarray:
        """
        Returns all parameters of the NN model as one flattened list.
        :return: Flattened list of all parameters of the NN model.
        """
        parameters = [param.view(-1) for param in self.linear.parameters()]  # Flatten all parameters
        return torch.cat(parameters).detach().cpu().numpy()

    def set_parameters(self, flat_params: list | np.ndarray | torch.Tensor) -> None:
        """
        Takes a one-dimensional / flat list and uses it to assign the parameters of the NN.

        :param flat_params: Flat list of parameters of the NN model.
                            Must have same size as the list gotten with 'self.flatten_parameters()'.
        :return: None
        """

        # Converting flat_params to type torch.tensor
        if type(flat_params) is np.ndarray:
            flat_params = torch.from_numpy(flat_params)
        elif type(flat_params) is list:
            flat_params = torch.tensor(flat_params)

        # Making sure that parameter flat_params has the correct length.
        total_numel = sum(param.numel() for param in self.linear.parameters())
        if len(flat_params) != total_numel:
            raise ValueError(f"Parameter flat_params must be of correct length. Should be of length {total_numel} in this case.")

        # Assign the parameters.
        current_position = 0
        for param in self.linear.parameters():
            # Determine number of data points in the parameters.
            param_size = param.numel()
            # Saves corresponding section of array.
            param.data.copy_(flat_params[current_position:current_position + param_size].view_as(param)) # type: ignore
            # Update current position.
            current_position += param_size


if __name__ == "__main__":
    model = NeuralNetworkModel(2, 1)
    flat = model.get_parameters()
    new = np.random.random(len(flat))
    print("flattened old: ", flat)
    print("new random values: ", new)
    model.set_parameters(new)
    print("test of new params", model.get_parameters())