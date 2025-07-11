import torch
from torch import nn


class NNModel(nn.Module):
    """A two-layer neural network model.

    Args:
        input_dim (int): The dimension of the input layer.
        hidden_dim (int): The dimension of the hidden layer.
            Default: 200
        output (str): The activation function for the output layer.
            Must be one of "sigmoid", "clip", or "clamp".
            Default: "clip"
    """

    def __init__(self, input_dim, hidden_dim=200, output="clip"):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        if output == "sigmoid":
            self.output_activation = torch.nn.functional.sigmoid
        elif output in ["clip", "clamp"]:
            self.output_activation = lambda x: torch.clamp(x, 0.0, 1.0)
            self.init_weights()
        else:
            raise ValueError("output must be 'sigmoid', 'clip', or 'clamp'.")

    def init_weights(self):
        """Initializes weights with a uniform distribution."""
        nn.init.uniform_(self.fc1.weight, 0.0, 0.01)
        nn.init.uniform_(self.fc2.weight, 0.0, 0.01)
        nn.init.uniform_(self.fc1.bias, 0.0, 0.01)
        nn.init.uniform_(self.fc2.bias, 0.0, 0.01)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return self.output_activation(x).squeeze()


class RandomModel(nn.Module):
    """A model that returns random predictions.

    This is intended for use as a baseline in comparative experiments.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        batch_size, n_movies, n_items, n_features = x.shape
        return torch.rand(batch_size, n_movies, n_items).to(x.device)
