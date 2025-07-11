from abc import ABCMeta, abstractmethod

import torch


class BaseUtility(torch.nn.Module, metaclass=ABCMeta):
    """Abstract base class for a utility function.

    Inherits from torch.nn.Module.
    """

    @abstractmethod
    def get_n_items(self) -> int:
        """Gets the number of items from the parameters theta.

        Returns:
            int: The number of items.
        """
        pass

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return self.forward(X)

    @abstractmethod
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Computes the utility.

        Args:
            X (torch.Tensor[float], shape: (batch_size, n_items)):
                A tensor of selection probabilities for each item.
        Returns:
            torch.Tensor (shape: (batch_size, )): The utility value.
        """
        pass

    @abstractmethod
    def compute_marginal_gain(
        self, X: torch.Tensor, candidates: torch.Tensor
    ) -> torch.Tensor:
        """Computes the marginal gain for candidate items.

        The marginal gain is gain(X, v) := f(X + v) - f(X) for v not in X.

        Args:
            X (torch.Tensor[float], shape: (batch_size, n_items)):
                A tensor of selection probabilities for each item.
            candidates (torch.Tensor[bool], shape: (batch_size, n_items)):
                A boolean tensor indicating candidate items.
        Returns:
            gain (torch.Tensor[float], shape: (batch_size, n_items)):
                A tensor of marginal gains for each candidate item.
        """
        pass

    @abstractmethod
    def set_params(self):
        """Set parameters and initializes caches."""
        pass
