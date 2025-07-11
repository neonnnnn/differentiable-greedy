from abc import ABCMeta, abstractmethod

import torch


class BaseConstraint(metaclass=ABCMeta):
    """Base class for all constraint implementations."""

    @abstractmethod
    def get_candidates(self, X: torch.Tensor) -> torch.Tensor:
        """Returns candidate items that can be added to the current selection X
        without violating the constraint.

        Args:
            X (torch.Tensor[bool], shape: (batch_size, n_items)):
                A boolean tensor representing the current selection.

        Returns:
            candidates (torch.Tensor[bool], shape: (batch_size, n_items)):
                A boolean tensor where `True` indicates a candidate item.
        """
        pass

    @abstractmethod
    def project_continuous(self, X: torch.Tensor) -> torch.Tensor:
        """Projects a continuous selection tensor onto the feasible set defined
        by the relaxed constraint.

        Args:
            X (torch.Tensor[float], shape: (batch_size, n_items)):
                A tensor of selection probabilities.

        Returns:
            X_projected (torch.Tensor[float], shape: (batch_size, n_items)):
                A tensor of selection probabilities that satisfies the constraint.
        """
        pass

    @abstractmethod
    def project_discrete(self, X: torch.Tensor) -> torch.Tensor:
        """Projects a continuous selection tensor onto a discrete (0 or 1) tensor
        that satisfies the constraint.

        Args:
            X (torch.Tensor[float], shape: (batch_size, n_items)):
                A tensor of selection probabilities.

        Returns:
            X_projected (torch.Tensor[float], shape: (batch_size, n_items)):
                A discrete (0 or 1) tensor representing a valid selection.
        """
        pass

    @abstractmethod
    def solve_frank_wolfe(self, grad: torch.Tensor) -> torch.Tensor:
        """Solves the Frank-Wolfe optimization step for the relaxed constraint.
        This finds a vertex in the constraint polytope that maximizes the
        dot product with the given gradient.

        Args:
            grad (torch.Tensor[float], shape: (batch_size, n_items)):
                The gradient tensor from the utility function.

        Returns:
            v (torch.Tensor[float], shape: (batch_size, n_items)):
                A tensor representing the vertex that maximizes the linear objective.
        """
        pass
