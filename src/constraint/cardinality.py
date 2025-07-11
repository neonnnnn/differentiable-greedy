import torch

from .base import BaseConstraint


def project_uniform_matroid_boundary(
    x: torch.Tensor, k: torch.Tensor | float = 1.0, c: torch.Tensor | float = 1.0
) -> torch.Tensor:
    """
    Projects x onto the set {y: 0 <= y <= 1/c, c^T x = k}.

    Args:
        x (torch.Tensor, shape: (batch_size, n_items)): input tensor.
        k (torch.Tensor, shape: (batch_size, )): constraint tensor.
        c (torch.Tensor, shape: (batch_size, n_items)): constraint tensor.

    Returns:
        x_projected (torch.Tensor, shape: (batch_size, n_items)): projected tensor.
    """
    batch_size, n_items = x.shape
    device = x.device
    if isinstance(c, (int, float)):
        c = torch.ones((batch_size, n_items), device=device) * c
    if isinstance(k, (int, float)):
        k = torch.ones((batch_size, 1), device=device) * k
    if len(k) == 1:
        k = k.view(-1, 1)

    alpha_upper = x / c
    alpha_lower = (x * c - 1) / c**2
    S = torch.hstack([alpha_lower, alpha_upper])
    c_squared = torch.hstack([-(c**2), c**2])
    S, indices = torch.sort(S, dim=1, descending=False)
    c_squared = torch.take_along_dim(c_squared, indices, dim=1)
    h_prime = torch.zeros((batch_size, n_items * 2), device=device)
    h_prime[:, 0] = n_items
    # compute h by cumsum
    diff_alpha = S[:, 1:] - S[:, :-1]
    m = torch.cumsum(c_squared[:, :-1], dim=1)
    h_prime[:, 1:] = diff_alpha * m
    h_prime = torch.cumsum(h_prime, dim=1)
    h_prime = torch.hstack([h_prime, torch.zeros(batch_size, 1, device=device)])
    S = torch.hstack([S, S[:, -1].view(-1, 1)])
    # find pivot
    pivot_prime = torch.sum(h_prime >= k.view(-1, 1), dim=1, keepdims=True)
    # compute alpha_star
    h = torch.take_along_dim(h_prime, pivot_prime - 1, dim=1)
    h_prime = torch.take_along_dim(h_prime, pivot_prime, dim=1)
    alpha = torch.take_along_dim(S, pivot_prime - 1, dim=1)
    alpha_i = torch.take_along_dim(S, pivot_prime, dim=1)
    alpha_star = (alpha_i - alpha) * (h - k) / (h - h_prime) + alpha
    # project
    return torch.clamp(x - alpha_star * c, torch.zeros_like(x), 1.0 / c)


class Cardinality(BaseConstraint):
    """Implements a cardinality constraint, limiting the number of selected items.

    This constraint ensures that the total number of selected items does not
    exceed a given maximum `k`.

    Attributes:
        k (int): The maximum number of items that can be selected.
    """

    def __init__(self, max_cardinality: int):
        self.max_cardinality = max_cardinality

    def get_candidates(self, X: torch.Tensor) -> torch.Tensor:
        """Returns candidate items.

        An item is a candidate if it is not already in the current selection `X`,
        and adding it would not violate the cardinality constraint (i.e., the
        total number of selected items is less than `k`).

        Args:
            X (torch.Tensor[bool], shape: (batch_size, n_items)):
                A boolean tensor representing the current selection.

        Returns:
            candidates (torch.Tensor[bool], shape: (batch_size, n_items)):
                A boolean tensor indicating candidate items.
        """
        n_selected_items = torch.sum(X, dim=1, keepdims=True)
        return torch.where(
            n_selected_items >= self.max_cardinality, False, torch.logical_not(X)
        )

    def project_continuous(self, X: torch.Tensor) -> torch.Tensor:
        """Projects a continuous selection onto the cardinality-constrained polytope.

        Args:
            X (torch.Tensor[float], shape: (batch_size, n_items)):
                A tensor of selection probabilities.

        Returns:
            X_projected (torch.Tensor[float], shape: (batch_size, n_items)):
                A tensor of selection probabilities that satisfies the constraint.
        """
        return project_uniform_matroid_boundary(X, k=self.max_cardinality, c=1.0)

    def project_discrete(self, X: torch.Tensor) -> torch.Tensor:
        """Projects a continuous selection onto a discrete one satisfying the
        cardinality constraint.

        This is done by selecting the top-k items with the highest probabilities.

        Args:
            X (torch.Tensor[float], shape: (batch_size, n_items)):
                A tensor of selection probabilities.

        Returns:
            X_projected (torch.Tensor[float], shape: (batch_size, n_items)):
                A discrete (0 or 1) tensor with at most `k` items selected.
        """
        return self.solve_frank_wolfe(X)

    def solve_frank_wolfe(self, grad: torch.Tensor) -> torch.Tensor:
        """Solves the Frank-Wolfe step for the cardinality constraint.

        This selects the top-k items with the largest gradients.

        Args:
            grad (torch.Tensor[float], shape: (batch_size, n_items)):
                The gradient tensor.

        Returns:
            v (torch.Tensor[float], shape: (batch_size, n_items)):
                A discrete tensor representing the k items with the highest gradients.
        """
        _, indices = torch.topk(grad, k=self.max_cardinality, dim=1)
        v = torch.zeros_like(grad)
        v.scatter_(1, indices, 1.0)
        return v
