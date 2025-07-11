import torch

from ..constraint import BaseConstraint
from ..utility import BaseUtility
from .base import BaseSelector


class FWSelector(BaseSelector):
    """Selects items by continuous (Frank-Wolfe) greedy selection.

    Attributes:
        utility (BaseUtility): The utility instance to be maximized.
        constraint (BaseConstraint): The constraint instance.
        delta (float): The step size parameter for the Frank-Wolfe optimization.
            default: 0.1

    Reference:
        Gruia Calinescu et al.
        Maximizing a monotone submodular function subject to a matroid constraint
        XIAM Journal on Computing. 2011.
    """

    def __init__(
        self, utility: BaseUtility, constraint: BaseConstraint, delta: float = 0.1
    ):
        super().__init__(utility, constraint)
        self.delta = delta

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        """Selects items using the continuous (Frank-Wolfe) greedy algorithm.

        Args:
            theta (torch.tensor[float], shape: (batch_size, *parameter_size)):
                The parameter tensor of the utility function.

        Returns:
            X (torch.Tensor[float], shape: (batch_size, n_items)):
                A tensor of selection probabilities for each item.
        """
        batch_size = len(theta)
        device = theta.device
        self.utility.set_params(theta)
        n_items = self.utility.get_n_items()
        # initialize X
        X = torch.zeros(batch_size, n_items, device=device, requires_grad=True)
        t = 0.0
        self.utility.set_params(theta)
        # optimize X by Frank-Wolfe algorihm
        while t < 1.0:
            delta = min(self.delta, 1.0 - t)
            utility = self.utility(X)
            grad = torch.autograd.grad(
                torch.sum(utility), X, create_graph=self.training
            )[0]
            v = self.constraint.solve_frank_wolfe(grad)
            X = X + delta * v
            t += self.delta
        if self.training:  # output continuous relaxed tensor
            return X
        else:
            return self.constraint.project_discrete(X)
