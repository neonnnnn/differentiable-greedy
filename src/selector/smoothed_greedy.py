import torch

from ..constraint import BaseConstraint
from ..utility import BaseUtility
from .base import BaseSelector


class SmoothedGreedySelector(BaseSelector):
    """Selects items using the smoothed greedy algorithm.

    Attributes:
        utility (BaseUtility): The utility instance to be maximized.
        constraint (BaseConstraint): The constraint instance.
        epsilon (float): The temperature parameter for the softmax function.
            Default: 0.2

    Reference:
        Shinsaku Sakaue.
        Differentiable Greedy Algorithm for Monotone Submodular Maximization:
        Guarantees, Gradient Estimators, and Applications.
        In Proc AISTATS. 2021.
    """

    def __init__(
        self, utility: BaseUtility, constraint: BaseConstraint, epsilon: float = 0.2
    ):
        super().__init__(utility, constraint)
        self.epsilon = epsilon

    def forward(
        self, theta: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Selects items using the smoothed greedy algorithm.

        Args:
            theta (torch.tensor[float], shape: (batch_size, *parameter_size)):
                The parameter tensor.

        Returns:
            If training:
                X (torch.Tensor[float], shape: (batch_size, n_items)):
                    A tensor representing the selected items (0 or 1).
                log_prob (torch.Tensor[float], shape: (batch_size)):
                    The log probability of the selection, log P(X, theta).
            If not training:
                X (torch.Tensor[float], shape: (batch_size, n_items)):
                    A tensor representing the selected items (0 or 1).
        """
        batch_size = len(theta)
        device = theta.device
        self.utility.set_params(theta)
        n_items = self.utility.get_n_items()
        X = torch.zeros(batch_size, n_items, device=device, dtype=torch.bool)
        if self.training:
            log_prob = torch.zeros(batch_size, device=device)
        for _ in range(n_items):
            candidates = self.constraint.get_candidates(X)
            # candidates are empty
            if not torch.any(torch.any(candidates, dim=1)).item():
                break
            gains = self.utility.compute_marginal_gain(X, candidates)
            gains = torch.where(candidates, gains, -torch.inf)
            probs = torch.nn.functional.softmax(gains / self.epsilon, dim=1)
            dist = torch.distributions.Categorical(probs)
            selected_items = dist.sample()  # shape: (batch_size, )
            selected_mask = torch.nn.functional.one_hot(
                selected_items, num_classes=n_items
            )

            X = X | selected_mask.to(torch.bool)
            if self.training:
                log_prob = log_prob + dist.log_prob(
                    selected_items
                )  # shape:(batch_size, )
        if self.training:
            return X.to(torch.float32), log_prob
        else:
            return X.to(torch.float32)
