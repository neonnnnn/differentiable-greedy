import torch

from .base import BaseUtility


class InfluenceUtility(BaseUtility):
    """Influence utility for bipartite graphs."""

    def get_n_items(self) -> int:
        return self.theta.shape[1]

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Computes the utility of the selected items X. Defined as
        f(X, theta) = sum_{t in users} (1 - prod_{v in items} (1 - X_v * theta_{v,t}))

        Args:
            X (torch.Tensor[float], shape: (batch_size, n_items)):
                A tensor of selection probabilities for each item.
        Returns:
            utility (torch.tensor[float], shape: (batch_size, )): utility tensor.
        """
        one_minus_theta = 1.0 - X[:, :, None] * self.theta
        prob_influenced = 1.0 - torch.prod(
            one_minus_theta, dim=1
        )  # (batch_size, n_users)
        utility = torch.sum(prob_influenced, dim=1)
        return utility

    def set_params(self, theta: torch.Tensor):
        self.theta = theta

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
        one_minus_theta = 1.0 - X[:, :, None] * self.theta
        prob_not_influenced = torch.clip(
            torch.prod(one_minus_theta, dim=1), 0.0, 1.0
        )  # (batch_size, n_users)
        gains = torch.sum(self.theta * prob_not_influenced[:, None, :], dim=2)
        gains = torch.where(candidates, gains, 0.0)
        return gains
