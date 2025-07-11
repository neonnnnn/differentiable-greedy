import torch

from ..utility.base import BaseUtility


class InfluenceUtilitySlow(BaseUtility):
    def get_n_items(self) -> int:
        return self.theta.shape[1]

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        batch_size, n_items, n_users = self.theta.shape
        utility = torch.zeros(
            batch_size,
        )
        for batch in range(batch_size):
            for user in range(n_users):
                prob_not_influenced = 1.0
                for item in range(n_items):
                    prob_not_influenced *= (
                        1.0 - X[batch, item] * self.theta[batch, item, user]
                    )
                utility[batch] += 1.0 - prob_not_influenced
        return utility

    def set_params(self, theta: torch.Tensor):
        self.theta = theta

    def compute_marginal_gain(
        self, X: torch.Tensor, candidates: torch.Tensor
    ) -> torch.Tensor:
        batch_size, n_items, n_users = self.theta.shape
        gains = torch.zeros(batch_size, n_items)
        for batch in range(batch_size):
            for user in range(n_users):
                prob_not_influenced = 1.0
                for item in range(n_items):
                    if X[batch, item]:
                        prob_not_influenced *= 1.0 - self.theta[batch, item, user]
                for item, cand in enumerate(candidates[batch]):
                    if cand:
                        coef = 1.0 - self.theta[batch, item, user]
                        gains[batch, item] += (1.0 - prob_not_influenced * coef) - (
                            1.0 - prob_not_influenced
                        )  # prob_not_influenced * (1.0 - coef)
        return gains
