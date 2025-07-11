import torch
from torch import nn

from ..constraint import BaseConstraint
from ..selector import SmoothedGreedySelector
from ..utility import BaseUtility
from .base import BaseLitModule


class ReinforceLoss(nn.Module):
    """
    REINFORCE Loss module.
    """

    def __init__(
        self,
        utility: BaseUtility,
        constraint: BaseConstraint,
        n_sampling: int,
        epsilon: float,
        variance_reduction=True,
    ):
        """
        Args:
            utility (BaseUtility): The utility instance.
            constraint (BaseConstraint): The constraint instance.
            n_sampling (int): The number of sampling.
            epsilon (float): The temperature parameter for SmoothedGreedySelector.
            variance_reduction (bool): Whether using baseline technique or not.
                default: True
        """
        super().__init__()
        self.utility = utility
        self.constraint = constraint
        self.n_sampling = n_sampling
        self.epsilon = epsilon
        self.variance_reduction = variance_reduction

        self.selector = SmoothedGreedySelector(
            self.utility, self.constraint, epsilon=self.epsilon
        )

    def forward(
        self, theta_hat: torch.Tensor, theta_true: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the approximated loss.

        Args:
            theta_hat (torch.Tensor, shape: (batch_size, *parameter_size)):
                The estimated parameter.
            theta_true (torch.Tensor, shape: (batch_size, *parameter_size):
                The true parameter.

        Returns:
            loss (torch.Tensor, shape: (1, )): the approximated loss.
            utility (torch.Tensor, shape: (1, )): the approximated utility.
        """
        rewards = []
        log_probs = []
        is_train = self.selector.training
        if not is_train:
            self.selector.train()
        for _ in range(self.n_sampling):
            self.utility.set_params(theta_hat)
            X, log_prob = self.selector(theta_hat)

            with torch.no_grad():
                self.utility.set_params(theta_true)
                reward = self.utility(X)

            rewards.append(reward)
            log_probs.append(log_prob)

        rewards_tensor = torch.stack(rewards)
        log_probs_tensor = torch.stack(log_probs)

        if self.n_sampling > 1 and self.variance_reduction:
            baselines = torch.mean(rewards_tensor, dim=0, keepdims=True)
        else:
            baselines = 0.0
        loss = -torch.mean((rewards_tensor - baselines) * log_probs_tensor)
        utility = torch.mean(rewards_tensor)
        if not is_train:
            self.selector.eval()
        return loss, utility


class DFLSmoothedGreedy(BaseLitModule):
    """Decision-focused learning method using a smoothed greedy algorithm.

    Attributes:
        model (nn.Module): The predictive model.
        utility (BaseUtility): The utility instance.
        constraint (BaseConstraint): The constraint instance.
        epsilon (float): The temperature parameter.
            Default: 0.2
        n_sampling (int): The number of samples for REINFORCE.
            Default: 10
        lr (float): The learning rate parameter for the Adam optimizer
            used to train the `model`.
            Default: 1e-3
        variance_reduction (bool): Whether to use the baseline technique.
            Default: True


    Reference
        Shinsaku Sakaue.
        Differentiable Greedy Algorithm for Monotone Submodular Maximization:
        Guarantees, Gradient Estimators, and Applications.
        In Proc AISTATS. 2021.
    """

    def __init__(
        self,
        model: nn.Module,
        utility: BaseUtility,
        constraint: BaseConstraint,
        epsilon: float = 0.2,
        n_sampling: int = 10,
        lr: float = 1e-3,
        variance_reduction: bool = True,
    ):
        super().__init__(model, utility, constraint, lr)
        self.loss_fn = ReinforceLoss(
            self.utility, self.constraint, n_sampling, epsilon, variance_reduction
        )

    def training_step(self, batch, batch_idx):
        features, theta_true = batch
        theta_hat = self.model(features)
        loss, utility = self.loss_fn(theta_hat, theta_true)
        self.log("train_loss", loss)
        self.log("train_utility", utility)
        return loss
