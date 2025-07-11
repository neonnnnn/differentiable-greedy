import torch
from torch import nn

from ..constraint import BaseConstraint
from ..selector import SGASelector
from ..utility import BaseUtility
from .base import BaseLitModule


class DFLContinuous(BaseLitModule):
    """Decision-focused learning method by continuous relaxation and unrolling.

    Attributes:
        model (nn.Module): The predictive model.
        utility (BaseUtility): The utility instance.
        constraint (BaseConstraint): The constraint instance.
        lr (float): The learning rate parameter for the Adam optimizer
            used to train the `model`.
            Default: 1e-3
        lr_sga (float): The learning rate parameter for the SGD optimizer
            for the decision.
            Default: 1e-1

    Reference:
        Bryan Wilder, Bistra Dilkina, and Milind Tambe.
        Melding the Data-Decisions Pipeline: Decision-Focused Learning
        for Combinatorial Optimization.
        In Proc AAAI. 2019.
    """

    def __init__(
        self,
        model: nn.Module,
        utility: BaseUtility,
        constraint: BaseConstraint,
        lr: float = 1e-3,
        lr_sga: float = 1e-1,
    ):
        super().__init__(model, utility, constraint, lr)
        self.loss_fn = nn.functional.mse_loss
        self.lr_sga = lr_sga
        self.selector_eval = SGASelector(self.utility, self.constraint, lr=lr_sga)

    def training_step(self, batch, batch_idx):
        features, theta_true = batch
        theta_hat = self.model(features)
        self.utility.set_params(theta_hat)
        is_train = self.selector_eval.training
        self.selector_eval.train()
        X = self.selector_eval(theta_hat)
        self.utility.set_params(theta_true)
        utility = self.utility(X)
        loss = -torch.mean(utility)
        self.log("train_loss", loss)
        self.log("train_utility", -loss)
        if not is_train:
            self.selector_eval.eval()
        return loss

    def validation_step(self, batch, batch_idx):
        _, theta_true = batch
        with torch.enable_grad():
            X_continuous = self.predict_step(batch, batch_idx)
        # rounding
        X_discrete = self.constraint.project_discrete(X_continuous)
        self.utility.set_params(theta_true)
        utility = self.utility(X_discrete)
        self.log_dict({"val_utility": torch.mean(utility)})

    def test_step(self, batch, batch_idx):
        _, theta_true = batch
        with torch.enable_grad():
            X_continuous = self.predict_step(batch, batch_idx)
        # rounding
        X_discrete = self.constraint.project_discrete(X_continuous)
        self.utility.set_params(theta_true)
        utility = self.utility(X_discrete)
        self.log_dict({"test_utility": torch.mean(utility)})
