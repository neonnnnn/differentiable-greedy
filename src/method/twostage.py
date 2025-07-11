import torch
from torch import nn

from ..constraint import BaseConstraint
from ..selector import FWSelector, GreedySelector, RandomSelector, SGASelector
from ..utility import BaseUtility
from .base import BaseLitModule


class TwoStage(BaseLitModule):
    """
    Two-stage (predict-then-optimize) method.

    This method first trains the predictive model by minimizing the estimation loss,
    and then outputs decisions by optimizing a non-differentiable discrete function.

    Attributes:
        model (nn.Module): The predictive model.
        utility (BaseUtility): The utility instance.
        constraint (BaseConstraint): The constraint instance.
        lr (float): The learning rate parameter for the Adam optimizer
            used to train the `model`.
            Default: 1e-3
        lr_sga (float): The step size parameter for the SGD optimizer
            used to optimize the decision.
            Default: 1e-1
        selector (str): The selection method: 'greedy', 'fw', 'random', or 'sga'.
            Default: 'greedy'
    """

    def __init__(
        self,
        model: nn.Module,
        utility: BaseUtility,
        constraint: BaseConstraint,
        lr: float = 1e-3,
        lr_sga: float = 1e-1,
        selector: str = "greedy",
    ):
        super().__init__(model, utility, constraint, lr)
        self.loss_fn = nn.functional.mse_loss
        self.lr_sga = lr_sga
        if selector == "greedy":
            self.selector_eval = GreedySelector(self.utility, self.constraint)
        elif selector == "fw":
            self.selector_eval = FWSelector(self.utility, self.constraint)
        elif selector == "sga":
            self.selector_eval = SGASelector(self.utility, self.constraint, lr=lr_sga)
        elif selector == "random":
            self.selector_eval = RandomSelector(self.utility, self.constraint)
        else:
            raise ValueError("'selector' must be 'greedy', 'fw', 'sga' or 'random'.")

    def training_step(self, batch, batch_idx):
        features, theta_true = batch
        batch_size = len(features)
        theta_hat = self.model(features)
        loss = self.loss_fn(theta_hat, theta_true, reduction="sum") / batch_size
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        X = self.selector_eval(theta_hat.detach())
        self.utility.set_params(theta_true)
        utility = self.utility(X)
        self.log("train_utility", torch.mean(utility))
        return loss

    def validation_step(self, batch, batch_idx):
        _, theta_true = batch
        if isinstance(self.selector_eval, (FWSelector, SGASelector)):
            with torch.enable_grad():
                X_continuous = self.predict_step(batch, batch_idx)
            # rounding
            X_discrete = self.constraint.project_discrete(X_continuous)
        elif isinstance(self.selector_eval, (RandomSelector, GreedySelector)):
            X_discrete = self.predict_step(batch, batch_idx)
        self.utility.set_params(theta_true)
        utility = self.utility(X_discrete)
        self.log_dict({"val_utility": torch.mean(utility)})

    def test_step(self, batch, batch_idx):
        _, theta_true = batch
        if isinstance(self.selector_eval, (FWSelector, SGASelector)):
            with torch.enable_grad():
                X_continuous = self.predict_step(batch, batch_idx)
            # rounding
            X_discrete = self.constraint.project_discrete(X_continuous)
        elif isinstance(self.selector_eval, (RandomSelector, GreedySelector)):
            X_discrete = self.predict_step(batch, batch_idx)
        self.utility.set_params(theta_true)
        utility = self.utility(X_discrete)
        self.log_dict({"test_utility": torch.mean(utility)})
