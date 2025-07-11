import lightning as L
import torch
from torch import nn, optim

from ..constraint import BaseConstraint
from ..selector import GreedySelector
from ..utility import BaseUtility


class BaseLitModule(L.LightningModule):
    """Base class for all training method implementations."""

    def __init__(
        self,
        model: nn.Module,
        utility: BaseUtility,
        constraint: BaseConstraint,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.utility = utility
        self.constraint = constraint
        self.selector_eval = GreedySelector(self.utility, self.constraint)

    def training_step(self, batch, batch_idx):
        features, theta_true = batch
        theta_hat = self.model(features)
        loss = self.loss_fn(theta_hat, theta_true)
        self.log("train_loss", loss)
        X = self.predict_step(batch, batch_idx)
        self.utility.set_params(theta_true)
        utility = self.utility(X)
        self.log("train_utility", torch.mean(utility))
        return loss

    def predict_step(self, batch, batch_idx):
        features, _ = batch
        theta_hat = self.model(features)
        self.selector_eval.utility.set_params(theta_hat.detach())
        X_eval = self.selector_eval(theta_hat.detach())
        return X_eval

    def validation_step(self, batch, batch_idx):
        _, theta_true = batch
        X = self.predict_step(batch, batch_idx)
        self.utility.set_params(theta_true)
        utility = self.utility(X)
        self.log_dict({"val_utility": torch.mean(utility)})

    def test_step(self, batch, batch_idx):
        _, theta_true = batch
        X = self.predict_step(batch, batch_idx)
        self.utility.set_params(theta_true)
        utility = self.utility(X)
        self.log_dict({"test_utility": torch.mean(utility)})

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer
