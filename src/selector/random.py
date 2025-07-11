import torch

from .base import BaseSelector


class RandomSelector(BaseSelector):
    """Selects items randomly subject to constraints.

    Attributes:
        utility (BaseUtility): The utility instance to be maximized.
        constraint (BaseConstraint): The constraint instance.
    """

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        """Selects items randomly.

        Args:
            theta (torch.tensor[float], shape: (batch_size, *parameter_size)):
                The parameter tensor.

        Returns:
            X (torch.Tensor[float], shape: (batch_size, n_items)):
                A tensor representing the selection. During training, this is a
                continuous tensor of probabilities. Otherwise, it is a discrete
                (0 or 1) tensor.
        """
        batch_size = len(theta)
        device = theta.device
        self.utility.set_params(theta)
        n_items = self.utility.get_n_items()
        X = torch.rand(batch_size, n_items, device=device)
        X = self.constraint.project_continuous(X)
        if self.training:
            return X
        else:
            return self.constraint.project_discrete(X)
