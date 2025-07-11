import torch

from .base import BaseSelector


class GreedySelector(BaseSelector):
    """Selects items using a standard discrete greedy selection algorithm.

    Attributes:
        utility (BaseUtility): The utility instance to be maximized.
        constraint (BaseConstraint): The constraint instance.
    """

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        """Selects items using the standard discrete greedy algorithm.

        Args:
            theta (torch.tensor[float], shape: (batch_size, *parameter_size)):
                The estimated parameter tensor.

        Returns:
            X (torch.Tensor[float], shape: (batch_size, n_items)):
                A tensor representing the selected items (0 or 1).
        """
        batch_size = len(theta)
        device = theta.device
        self.utility.set_params(theta)
        n_items = self.utility.get_n_items()
        X = torch.zeros(batch_size, n_items, device=device, dtype=torch.bool)
        for _ in range(n_items):
            candidates = self.constraint.get_candidates(X)
            if not torch.any(torch.any(candidates, dim=1)).item():
                break
            gains = self.utility.compute_marginal_gain(X, candidates)
            selected_items = torch.argmax(
                torch.where(candidates, gains, -torch.inf), dim=1
            )
            selected_mask = torch.nn.functional.one_hot(
                selected_items, num_classes=n_items
            )
            X = X | selected_mask.to(torch.bool)
        return X.to(torch.float32)
