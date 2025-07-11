import torch

from ..constraint import BaseConstraint
from ..utility import BaseUtility
from .base import BaseSelector


class SGASelector(BaseSelector):
    """Selects items using continuous relaxation and stochastic gradient ascent (SGA).

    Attributes:
        utility (BaseUtility): The utility instance to be maximized.
        constraint (BaseConstraint): The constraint instance.
        lr (float): The learning rate parameter for the SGA optimizer.
            Default: 1e-1
        momentum (float): The momentum factor for the SGA optimizer.
            Default: 0.9
        max_epochs (int): The number of SGA iterations.
            Default: 10

    Reference:
        Mohammad Karimi et al.
        Stochastic Submodular Maximization: The Case of Coverage Functions.
        In Proc NeurIPS. 2017.
    """

    def __init__(
        self,
        utility: BaseUtility,
        constraint: BaseConstraint,
        lr: float = 0.1,
        momentum=0.9,
        max_epochs: int = 10,
    ):
        super().__init__(utility, constraint)
        self.lr = lr
        self.momentum = momentum
        self.max_epochs = max_epochs

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Selects items using continuous relaxation and stochastic gradient ascent.

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
        # initialize X as it satisfies the constraints
        X = torch.rand(
            (batch_size, n_items),
            device=device,
            requires_grad=True,
        )
        X = self.constraint.project_continuous(X)
        # optimize X by stochastic gradient ascent
        momentum = None
        for _ in range(self.max_epochs):
            loss = -torch.mean(self.utility(X))
            grad = torch.autograd.grad(
                loss, X, retain_graph=self.training, create_graph=self.training
            )[0]
            if momentum is None:
                momentum = grad
            else:
                momentum = self.momentum * momentum + grad
            X = X - self.lr * (grad + self.momentum * momentum)
            X = self.constraint.project_continuous(X)
        if self.training:
            return X
        else:
            return self.constraint.project_discrete(X)
