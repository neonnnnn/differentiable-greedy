from abc import ABCMeta

import torch

from ..constraint import BaseConstraint
from ..utility import BaseUtility


class BaseSelector(torch.nn.Module, metaclass=ABCMeta):
    """Abstract base class for an item selection algorithm (Selector)."""

    def __init__(self, utility: BaseUtility, constraint: BaseConstraint):
        super().__init__()
        self.utility = utility
        self.constraint = constraint
        self.training = False
