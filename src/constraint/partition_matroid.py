from functools import reduce

import torch

from .base import BaseConstraint
from .cardinality import project_uniform_matroid_boundary


def is_partition(partition: list[list[int]]) -> bool:
    all_elements = sum([len(x) for x in partition])
    all_distinct_elements = len(set(reduce(lambda x, y: x + y, partition)))
    return all_elements == all_distinct_elements


class PartitionMatroid(BaseConstraint):
    """Partition matroid constraint.
    In this constraint, items are divided into disjoint partitions, and a
    separate cardinality limit is applied to each partition.

    Attributes:
        max_cardinalities (list[int]):
            A list where max_cardinalities[i] is the cardinality constraint
            for i-th partition.
        parition (list[list[int]]):
            A list where partition[i] is the list of item indices in i-th partition.
    """

    def __init__(self, max_cardinalities: list[int], partition: list[list[int]]):
        if len(max_cardinalities) != len(partition):
            raise ValueError("len(max_cardinalities) != len(partition)")
        if not is_partition(partition):
            raise ValueError(f"{partition} is not partition.")
        self.max_cardinalities = max_cardinalities
        self.partition = partition

    def get_candidates(self, X: torch.Tensor) -> torch.Tensor:
        """Returns candidate items.

        An item is a candidate if it is not already selected, and adding it
        does not violate the cardinality limit of its partition.

        Args:
            X (torch.Tensor[bool], shape: (batch_size, n_items)):
                A boolean tensor representing the current selection.

        Returns:
            candidates (torch.Tensor[bool], shape: (batch_size, n_items)):
                A boolean tensor indicating candidate items.
        """

        candidates = torch.logical_not(X)
        for max_cardinality, indices in zip(self.max_cardinalities, self.partition):
            n_selected_items = torch.sum(X[:, indices], dim=1, keepdims=True)
            candidates[:, indices] = torch.logical_and(
                n_selected_items < max_cardinality, candidates[:, indices]
            )
        return candidates

    def project_continuous(self, X: torch.Tensor) -> torch.Tensor:
        """Projects a continuous selection onto the partition matroid polytope.

        Args:
            X (torch.Tensor[float], shape: (batch_size, n_items)):
                A tensor of selection probabilities.

        Returns:
            X_projected (torch.Tensor[float], shape: (batch_size, n_items)):
                A tensor of selection probabilities that satisfies the constraint.
        """
        X_projected = torch.zeros_like(X)
        for max_cardinality, indices in zip(self.max_cardinalities, self.partition):
            X_projected[:, indices] = project_uniform_matroid_boundary(
                X[:, indices], k=max_cardinality, c=1.0
            )
        return X_projected

    def project_discrete(self, X: torch.Tensor) -> torch.Tensor:
        """Projects a continuous selection to a discrete one for the partition matroid.

        For each partition, this method selects the top-k items with the
        highest probabilities, where k is the limit for that partition.

        Args:
            X (torch.Tensor[float], shape: (batch_size, n_items)):
                A tensor of selection probabilities.

        Returns:
            X_projected (torch.Tensor[float], shape: (batch_size, n_items)):
                A discrete (0 or 1) tensor representing a valid selection.
        """
        return self.solve_frank_wolfe(X)

    def solve_frank_wolfe(self, grad: torch.Tensor) -> torch.Tensor:
        """Solves the Frank-Wolfe step for the partition matroid constraint.

        For each partition, this method selects the top-k items with the
        highest gradients, where k is the limit for that partition.

        Args:
            grad (torch.Tensor[float], shape: (batch_size, n_items)):
                The gradient tensor.

        Returns:
            v (torch.Tensor[float], shape: (batch_size, n_items)):
                A discrete tensor representing the vertex that maximizes the objective.
        """
        v = torch.zeros_like(grad)
        for max_cardinality, indices in zip(self.max_cardinalities, self.partition):
            _, indices_top_k = torch.topk(grad[:, indices], k=max_cardinality, dim=1)
            v[:, indices] = v[:, indices].scatter_(1, indices_top_k, 1.0)
        return v
