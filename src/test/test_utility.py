import numpy as np
import torch

from ..utility import InfluenceUtility
from .utility_slow import InfluenceUtilitySlow

rng = np.random.RandomState(1)
batch_size = 32
n_items = 30
n_users = 50


def test_influence_utility_forward():
    theta = rng.rand(batch_size, n_items, n_users)
    mask = rng.rand(batch_size, n_items, n_users) > 0.8
    theta = np.where(mask, theta, 0.0)
    utility = InfluenceUtility()
    utility_slow = InfluenceUtilitySlow()
    utility.set_params(torch.tensor(theta))
    utility_slow.set_params(torch.tensor(theta))
    # binary input
    for p in np.linspace(0, 1.0, 6):
        X = torch.bernoulli(torch.ones(batch_size, n_items) * p)
        torch.testing.assert_close(
            utility.forward(X).to(torch.float32),
            utility_slow.forward(X).to(torch.float32),
        )
    # probability input
    for _ in range(5):
        X = torch.rand(batch_size, n_items)
        torch.testing.assert_close(
            utility.forward(X).to(torch.float32),
            utility_slow.forward(X).to(torch.float32),
        )


def test_influence_utility_gain():
    theta = rng.rand(batch_size, n_items, n_users)
    mask = rng.rand(batch_size, n_items, n_users) > 0.8
    theta = np.where(mask, theta, 0.0)
    utility = InfluenceUtility()
    utility_slow = InfluenceUtilitySlow()
    utility.set_params(torch.tensor(theta))
    utility_slow.set_params(torch.tensor(theta))
    # binary input
    for p in np.linspace(0, 1.0, 6):
        X = torch.bernoulli(torch.ones(batch_size, n_items) * p) == 1.0
        candidates = torch.ones_like(X).to(torch.bool)
        actual = utility.compute_marginal_gains(X, candidates).to(torch.float32)
        expected = utility_slow.compute_marginal_gains(X, candidates).to(torch.float32)
        torch.testing.assert_close(actual, expected)
