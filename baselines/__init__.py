"""Comparison methods."""

from baselines.controls import naive_posterior_product, posterior_average
from baselines.fedavg import fedavg_state_dict
from baselines.generative import sample_naive_product, sample_posterior_average, sample_single

__all__ = [
    "fedavg_state_dict",
    "naive_posterior_product",
    "posterior_average",
    "sample_naive_product",
    "sample_posterior_average",
    "sample_single",
]
