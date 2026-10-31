"""Federated posterior product composition (PPC)."""

from composition import compose_ab, posterior_product, posterior_product_two
from dirichlet_flow import flow_from_probability, ppc_velocity
from federated_PPC.coordinator import FederatedCompositionPlan

__all__ = [
    "FederatedCompositionPlan",
    "compose_ab",
    "flow_from_probability",
    "posterior_product",
    "posterior_product_two",
    "ppc_velocity",
]
