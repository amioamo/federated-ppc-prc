"""Compose two nucleotide posteriors."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from baselines import naive_posterior_product, posterior_average
from federated_PPC import compose_ab


def main() -> None:
    generator = torch.Generator().manual_seed(0)
    log_q0, log_qa, log_qb = torch.randn(3, 8, 4, generator=generator).log_softmax(-1)
    composed = compose_ab(log_q0, log_qa, log_qb)
    naive = naive_posterior_product((log_qa, log_qb)).exp()
    average = posterior_average((log_qa.exp(), log_qb.exp()))
    print("PPC mean entropy", float(-(composed.ppc_probability * composed.ppc_probability.clamp_min(1e-12).log()).sum(-1).mean()))
    print("PRC mean entropy", float(-(composed.prc_probability * composed.prc_probability.clamp_min(1e-12).log()).sum(-1).mean()))
    print("naive product vs PPC max |diff|", float((naive - composed.ppc_probability).abs().max()))
    print("average vs PPC max |diff|", float((average - composed.ppc_probability).abs().max()))


if __name__ == "__main__":
    main()
