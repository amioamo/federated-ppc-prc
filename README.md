# Federated PPC, PRC, and private summary composition

License: MIT (see `LICENSE`).

## What is included

Federated methods:

- `federated_PPC`: posterior product \(q^\star \propto q_A q_B / q_0\)
- `PRC`: projected first-order residual composition
- `private_summary_composition`: differentially private Gaussian summary
  composition on bounded public sequence features

Coordinator helpers for PPC/PRC and sample-count FedAvg of model parameters
are included. The uncorrected product \(\mathrm{Norm}(q_A q_B)\) and the
component average \((q_A+q_B)/2\) are provided as composition controls.
Unit tests are included.

## What is not included

This repository does not ship assay tables, genome FASTA files, evaluator
weights, or the Dirichlet flow-matching checkpoints trained for the article.
Those assets are either third-party releases (links below) or
article-specific trained weights. Retraining follows the article protocol
after the public files have been obtained under their original licenses.

## Data, code, and weights

Follow each source's license. Do not re-upload restricted files into a public
fork of this repository unless that license allows it.

| Asset | What to download | Where |
|---|---|---|
| DeepSTARR sequences and activity labels; official Keras predictor (`DeepSTARR.model.json`, `DeepSTARR.model.h5`) | Manuscript data and trained model | [Zenodo 5502060](https://doi.org/10.5281/zenodo.5502060); code at [bernardo-de-almeida/DeepSTARR](https://github.com/bernardo-de-almeida/DeepSTARR) |
| Human DHS index (Meuleman et al.) | Consensus DHS coordinates, vocabulary and occurrence matrices | [Zenodo 3838751](https://doi.org/10.5281/zenodo.3838751); ENCODE annotation [ENCSR857UZV](https://www.encodeproject.org/annotations/ENCSR857UZV/) |
| Sei sequence model | `sei_model.tar.gz` (chromatin-profile network and 40-class projection) | [Zenodo 4906997](https://doi.org/10.5281/zenodo.4906997); inference code at [FunctionLab/sei-framework](https://github.com/FunctionLab/sei-framework) |
| Dirichlet flow matching (LRAR / Riemannian Flow backbone) | Published DNA sequence-design implementation | [HannesStark/dirichlet-flow-matching](https://github.com/HannesStark/dirichlet-flow-matching) |

The DeepSTARR and Sei files above are the official predictor weights used as
evaluators. The article's own generators are trained from the public sequence
sets using the operators in this repository; those generator checkpoints are
not hosted here.

## Quick check

Python 3.10+ with the packages in `requirements.txt`. From this directory:

```sh
pip install -r requirements.txt
PYTHONPATH=. python examples/toy_composition.py
PYTHONPATH=. python examples/toy_private_composition.py
PYTHONPATH=. python -m unittest tests.test_composition tests.test_coordinator tests.test_private_summary_composition
```
