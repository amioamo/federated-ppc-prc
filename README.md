# Federated PPC and PRC

Reference implementation of posterior product composition (PPC), projected residual composition (PRC), and differentially private summary composition for regulatory sequence design.

The composition entry point is `posterior_product` in `composition.py`. Generation uses `sample_composition` in `method/`. Sequence discrepancy, DeepSTARR activity scores, and the DHS scorers are in `evaluation/`.

Associated manuscript: *Composing Condition-Specific Distributions for Federated Regulatory DNA Design*.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+ recommended.

## Data

Assay tables and trained weights are obtained from their original releases. Do not re-upload restricted files into a public fork unless that license allows it.

- DeepSTARR sequences, activity labels, and the official predictor (`DeepSTARR.model.json`, `DeepSTARR.model.h5`): [Zenodo 5502060](https://doi.org/10.5281/zenodo.5502060); code at [bernardo-de-almeida/DeepSTARR](https://github.com/bernardo-de-almeida/DeepSTARR).
- Human DHS index: [Zenodo 3838751](https://doi.org/10.5281/zenodo.3838751); ENCODE annotation [ENCSR857UZV](https://www.encodeproject.org/annotations/ENCSR857UZV/).
- Sei sequence model (`sei_model.tar.gz`): [Zenodo 4906997](https://doi.org/10.5281/zenodo.4906997); inference code at [FunctionLab/sei-framework](https://github.com/FunctionLab/sei-framework).
- Dirichlet flow matching: [HannesStark/dirichlet-flow-matching](https://github.com/HannesStark/dirichlet-flow-matching).

`evaluation.deepstarr.load_model` reads a directory that contains the two official DeepSTARR files. DHS scorers in `evaluation.dhs` are fit on caller-supplied 200-bp tokens. Generator checkpoints trained for the article are not hosted here.

## Experiments

Run commands from the repository root.

```bash
python examples/toy_composition.py
python examples/toy_private_composition.py
python examples/toy_evaluation.py
python -m unittest discover -s tests
```

## Core library

- `composition.py` — PPC \(q^\star \propto q_A q_B / q_0\) and projected residual composition
- `PRC/` — projected first-order residual composition
- `method/` — Dirichlet denoiser, flow-matching loss, and PPC/PRC sampling
- `federated_PPC/` — coordinator coefficients for those two operators
- `private_summary_composition/` — Gaussian summary release, fit, and sampling
- `dirichlet_flow.py` — probability-space velocity used by PPC and PRC
- `evaluation/` — k-mer discrepancy, activity distances, DeepSTARR inference, and DHS scorers

## Baselines

Comparison generators live in `baselines/` and are not part of the method:

- `naive_posterior_product` — uncorrected product \(\mathrm{Norm}(q_A q_B)\)
- `posterior_average` — mixture \((q_A + q_B) / 2\)
- `fedavg_state_dict` — sample-count average of local parameters
- `sample_single`, `sample_naive_product`, and `sample_posterior_average` — Dirichlet sampling for those controls

## Citation

If you use this code, please cite the associated article and archive DOI (to be added upon deposit).

## License

MIT License. See [LICENSE](LICENSE).
