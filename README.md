# Federated PPC and PRC

Reference implementation of posterior product composition (PPC), projected residual composition (PRC), and differentially private summary composition for regulatory sequence design.

The generation entry point is `sample_composition` in `method/`. Sequence discrepancy, DeepSTARR activity scores, and the DHS scorers are in `evaluation/`. Comparison generators are in `baselines/`.

Associated manuscript: *Composing Condition-Specific Distributions for Federated Regulatory DNA Design*.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+ recommended.

## Data

Assay tables and trained weights come from their original releases.

- DeepSTARR sequences, activity labels, and the official predictor (`DeepSTARR.model.json`, `DeepSTARR.model.h5`): [Zenodo 5502060](https://doi.org/10.5281/zenodo.5502060); code at [bernardo-de-almeida/DeepSTARR](https://github.com/bernardo-de-almeida/DeepSTARR).
- Human DHS index: [Zenodo 3838751](https://doi.org/10.5281/zenodo.3838751); ENCODE annotation [ENCSR857UZV](https://www.encodeproject.org/annotations/ENCSR857UZV/).
- Sei sequence model (`sei_model.tar.gz`): [Zenodo 4906997](https://doi.org/10.5281/zenodo.4906997); inference code at [FunctionLab/sei-framework](https://github.com/FunctionLab/sei-framework).
- Dirichlet flow matching: [HannesStark/dirichlet-flow-matching](https://github.com/HannesStark/dirichlet-flow-matching).

`evaluation.deepstarr.load_model` reads a directory containing the two official DeepSTARR files. DHS scorers in `evaluation.dhs` are fit on caller-supplied 200-bp tokens.

## Experiments

Run commands from the repository root.

```bash
python examples/toy_composition.py
python examples/toy_private_composition.py
python examples/toy_evaluation.py
python -m unittest discover -s tests
```

## Core library

- `method/` — PPC, PRC, the Dirichlet denoiser, coordinator coefficients, and private summary composition
- `baselines/` — uncorrected product, posterior average, FedAvg, and single-model sampling
- `evaluation/` — k-mer discrepancy, activity distances, DeepSTARR inference, and DHS scorers

## Citation

If you use this code, please cite the associated article and archive DOI (to be added upon deposit).

## License

MIT License. See [LICENSE](LICENSE).
