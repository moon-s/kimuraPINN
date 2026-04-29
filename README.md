# KimuraPINN

Research code for folded site frequency spectrum preprocessing and future
Kimura diffusion PINN modeling.

This initial slice implements:

- gnomAD-style VCF INFO parsing for population allele counts
- SNV-only filtering
- allele-count projection to population-specific cohort sizes
- count-indexed folded SFS output

Environment setup uses conda:

```bash
conda env create -f environment.yml
conda activate kimurapinn
```

## PINN core status

Implemented one-population differentiable Kimura components:

- `src/models/pinn.py`
- `src/models/selection_model.py`
- `src/pde/fokker_planck.py`
- `src/pde/boundary_conditions.py`

Covered by focused tests for forward shape, positivity, PDE residual autograd,
selection-model modes, smoothness loss, and boundary loss.

## Visualization

Generate run figures with:

```bash
python scripts/05_visualize_results.py \
  --run-dir results/time_varying_nfe_test \
  --log-sfs
```

Figures are written to `results/time_varying_nfe_test/figures/` as PNG and PDF:
SFS observed vs predicted, SFS residuals, gamma trajectory, loss history, and
optionally a `phi(x,t)` density heatmap when `model.pt` loads cleanly.

## Two-Population Admixture Status

The current two-population training path fits a joint density model to marginal
folded SFS files only:

```bash
python scripts/04_train_admixture_pinn.py \
  --sfs-a data/processed/folded_sfs_afr.tsv \
  --sfs-b data/processed/folded_sfs_nfe.tsv \
  --population-a afr \
  --population-b nfe \
  --config configs/two_pop_admixture.yaml \
  --output-dir results/admixture_afr_nfe_test \
  --epochs 100
```

This is an initial marginal-SFS approximation. True admixture inference requires
joint SFS, haplotypes, LD, or additional constraints. Current migration estimates
should be interpreted as effective coupling parameters, not definitive migration
rates.
