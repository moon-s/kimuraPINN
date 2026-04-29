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
