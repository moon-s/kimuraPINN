# Master Prompt Script for Codex / Agent: KimuraPINN

## Purpose

Use this master prompt to instruct an autonomous coding agent such as Codex to implement, test, visualize, and document a research-grade PINN framework for Kimura diffusion in population genomics.

The current scientific focus is:

- folded SFS now; unfolded SFS later when ancestral allele information becomes available
- recent evolutionary time scale, approximately 10–20 ky
- fluctuation of selection coefficient over time, \(\gamma(t)\)
- multiple complex admixture / migration histories for a given genomic region
- modular implementation with strong input/output contracts for every script
- validation, visualization, and manuscript drafting

---

# MASTER PROMPT

You are an autonomous scientific coding agent working on a population genomics project called **KimuraPINN**.

Your task is to build a research-grade Python/PyTorch framework that implements a **Physics-Informed Neural Network (PINN)** for the Kimura diffusion model of allele frequency evolution.

This project extends classical Kimura / Wright–Fisher diffusion modeling by allowing:

1. **Time-varying selection**  
   Selection coefficient \(\gamma(t)\) may fluctuate over recent evolutionary time.

2. **Complex demography**  
   Effective population size \(N_e(t)\) may vary through bottlenecks, expansion, or user-defined epochs.

3. **Multiple admixture / migration models**  
   For a given genomic region, multiple populations may exchange ancestry through migration or admixture matrices.

4. **Folded SFS now, unfolded SFS later**  
   The initial implementation must support folded SFS because ancestral allele information is not yet available. The design must make it easy to support unfolded SFS later.

5. **Script-level documentation**  
   Every script must contain a header describing:
   - purpose
   - biological/statistical assumptions
   - input files
   - output files
   - process schematic
   - example usage

Do not produce pseudo-code. Produce runnable, tested code.

---

# Repository Context

The repository should be organized as:

```text
KimuraPINN/
├── README.md
├── requirements.txt
├── pyproject.toml
├── data/
│   ├── raw/
│   │   └── sample.vcf
│   ├── processed/
│   └── simulated/
├── configs/
│   ├── single_pop_folded.yaml
│   ├── time_varying_selection.yaml
│   ├── two_pop_admixture.yaml
│   └── three_pop_admixture.yaml
├── src/
│   ├── io/
│   │   ├── vcf_parser.py
│   │   └── sfs_io.py
│   ├── sfs/
│   │   ├── folded.py
│   │   ├── projection.py
│   │   └── summary_stats.py
│   ├── pde/
│   │   ├── kimura.py
│   │   ├── fokker_planck.py
│   │   └── boundary_conditions.py
│   ├── models/
│   │   ├── pinn.py
│   │   ├── selection_model.py
│   │   ├── demography_model.py
│   │   └── admixture_model.py
│   ├── inference/
│   │   ├── losses.py
│   │   ├── inverse_solver.py
│   │   └── training.py
│   ├── simulation/
│   │   ├── simulate_wf.py
│   │   └── simulate_sfs.py
│   ├── visualization/
│   │   ├── plot_sfs.py
│   │   ├── plot_gamma.py
│   │   ├── plot_density.py
│   │   └── plot_admixture.py
│   └── utils/
│       ├── config.py
│       ├── logging.py
│       └── seed.py
├── scripts/
│   ├── 01_vcf_to_folded_sfs.py
│   ├── 02_train_single_pop_pinn.py
│   ├── 03_train_time_varying_gamma.py
│   ├── 04_train_admixture_pinn.py
│   ├── 05_visualize_results.py
│   └── 06_draft_methods.py
├── tests/
│   ├── test_vcf_parser.py
│   ├── test_folded_sfs.py
│   ├── test_projection.py
│   ├── test_pde_residual.py
│   ├── test_selection_model.py
│   ├── test_admixture_model.py
│   ├── test_inverse_solver.py
│   └── test_end_to_end.py
├── notebooks/
│   ├── 01_folded_sfs_demo.ipynb
│   ├── 02_time_varying_selection_demo.ipynb
│   └── 03_admixture_demo.ipynb
└── manuscript/
    ├── methods.md
    ├── results_template.md
    └── figure_legends.md
```

---

# Global Implementation Rules

## Programming language

Use Python 3.10+.

## Core libraries

Use:

- PyTorch for PINN and autograd
- NumPy and SciPy for numerical utilities
- pandas for tabular outputs
- PyYAML for configuration
- matplotlib for visualization
- pytest for tests

Avoid unnecessary dependencies unless clearly justified.

## Code quality

Every module must:

- be importable
- include type hints where practical
- include docstrings
- avoid hidden global state
- avoid hardcoded absolute paths
- support command-line execution where appropriate

## Reproducibility

Every training or simulation script must:

- accept a random seed
- log config parameters
- save outputs to a timestamped or user-specified output directory
- save a machine-readable `run_config.yaml`
- save summary metrics as `metrics.json`

## Testing

Every module must have pytest tests.

Tests must cover:

- normal input
- edge cases
- shape consistency
- numerical sanity checks
- end-to-end minimal example

---

# Biological and Statistical Assumptions

## Current stage

The current implementation uses **folded SFS** because ancestral allele information is not yet available.

For a variant with allele count `AC` and allele number `AN`:

```python
x = AC / AN
x_folded = min(x, 1.0 - x)
```

The folded SFS domain is:

```text
x ∈ [0, 0.5]
```

The code must be designed so that unfolded SFS can be added later.

## Future stage

When ancestral allele information becomes available, support:

```text
x_derived = derived_allele_count / AN
x ∈ [0, 1]
```

Do not hardwire folded-only logic into the PINN core. Keep folding as a data/interface layer.

---

# Input Data Requirements

## VCF input

Initial input is a gnomAD-style VCF or VCF-like file.

Required fields:

- `CHROM`
- `POS`
- `REF`
- `ALT`
- population-specific allele counts:
  - `AC_afr`
  - `AC_eas`
  - `AC_nfe`
- population-specific allele numbers when available:
  - `AN_afr`
  - `AN_eas`
  - `AN_nfe`
- variant type field:
  - `variant_type`

Only SNVs should be retained:

```text
variant_type == "snv"
```

The implementation must be robust to missing fields and should emit informative errors.

## Example input

The repository contains:

```text
data/raw/sample.vcf
```

This file contains three genomic blocks with SNVs from gnomAD-like data.

---

# Output Data Requirements

## From VCF parsing

Save:

```text
data/processed/allele_counts.tsv
```

Required columns:

```text
chrom
pos
ref
alt
population
ac
an
af
maf
block_id
variant_type
```

## From folded SFS construction

Save:

```text
data/processed/folded_sfs_<population>.tsv
```

Required columns:

```text
bin_left
bin_right
bin_center
count
population
```

Also save:

```text
data/processed/folded_sfs_summary.json
```

containing:

```json
{
  "n_variants_total": 0,
  "n_snvs_retained": 0,
  "populations": {},
  "n_bins": 0,
  "folded": true
}
```

## From PINN training

Save:

```text
results/<run_name>/model.pt
results/<run_name>/run_config.yaml
results/<run_name>/metrics.json
results/<run_name>/loss_history.tsv
results/<run_name>/gamma_trajectory.tsv
results/<run_name>/predicted_sfs.tsv
results/<run_name>/figures/
```

---

# Required Script Header Template

Every script in `scripts/` must start with a header like this:

```python
#!/usr/bin/env python3
"""
Script: <script_name>.py

Purpose:
    One-paragraph description of what this script does.

Biological Question:
    Describe the population-genetic question addressed by this script.

Assumptions:
    - Folded SFS is used because ancestral allele is unavailable.
    - Only SNVs are retained.
    - Allele count fields are taken from gnomAD-style INFO columns.

Input:
    - Path(s) to input files.
    - Required columns or VCF INFO fields.

Output:
    - Path(s) to generated files.
    - Description of output schema.

Process Schematic:
    Input VCF
        -> filter SNVs
        -> extract AC/AN
        -> compute AF and folded frequency
        -> construct folded SFS
        -> save processed data

Example:
    python scripts/<script_name>.py \
        --input data/raw/sample.vcf \
        --output-dir data/processed \
        --config configs/single_pop_folded.yaml
"""
```

---

# Mathematical Model

Implement the Kimura / Wright–Fisher diffusion in Fokker–Planck form.

For one population:

```text
∂φ/∂t = - ∂/∂x [M(x,t) φ] + 1/2 ∂²/∂x² [V(x,t) φ]
```

where:

```text
M(x,t) = 2 γ(t) h x(1-x)
V(x,t) = x(1-x) / ν(t)
```

Definitions:

- `x`: allele frequency
- `t`: time
- `φ(x,t)`: allele frequency density
- `γ(t)`: scaled selection coefficient over time
- `h`: dominance coefficient, initially fixed to 0.5
- `ν(t)`: relative effective population size

For folded SFS, the data are observed on:

```text
x_folded ∈ [0, 0.5]
```

However, internally the model should allow unfolded support later.

---

# Multi-population / Admixture Model

Implement a two-population model first, then make the API generalizable to K populations.

For population `i`:

```text
∂φ/∂t = Σ_i [-∂/∂x_i(M_i φ) + 1/2 ∂²/∂x_i²(V_i φ)]
         + Σ_{i≠j} m_ij ∂/∂x_i[(x_j - x_i) φ]
```

where:

- `m_ij`: migration/admixture rate from population `j` into population `i`
- `γ_i(t)`: population-specific selection trajectory
- `ν_i(t)`: population-specific effective size trajectory

Initial target populations:

```text
AFR
EAS
NFE
```

Population-specific VCF fields:

```text
AC_afr, AN_afr
AC_eas, AN_eas
AC_nfe, AN_nfe
```

The first implemented admixture model should support:

1. two-population AFR–NFE
2. two-population EAS–NFE
3. three-population AFR–EAS–NFE, if feasible after tests pass

---

# PINN Architecture

Implement a model class:

```python
class KimuraPINN(torch.nn.Module):
    def forward(self, x, t, context=None):
        ...
```

Inputs:

- `x`: allele frequency tensor, shape `[batch, K]` for K populations or `[batch, 1]` for one population
- `t`: time tensor, shape `[batch, 1]`
- `context`: optional tensor containing region-level features, demographic parameters, or conditioning variables

Output:

- non-negative allele frequency density `φθ(x,t)`

Requirements:

- use Softplus output or another positivity-preserving transform
- support Fourier features for `x` and `t`
- support residual MLP blocks
- avoid numerical instability near `x=0`

---

# Selection Model

Implement:

```python
class SelectionModel(torch.nn.Module):
    def forward(self, t, population=None, context=None):
        ...
```

Support three modes:

1. constant selection
2. piecewise linear \(\gamma(t)\)
3. neural-network \(\gamma(t)\)

The main scientific target is fluctuating selection over recent time.

Therefore, include smoothness regularization:

```text
L_smooth = ∫ |dγ/dt|² dt
```

Also include optional sparsity or total-variation regularization to prevent overfitting.

Save inferred trajectory as:

```text
gamma_trajectory.tsv
```

with columns:

```text
time
gamma
population
```

---

# Demography Model

Implement:

```python
class DemographyModel(torch.nn.Module):
    def forward(self, t, population=None):
        ...
```

Support:

1. constant \(N_e\)
2. epoch model
3. spline or neural-network \(N_e(t)\)

Initial implementation can use fixed demography from config, but the code must allow learnable demography later.

---

# Loss Functions

Implement a composite loss:

```text
L_total = λ_pde L_pde
        + λ_data L_data
        + λ_bc L_boundary
        + λ_ic L_initial
        + λ_gamma L_gamma_smooth
        + λ_reg L_regularization
```

## PDE loss

Computed by autograd at collocation points.

## Data loss

Compare observed folded SFS to predicted folded SFS.

Support:

1. MSE loss
2. Poisson negative log likelihood

## Boundary loss

Enforce biologically appropriate behavior near:

```text
x = 0
x = 0.5 for folded SFS
x = 1 for unfolded SFS later
```

## Initial condition loss

Support an initial stationary or user-specified prior distribution.

---

# SFS Projection

Implement projection from continuous density to discrete SFS.

For unfolded SFS:

```text
SFS(k) = ∫ Binomial(k; n, x) φ(x,t) dx
```

For folded SFS:

```text
SFS_folded(k) = SFS(k) + SFS(n-k), for k < n/2
```

For now, observed data are folded, so predicted SFS must also be folded before comparison.

---

# Implementation Tasks

Proceed in this exact order.

---

## Task 1: Project scaffolding

Create the repository structure.

Create:

- `requirements.txt`
- `pyproject.toml`
- base package layout
- `README.md`
- test structure

Acceptance criteria:

- `pytest` runs, even if only placeholder smoke tests exist initially
- package imports successfully

---

## Task 2: VCF parser

Implement:

```text
src/io/vcf_parser.py
scripts/01_vcf_to_folded_sfs.py
```

Requirements:

- parse gnomAD-style INFO fields
- retain only SNVs
- support populations: AFR, EAS, NFE
- output allele count table
- assign `block_id` based on genomic blocks if present; otherwise infer blocks from distance or keep as one block

Tests:

- parse synthetic VCF
- filter non-SNV variants
- handle missing AC/AN gracefully
- verify AF and MAF

Acceptance criteria:

```bash
python scripts/01_vcf_to_folded_sfs.py \
  --input data/raw/sample.vcf \
  --output-dir data/processed \
  --populations afr eas nfe
```

produces allele count and folded SFS files.

---

## Task 3: Folded SFS construction

Implement:

```text
src/sfs/folded.py
src/sfs/summary_stats.py
```

Requirements:

- compute folded frequency
- bin into user-defined number of bins
- compute singleton density
- compute total variant count
- compute per-block SFS if block information exists

Tests:

- AC=1 and AC=AN-1 fold to same bin
- total variant count preserved after folding
- empty bins handled correctly

---

## Task 4: PINN core

Implement:

```text
src/models/pinn.py
src/pde/fokker_planck.py
src/pde/boundary_conditions.py
```

Requirements:

- PyTorch model
- positivity-preserving output
- autograd derivatives
- one-population PDE residual
- folded-domain support

Tests:

- forward pass shape
- positivity
- finite PDE residual
- gradients exist for trainable parameters

---

## Task 5: Time-varying selection

Implement:

```text
src/models/selection_model.py
```

Requirements:

- constant mode
- piecewise linear mode
- neural-network mode
- smoothness regularization
- export trajectory

Tests:

- constant model returns constant gamma
- piecewise model is continuous
- neural model outputs correct shape
- smoothness penalty is finite

---

## Task 6: Demography model

Implement:

```text
src/models/demography_model.py
```

Requirements:

- constant mode
- epoch mode
- optional learnable mode

Tests:

- returns positive Ne or ν
- handles multiple populations
- handles time tensors correctly

---

## Task 7: SFS projection and data loss

Implement:

```text
src/sfs/projection.py
src/inference/losses.py
```

Requirements:

- project continuous density to discrete SFS
- fold predicted SFS
- compute MSE and Poisson NLL

Tests:

- predicted SFS shape is correct
- folded projection preserves expected symmetry
- loss decreases in trivial matching case

---

## Task 8: Single-population inverse solver

Implement:

```text
src/inference/inverse_solver.py
src/inference/training.py
scripts/02_train_single_pop_pinn.py
scripts/03_train_time_varying_gamma.py
```

Requirements:

- load folded SFS
- initialize PINN, SelectionModel, DemographyModel
- train with composite loss
- save outputs
- support CPU and GPU

Tests:

- one short training run completes
- loss history saved
- gamma trajectory saved
- predicted SFS saved

Acceptance criteria:

```bash
python scripts/03_train_time_varying_gamma.py \
  --sfs data/processed/folded_sfs_nfe.tsv \
  --config configs/time_varying_selection.yaml \
  --output-dir results/time_varying_nfe_test \
  --epochs 100
```

runs without error and saves outputs.

---

## Task 9: Multi-population admixture model

Implement:

```text
src/models/admixture_model.py
src/pde/kimura.py
scripts/04_train_admixture_pinn.py
```

Requirements:

- support K=2 initially
- support migration matrix `m_ij`
- support population-specific γ_i(t)
- support population-specific demography ν_i(t)
- compute multi-dimensional PDE residual

Tests:

- zero migration reduces to independent populations
- symmetric migration matrix behaves symmetrically
- gradients exist for migration parameters
- short two-pop training run completes

Acceptance criteria:

```bash
python scripts/04_train_admixture_pinn.py \
  --sfs-a data/processed/folded_sfs_afr.tsv \
  --sfs-b data/processed/folded_sfs_nfe.tsv \
  --config configs/two_pop_admixture.yaml \
  --output-dir results/admixture_afr_nfe_test \
  --epochs 100
```

runs and saves migration estimates plus γ trajectories.

---

## Task 10: Simulation-based validation

Implement:

```text
src/simulation/simulate_wf.py
src/simulation/simulate_sfs.py
```

Requirements:

- simulate Wright–Fisher allele frequency trajectories
- support known γ(t)
- support known Ne(t)
- support simple two-pop migration
- generate folded SFS

Tests:

- stronger positive selection shifts frequency distribution upward
- neutral simulation produces expected excess low-frequency variants
- known migration reduces divergence between populations

Acceptance criteria:

- generate synthetic SFS under known parameters
- train PINN on synthetic SFS
- recover parameters qualitatively

---

## Task 11: Visualization

Implement:

```text
src/visualization/plot_sfs.py
src/visualization/plot_gamma.py
src/visualization/plot_density.py
src/visualization/plot_admixture.py
scripts/05_visualize_results.py
```

Required figures:

1. observed vs predicted folded SFS
2. inferred γ(t) trajectory
3. PINN density heatmap φ(x,t)
4. loss curves
5. estimated migration/admixture matrix heatmap
6. per-population comparison of SFS

Requirements:

- matplotlib only
- no seaborn
- save PDF and PNG
- publication-quality labels
- log scale option for SFS

Acceptance criteria:

```bash
python scripts/05_visualize_results.py \
  --run-dir results/time_varying_nfe_test
```

creates figures in:

```text
results/time_varying_nfe_test/figures/
```

---

## Task 12: Manuscript draft

Implement:

```text
scripts/06_draft_methods.py
manuscript/methods.md
manuscript/results_template.md
manuscript/figure_legends.md
```

The script should read run outputs and produce a draft Methods/Results markdown.

The manuscript should describe:

1. motivation
2. folded SFS construction
3. Kimura diffusion equation
4. PINN formulation
5. time-varying selection model
6. multi-population admixture model
7. training and optimization
8. simulation validation
9. visualization outputs
10. limitations and transition to unfolded SFS when ancestral alleles become available

Use formal academic tone suitable for a population genomics or computational biology manuscript.

---

# Configuration Files

Create YAML configs.

## `configs/single_pop_folded.yaml`

```yaml
seed: 123
mode: single_population
sfs:
  folded: true
  n_bins: 50
  population: nfe
model:
  hidden_dim: 128
  n_layers: 4
  fourier_features: true
selection:
  mode: constant
  initial_gamma: 0.0
demography:
  mode: constant
  nu: 1.0
training:
  epochs: 1000
  batch_collocation: 4096
  lr: 0.001
loss_weights:
  pde: 1.0
  data: 10.0
  boundary: 1.0
  initial: 1.0
  gamma_smooth: 0.1
```

## `configs/time_varying_selection.yaml`

```yaml
seed: 123
mode: time_varying_selection
sfs:
  folded: true
  n_bins: 50
  population: nfe
time:
  t_min: 0.0
  t_max: 0.02
  units: coalescent_scaled
model:
  hidden_dim: 128
  n_layers: 4
  fourier_features: true
selection:
  mode: piecewise_linear
  n_breakpoints: 6
  initial_gamma: 0.0
  smoothness_penalty: 0.1
demography:
  mode: epoch
  epochs:
    - {start: 0.0, end: 0.005, nu: 1.0}
    - {start: 0.005, end: 0.015, nu: 0.2}
    - {start: 0.015, end: 0.02, nu: 2.0}
training:
  epochs: 5000
  batch_collocation: 4096
  lr: 0.001
loss_weights:
  pde: 1.0
  data: 10.0
  boundary: 1.0
  initial: 1.0
  gamma_smooth: 0.5
```

## `configs/two_pop_admixture.yaml`

```yaml
seed: 123
mode: two_population_admixture
sfs:
  folded: true
  n_bins: 50
  populations: [afr, nfe]
time:
  t_min: 0.0
  t_max: 0.02
model:
  hidden_dim: 160
  n_layers: 5
  fourier_features: true
selection:
  mode: piecewise_linear
  n_breakpoints: 6
  population_specific: true
demography:
  mode: epoch
  population_specific: true
admixture:
  learn_migration: true
  initial_matrix:
    - [0.0, 0.001]
    - [0.001, 0.0]
training:
  epochs: 5000
  batch_collocation: 8192
  lr: 0.001
loss_weights:
  pde: 1.0
  data: 10.0
  boundary: 1.0
  gamma_smooth: 0.5
  migration_reg: 0.1
```

---

# Validation Experiments

Implement at least four validation experiments.

## Experiment 1: folded SFS parsing sanity check

Input:

```text
data/raw/sample.vcf
```

Output:

- folded SFS for AFR, EAS, NFE
- summary JSON

Expected:

- only SNVs retained
- AC and AN extracted correctly
- folded counts are consistent

## Experiment 2: constant selection baseline

Goal:

- ensure PINN can fit simple folded SFS

Compare:

- observed folded SFS
- predicted folded SFS

## Experiment 3: time-varying selection recovery

Goal:

- simulate SFS under known γ(t)
- infer γ(t)
- compare inferred vs true trajectory

Required figure:

```text
true_vs_inferred_gamma.pdf
```

## Experiment 4: two-population admixture recovery

Goal:

- simulate two populations with migration
- infer migration matrix and γ_i(t)

Required figures:

```text
migration_matrix.pdf
gamma_by_population.pdf
sfs_by_population.pdf
```

---

# Visualization Requirements

All figures must be saved as both `.png` and `.pdf`.

Required figure style:

- clear axis labels
- units stated when available
- legends outside plot if crowded
- log-scale option for SFS
- no unnecessary grid clutter
- no seaborn dependency

---

# README Update Requirements

Update `README.md` to include:

1. project overview
2. biological motivation
3. Kimura diffusion background
4. folded SFS now / unfolded SFS later
5. VCF input schema
6. pipeline schematic
7. module-level input/output contracts
8. quickstart commands
9. validation plan
10. visualization outputs
11. manuscript generation
12. limitations

Include this schematic:

```text
sample.vcf
   ↓
SNV filtering + AC/AN extraction
   ↓
Population-specific allele frequencies
   ↓
Folded SFS per population and block
   ↓
PINN with Kimura Fokker–Planck residual
   ↓
Inference of γ(t), Ne(t), and migration/admixture
   ↓
Visualization + manuscript draft
```

---

# Manuscript Draft Requirements

Create `manuscript/methods.md` with sections:

```text
1. Variant processing and folded SFS construction
2. Kimura diffusion model
3. Physics-informed neural network formulation
4. Time-varying selection model
5. Multi-population admixture extension
6. Loss function and inverse inference
7. Simulation-based validation
8. Visualization and reproducibility
9. Limitations of folded SFS and future unfolded extension
```

Create `manuscript/figure_legends.md` with legends for:

- Figure 1: Pipeline schematic
- Figure 2: Folded SFS construction
- Figure 3: PINN model architecture
- Figure 4: Time-varying selection inference
- Figure 5: Multi-population admixture inference
- Figure 6: Simulation validation

---

# Final Deliverables

When finished, provide a concise completion report with:

1. files created or modified
2. tests implemented
3. commands to reproduce the full pipeline
4. known limitations
5. next recommended steps

The final pipeline should run with:

```bash
pip install -e .
pytest

python scripts/01_vcf_to_folded_sfs.py \
  --input data/raw/sample.vcf \
  --output-dir data/processed \
  --populations afr eas nfe

python scripts/03_train_time_varying_gamma.py \
  --sfs data/processed/folded_sfs_nfe.tsv \
  --config configs/time_varying_selection.yaml \
  --output-dir results/time_varying_nfe_test \
  --epochs 100

python scripts/04_train_admixture_pinn.py \
  --sfs-a data/processed/folded_sfs_afr.tsv \
  --sfs-b data/processed/folded_sfs_nfe.tsv \
  --config configs/two_pop_admixture.yaml \
  --output-dir results/admixture_afr_nfe_test \
  --epochs 100

python scripts/05_visualize_results.py \
  --run-dir results/time_varying_nfe_test

python scripts/06_draft_methods.py \
  --run-dir results/time_varying_nfe_test \
  --output-dir manuscript
```

---

# Critical Scientific Cautions

Do not overclaim biological inference.

Explicitly state in README and manuscript:

- folded SFS loses ancestral directionality
- selection and demography can be partially confounded
- migration and shared demography can be difficult to distinguish
- PINN inference must be benchmarked against simulation and classical solvers
- current results are methodological until validated on known selection loci or simulations

---

# End of Master Prompt

