# MR-SF-TDDFT Reference-State and Response Workflow

## Overview

This repository contains a prototype implementation of a mixed-reference
spin-flip time-dependent density functional theory (MR-SF-TDDFT) workflow.

The workflow starts from a triplet unrestricted Kohn-Sham (UKS) reference and
progressively constructs:

1. A triplet UKS reference state.
2. Spatially aligned alpha and beta molecular orbitals.
3. An equiensemble mixed-reference density matrix (MR-RDM).
4. Clebsch-Gordan singlet/triplet spin-projection matrices.
5. A spin-flip exchange-correlation (XC) kernel and TDA response matrix.
6. Excitation energies through direct diagonalization or an iterative Davidson
   solver.

The intended downstream application is MR-SF-TDDFT and, ultimately,
SAPT(MR-SF-TDDFT).

> **Important:** The supplied implementation is a research/prototype codebase.
> Several routines explicitly contain simplified or placeholder approximations.
> Results should therefore be treated as development results unless the relevant
> components have been replaced or independently validated.

## Workflow

```text
                 Triplet UKS SCF
                       |
                       v
             Alpha/Beta MOs and energies
                       |
                       v
             Spatial orbital alignment
                       |
                       v
                  MR-RDM
             (50/50 triplet ensemble)
                       |
                       v
             Clebsch-Gordan projection
                /              \
               v                v
           Singlet space     Triplet space
                \              /
                 \            /
                  v          v
                 TDA response
                       |
                       v
             Davidson diagonalization
                       |
                       v
             Excitation energies
                       |
                       v
              MR-SF-TDDFT / SAPT
```

The source implementation contains these stages in a single Python source
file and uses NumPy/SciPy together with local UKS/DFT functionality and Psi4.

## Main Components

### 1. Triplet UKS reference

The first stage performs an unrestricted Kohn-Sham calculation with

```text
N_alpha - N_beta = 2
```

corresponding to a triplet spin configuration.

Main entry point:

```python
run_triplet_reference(
    molecule_string,
    basis="sto-3g",
    functional="b3lyp",
    n_radial=100,
    grid_order=14,
    max_iter=100,
    e_convergence=1e-8,
    d_convergence=1e-6,
    diis_start=5,
    max_diis=10,
    print_level=1,
    use_psi4=True,
    save_results=True,
    output_dir="./results",
)
```

The returned `TripletReferenceResult` contains the total energy, alpha/beta
orbital energies and coefficients, density matrices, Fock/XC matrices,
electron counts, spin contamination, convergence information, and timing.

The implementation also reports alpha/beta HOMO-LUMO information and spin
properties.

### 2. Spatial orbital alignment

The second stage aligns alpha and beta spatial orbitals.

The cross-overlap matrix is

```text
O = C_alpha^T S C_beta
```

where `S` is the AO overlap matrix.

Jacobi rotations are applied within occupied-occupied and virtual-virtual
blocks to make the alpha and beta orbital spaces more spatially coherent.

Main functions:

```python
compute_cross_overlap(...)
compute_jacobi_rotation(...)
align_spatial_orbitals(...)
align_from_files(...)
verify_alignment(...)
```

`AlignmentResult` contains aligned alpha/beta orbitals, rotation matrices,
initial/final cross-overlap matrices, convergence information, and timing.

### 3. Equiensemble MR-RDM

The third stage constructs a mixed-reference density matrix representing a
50/50 mixture of the `M_s = +1` and `M_s = -1` triplet components.

Two singly occupied molecular orbitals (SOMOs) are identified as:

```text
h = hole
p = particle
```

The active space contains these two orbitals. Core orbitals are treated as
doubly occupied, while the active orbitals receive equal alpha and beta
contributions.

Main functions:

```python
find_singly_occupied_molecular_orbitals(...)
construct_mr_rdm(...)
construct_mr_rdm_from_results(...)
build_mr_rdm_from_files(...)
verify_mr_rdm(...)
```

`MRRDMResult` stores `P_MR`, alpha/beta densities, aligned orbitals,
active-space indices and energies, electron counts, trace, idempotency status,
and related metadata.

When an AO overlap matrix is supplied, the implementation checks:

```text
P S P = P
```

### 4. Clebsch-Gordan spin projection

The fourth stage constructs singlet and triplet projection spaces using
Clebsch-Gordan coefficients.

The principal projection matrices are:

```text
T_S
T_T
```

for singlet and triplet response spaces.

Main functions:

```python
compute_clebsch_gordan_coefficients(...)
construct_transition_space(...)
construct_spin_projection_matrices(...)
construct_full_projection_matrices(...)
project_mr_rdm(...)
perform_spin_projection(...)
build_spin_projection_from_files(...)
verify_spin_projection(...)
```

`SpinProjectionResult` contains the projection matrices, projected MR-RDMs,
Clebsch-Gordan coefficients, transition counts, active-space indices, and
other statistics.

### 5. Spin-flip XC kernel and response matrix

The response stage evaluates a grid-based approximation to the spin-flip XC
kernel and constructs a TDA response matrix.

Relevant routines include:

```python
compute_spin_flip_kernel(...)
compute_spin_flip_xc_integral(...)
build_response_matrix(...)
diagonalize_response_matrix(...)
perform_response_calculation(...)
verify_response_matrix(...)
```

The calculation produces the response matrix, singlet/triplet blocks,
excitation energies, transition vectors, grid statistics, and timing data.

Excitation energies are converted from Hartree to eV using:

```text
1 Hartree = 27.2114 eV
```

### 6. Davidson diagonalization

A matrix-free Davidson solver is included for larger response spaces.

Main components:

```python
ResponseMatrixOperator
DavidsonResult
davidson_diagonalization(...)
solve_response_with_davidson(...)
verify_davidson_results(...)
```

The solver supports:

- singlet-channel calculations;
- triplet-channel calculations;
- multiple target roots;
- residual-based convergence;
- diagonal preconditioning;
- subspace expansion;
- restart capability.

## Requirements

The supplied source imports:

- Python 3
- NumPy
- SciPy
- Psi4
- local `UKS.py`
- local `helper_KS.py`

The UKS implementation is imported with:

```python
from UKS import UnrestrictedKohnSham, UKSResult
```

DFT, molecular-grid, basis-set, and functional functionality is imported from:

```python
from helper_KS import ...
```

Therefore, `UKS.py` and `helper_KS.py` must be available on the Python path.

### Basic environment

```bash
python -m venv mr-sf-tddft-env
source mr-sf-tddft-env/bin/activate

pip install numpy scipy
```

Install Psi4 using the recommended method for your operating system.

Verify the installation:

```bash
python -c "import psi4; print(psi4.__version__)"
python -c "import numpy, scipy; print(numpy.__version__, scipy.__version__)"
```

## Example: Triplet Reference Calculation

The source provides example geometries for:

- water;
- ethylene;
- oxygen;
- methyl.

Example:

```bash
python R-MR-KS-TDDFT.py     --molecule water     --basis sto-3g     --functional b3lyp     --n-radial 100     --grid-order 14     --max-iter 100     --print-level 1     --output-dir ./results
```

To avoid saving results:

```bash
python R-MR-KS-TDDFT.py     --molecule water     --basis sto-3g     --functional b3lyp     --no-save
```

The default molecule is `water`.

## Python API Example

The intended workflow is approximately:

```python
from new import (
    run_triplet_reference,
    align_spatial_orbitals,
    construct_mr_rdm_from_results,
    perform_spin_projection,
    perform_response_calculation,
    solve_response_with_davidson,
)

molecule = '''
O  0.0  0.0  0.0
H  0.0  0.75  0.5
H  0.0 -0.75  0.5
'''

# Step 1: triplet UKS reference
triplet_result = run_triplet_reference(
    molecule_string=molecule,
    basis="sto-3g",
    functional="b3lyp",
)

# Step 2: align alpha/beta orbitals
alignment_result = align_spatial_orbitals(triplet_result)

# Step 3: construct MR-RDM
mr_rdm_result = construct_mr_rdm_from_results(
    triplet_result,
    alignment_result
)

# Step 4: singlet/triplet projection
spin_projection_result = perform_spin_projection(
    triplet_result,
    alignment_result,
    mr_rdm_result
)

# Step 5: response calculation
response_result = perform_response_calculation(
    triplet_result,
    alignment_result,
    mr_rdm_result,
    spin_projection_result,
    molecular_grid,
    basis_set,
    xc_functional,
    n_excitations=10,
)

# Step 6: Davidson diagonalization
davidson_results = solve_response_with_davidson(
    triplet_result,
    alignment_result,
    mr_rdm_result,
    spin_projection_result,
    response_result,
    n_singlet=5,
    n_triplet=5,
)

singlet_energies = davidson_results["singlet"].eigenvalues
triplet_energies = davidson_results["triplet"].eigenvalues
```

The objects `molecular_grid`, `basis_set`, and `xc_functional` must be
constructed using the local DFT/grid implementation before calling the
response routine.

## Output Files

### Triplet reference

Typical files:

```text
results/
├── triplet_ref_<molecule>_<basis>_<functional>.json
└── triplet_ref_<molecule>_<basis>_<functional>_orbitals.npz
```

The orbital NPZ file contains:

```text
C_alpha
C_beta
eps_alpha
eps_beta
Da
Db
```

### Aligned orbitals

Typical file:

```text
aligned_<molecule>_<basis>_<functional>_orbitals.npz
```

Contains quantities including:

```text
C_alpha_aligned
C_beta_aligned
rot_alpha
rot_beta
cross_overlap
initial_overlap
n_alpha
n_beta
n_basis
iterations
convergence
converged
```

### MR-RDM

Typical file:

```text
mr_rdm_<molecule>_<basis>_<functional>.npz
```

Contains:

```text
P_MR
P_alpha
P_beta
C_alpha_aligned
C_beta_aligned
h_index
p_index
h_energy
p_energy
gap
n_alpha
n_beta
n_core
n_basis
n_occ
trace
idempotent
```

### Spin projection

Typical file:

```text
spin_projection_<molecule>_<basis>_<functional>.npz
```

### Response calculation

Typical file:

```text
response_<molecule>_<basis>_<functional>.npz
```

### Davidson calculation

Davidson results contain:

```text
eigenvalues
eigenvectors
residuals
converged
iterations
n_roots
time
matrix_vector_ops
```

## Verification

Each major stage provides a verification routine.

### Triplet state

```python
verification = verify_triplet_state(triplet_result)
```

Checks include electron-count difference, two unpaired electrons, approximate
triplet `<S^2>`, and SCF convergence.

### Orbital alignment

```python
verification = verify_alignment(alignment_result)
```

Checks include cross-overlap behavior and rotation-matrix orthogonality.

### MR-RDM

```python
verification = verify_mr_rdm(mr_rdm_result, S=S)
```

Checks include trace, idempotency, active-space occupation, spin density, and
symmetry.

### Spin projection

```python
verification = verify_spin_projection(spin_projection_result)
```

Checks include orthogonality, idempotency, trace conservation, transition
counts, and real-valued matrices.

### Response matrix

```python
verification = verify_response_matrix(response_result)
```

Checks include excitation energies, singlet/triplet separation, matrix
symmetry, and transition-vector normalization.

### Davidson solver

```python
verification = verify_davidson_results(davidson_results)
```

Checks include eigenvalue positivity, convergence, eigenvector normalization,
and residual magnitudes.

## Computational Notes

### SCF settings

The default UKS settings include:

```text
Energy convergence     = 1e-8 Hartree
Density convergence    = 1e-6
Maximum SCF iterations = 100
DIIS start             = iteration 5
Maximum DIIS vectors   = 10
```

These can be changed through `run_triplet_reference()`.

### Molecular integration grid

The grid is controlled primarily through:

```text
n_radial
grid_order
```

Increasing grid resolution increases computational cost.

### Tamm-Dancoff approximation

The response implementation uses the Tamm-Dancoff approximation (TDA) by
default and treats the response problem in the form:

```text
A X = omega X
```

### Davidson solver

The Davidson implementation uses matrix-vector products, a diagonal
preconditioner, iterative subspace expansion, and restart capability.

## Current Implementation Limitations

The supplied source is explicitly a prototype and contains several
simplifications or placeholders.

### Orbital alignment

`compute_jacobi_rotation()` currently contains an approximate/placeholder
choice for the occupied-space size. Production use should propagate the
actual occupation information consistently.

### MR-RDM

The MR-RDM implementation constructs the specified 50/50 active-space
mixture. Its density construction and idempotency treatment should be
validated carefully against the intended MR-SF-TDDFT formalism before
quantitative production calculations.

### Spin projection

The spin-projection routines contain simplified transition-space and
full-projection representations. These should be validated against a
formal spin-adapted implementation.

### XC kernel

The spin-flip kernel includes simplified analytical approximations for B3LYP
and LDA and a numerical fallback for other functionals.

The B3LYP-related coefficients used in the source include:

```text
HF exchange fraction = 0.20
Becke88 fraction     = 0.72
LYP correlation      = 0.81
```

These should not be assumed to constitute a complete production-quality
spin-flip B3LYP kernel.

### Response matrix

The current response implementation contains simplified treatment of
exchange and XC contributions. The source notes that full two-electron
integrals would be required for a production implementation.

### File-based response and Davidson workflows

The functions:

```python
build_response_from_files(...)
solve_response_from_files(...)
```

contain placeholder implementations and should not be treated as complete
restart workflows.

## Recommended Project Structure

For maintainability, the current monolithic source can eventually be split
into:

```text
MR-SF-TDDFT/
├── README.md
├── requirements.txt
├── environment.yml
├── UKS.py
├── helper_KS.py
├── triplet_reference.py
├── orbital_alignment.py
├── mr_rdm.py
├── spin_projection.py
├── response.py
├── davidson.py
├── examples/
│   ├── water.py
│   ├── oxygen.py
│   └── methyl.py
├── tests/
│   ├── test_uks.py
│   ├── test_alignment.py
│   ├── test_mr_rdm.py
│   ├── test_spin_projection.py
│   ├── test_response.py
│   └── test_davidson.py
└── results/
```

This structure would make the theoretical components easier to test,
validate, and reuse.

## Scientific References

The source cites:

- Szabo and Ostlund, *Modern Quantum Chemistry*.
- Helgaker et al., *Molecular Electronic-Structure Theory*.
- Casida et al., work on molecular excitation energies from TDDFT.
- Shao et al., work on spin-flip density functional theory.
- Ziegler et al., work on spin-flip density functional theory.
- Vydrov et al., work on spin-flip density functional theory.
- Werner et al., *MOLPRO: a package of ab initio programs*.
- Davidson, work on iterative calculation of a few lowest eigenvalues.
- Olsen et al., work on the Davidson method.

The supplied source does not provide complete bibliographic metadata, so a
formal reference list should be added separately.

## Development Status

```text
Research prototype / development version
```

Recommended development path:

```text
UKS reference
    |
    v
Validated spatial orbital alignment
    |
    v
Validated MR-RDM
    |
    v
Validated spin projection
    |
    v
Validated spin-flip XC kernel
    |
    v
Validated TDA response matrix
    |
    v
Davidson solver
    |
    v
Benchmark against independent implementations
    |
    v
MR-SF-TDDFT excitation energies
    |
    v
SAPT(MR-SF-TDDFT)
```

Before production use, each stage should be tested against analytical limits,
small benchmark systems, and independent quantum-chemistry implementations.

## License

No license is specified in the supplied source. Add an appropriate license
before distributing the code publicly.

## Acknowledgement

This README was prepared from the functionality, interfaces, examples,
comments, and implementation details present in the supplied
`R-MR-KS-TDDFT.py` source file.
