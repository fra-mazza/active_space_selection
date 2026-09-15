# active_space_selection

A tool for automatically selecting and tracking active spaces in multi-reference quantum chemistry calculations across different molecular geometries.

## Please cite

Mazza, F., Trinari, M., Sepali, C. and Cappelli, C., 2026. Analytical Nuclear Gradients for the Multiconfigurational Self-Consistent Field Method Coupled with the Polarizable Fluctuating Charges Model. *Journal of Chemical Theory and Computation*, 22(3), pp.1350-1362.

## Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage: `active_space_selection.py`](#usage-active_space_selectionpy)
- [Usage: `get_pi_orbitals.py`](#usage-get_pi_orbitalspy)
- [Usage: `combine_alter_files.py`](#usage-combine_alter_filespy)
- [How it works](#how-it-works)
- [Tests](#tests)
- [Notes and caveats](#notes-and-caveats)
- [Troubleshooting](#troubleshooting)

---

## Overview

When you run a **CASSCF** (or other multi-reference) calculation at several points along a reaction path, on a PES scan, or across a series of related molecules, OpenMolcas has no memory of which orbitals belong in the active space from one geometry to the next: the ordering and shape of the MOs can shift, and re-selecting the active space by hand at every point is slow and error-prone. This repository automates that orbital-tracking problem.

The package has three command-line scripts:

| Script | Purpose |
|---|---|
| [`active_space_selection.py`](#usage-active_space_selectionpy) | Main tool. Aligns a reference geometry onto a target geometry, rotates the reference MOs into the target's orientation, and matches them to the target MOs by overlap, producing an OpenMolcas `ALTER` block. |
| [`get_pi_orbitals.py`](#usage-get_pi_orbitalspy) | Ranks MOs by π-character relative to a best-fit plane through a chosen set of atoms — useful for picking out the π/π* active space of a (quasi-)planar aromatic system without visual inspection. |
| [`combine_alter_files.py`](#usage-combine_alter_filespy) | Merges several partial `ALTER` files (e.g. one per independently-tracked molecular fragment) into a single `ALTER` block for the whole system. |

`active_space_selection.py` accepts either **Molden** or **OpenMolcas HDF5** (`.h5`) files as input. The two modes are numerically equivalent (see [How it works](#how-it-works)); HDF5 is faster because it reuses the AO overlap matrix already stored in the file instead of recomputing it.

### The `active_space_selection.py` workflow

1. **Geometry alignment** — Given one or more *reference* files and a single *target* file, the tool finds the optimal rigid-body rotation that superimposes each reference geometry onto the target using the [Kabsch algorithm](https://en.wikipedia.org/wiki/Kabsch_algorithm). When multiple references are given, the one with the lowest RMSD to the target is used.
2. **MO coefficient rotation** — The MO coefficients of the best-matching reference are rotated shell-by-shell in the AO basis using Wigner D-matrices (via the [sphecerix](https://github.com/ifilot/sphecerix) library), so the rotated reference MOs are expressed in the target's orientation.
3. **Orbital overlap calculation** — The overlap between each rotated reference active-space orbital and every target orbital is computed (via **orbkit** for Molden input, or directly from the file for HDF5 input). Each reference orbital is matched to the target orbital with the highest overlap.
4. **`ALTER` file generation** — An `ALTER.txt` file is written containing an OpenMolcas `ALTER` keyword block that reorders the target's active space to match the reference. Low-overlap matches (`|S| < 0.8` or `> 1.2`) are flagged as warnings for manual review.

---

## Requirements

- **Python ≥ 3.8**
- [NumPy](https://numpy.org/) **≥ 1.20**
- [SciPy](https://scipy.org/) **≥ 1.7, < 1.15** — SciPy ≥ 1.15 changes the `sph_harm` API and breaks sphecerix 0.5.0
- [sphecerix](https://github.com/ifilot/sphecerix) **== 0.5.0** — Wigner D-matrix library for real (tesseral) spherical harmonics
- [Cython](https://cython.org/) — needed to compile orbkit's C extensions at install time
- **orbkit** (modified) — vendored in this repository as a Git submodule, patched to correctly parse OpenMolcas Molden files. Only needed for Molden input; not used in HDF5 mode. Pulls in [matplotlib](https://matplotlib.org/) and [scikit-image](https://scikit-image.org/).
- [h5py](https://www.h5py.org/) **≥ 3.0** — only needed for OpenMolcas HDF5 (`.h5`) input

---

## Installation

### 1. Clone with the orbkit submodule

```bash
git clone --recurse-submodules https://github.com/fra-mazza/active_space_selection.git
cd active_space_selection
```

Already cloned without `--recurse-submodules`? Initialise it separately:

```bash
git submodule update --init --recursive
```

### 2. Install build prerequisites

orbkit's Cython extensions need NumPy and Cython present *before* it is built:

```bash
pip install "numpy>=1.20" cython setuptools
```

### 3. Install the modified orbkit

```bash
pip install --no-build-isolation --config-settings editable_mode=compat -e orbkit/
```

Both flags matter and are explained in [Troubleshooting](#troubleshooting):

- `--no-build-isolation` — without it, pip builds orbkit in a fresh environment that doesn't have Cython, and the install fails with `ModuleNotFoundError: No module named 'Cython'`.
- `--config-settings editable_mode=compat` — without it, running any script from the repository root (exactly what the examples below do) makes Python silently import the *wrong* orbkit and fail deep inside with a confusing `ImportError`.

**On macOS**, Apple's `clang` doesn't support `-fopenmp`, which orbkit's `detci` submodule needs, so point the build at a real GCC:

```bash
# Conda (recommended):
conda install -c conda-forge gcc
CC=gcc pip install --no-build-isolation --config-settings editable_mode=compat -e orbkit/

# Homebrew:
brew install gcc
CC=$(ls /opt/homebrew/bin/gcc-* | sort -V | tail -1) pip install --no-build-isolation --config-settings editable_mode=compat -e orbkit/
```

### 4. Install the remaining Python dependencies

```bash
pip install "scipy>=1.7,<1.15" "sphecerix==0.5.0" "h5py>=3.0" matplotlib
```

(`scipy<1.15` and the explicit `matplotlib` are both required for reasons that only bite in certain install orders — see [Troubleshooting](#troubleshooting).)

### 5. Verify the install

```bash
python -c "import orbkit.tools; print('orbkit OK:', orbkit.tools.__file__)"
python -m pytest test/test_suite.py -v
```

The first command must print a path ending in `.../orbkit/orbkit/tools.py` — see [Troubleshooting](#troubleshooting) if it doesn't. The test suite (60+ tests, no external QM software needed) is the recommended way to confirm the whole install — including the orbkit build — actually works; see [Tests](#tests).

---

## Usage: `active_space_selection.py`

```
python active_space_selection.py \
    --ref   <ref1.molden|ref1.h5> [<ref2.molden|ref2.h5> ...] \
    --target <target.molden|target.h5> \
    --ref_orbitals <orb_list_1> [<orb_list_2> ...] \
    (--active_space <active_orb_list> | --act_elect <n> --act_orb <n>) \
    [--atoms <atom_list>]
```

### Arguments

| Argument | Required | Description |
|---|---|---|
| `--ref` | yes | One or more reference Molden or HDF5 files (space-separated). All files must be of the same type. |
| `--target` | yes | Single target Molden or HDF5 file. Output files are written to the same directory. |
| `--ref_orbitals` | yes | Active-space orbital indices (1-based) for each reference file. Commas separate individual orbitals, hyphens denote ranges, colons separate the lists for different reference files. Example: `1-5,7:2-6,9` supplies `[1,2,3,4,5,7]` for the first reference and `[2,3,4,5,6,9]` for the second. |
| `--active_space` | one of `--active_space` / `--act_elect`+`--act_orb` | Active-space orbital indices (1-based) of the target system. Example: `4,6-9` → `[4,6,7,8,9]`. |
| `--act_elect`, `--act_orb` | one of `--active_space` / `--act_elect`+`--act_orb` | Instead of listing target active-space orbitals explicitly, specify the number of active electrons and active orbitals; the target's active space is computed automatically around the HOMO/LUMO. **Only valid for integer-occupation (e.g. SCF) target orbitals** — occupations that aren't exactly 0 or 2 (e.g. genuine CASSCF natural orbitals) raise an error, since the split between active-occupied and active-virtual is then ambiguous. Must be given together; mutually exclusive with `--active_space`. |
| `--atoms` | no | Subset of atom indices (1-based) to use for the Kabsch alignment. Useful when only part of the molecule is rigid (e.g. exclude a flexible substituent). If omitted, all atoms are used. Example: `1,3-5,8`. |

### Output files (written to the target file's directory)

| File | Contents |
|---|---|
| `rotate.csv` | 3×3 rotation matrix used to align the reference onto the target |
| `rotated.molden` | *(Molden mode only)* the reference, with MO coefficients rotated and target coordinates |
| `rotated.h5` | *(HDF5 mode only)* the reference, with MO coefficients rotated and target coordinates |
| `ALTER.txt` | OpenMolcas `ALTER` keyword block for reordering the target's active space |

### Example — phenol (Molden)

`test/phenol_scf/` contains a ready-to-use reference/target pair for phenol, with the orbital lists in `test/ref_orbitals.txt` and `test/active_orbitals.txt`.

```bash
REF_ORBS=$(cat test/ref_orbitals.txt)   # 19,23-27,34
ACTIVE=$(cat test/active_orbitals.txt)  # 22-28

python active_space_selection.py \
    --ref    test/phenol_scf/ref.molden \
    --target test/phenol_scf/target.molden \
    --ref_orbitals "$REF_ORBS" \
    --active_space "$ACTIVE"
```

`rotate.csv`, `rotated.molden`, and `ALTER.txt` appear in `test/phenol_scf/`.

### Example — phenol (HDF5)

The same case with OpenMolcas HDF5 files. HDF5 input reuses the AO overlap matrix already stored in the file, **skipping orbkit's analytical-integral step**:

```bash
python active_space_selection.py \
    --ref    test/phenol_scf/ref.h5 \
    --target test/phenol_scf/target.h5 \
    --ref_orbitals "$REF_ORBS" \
    --active_space "$ACTIVE"
```

`rotate.csv`, `rotated.h5`, and `ALTER.txt` appear in `test/phenol_scf/`.

---

## Usage: `get_pi_orbitals.py`

Given a (quasi-)planar set of atoms, this ranks all MOs by how much π-character they have relative to that plane — a per-atom perpendicular-p projector, weighted so compact (inner) contracted shells dominate over diffuse ones. It's a way to pick out a π/π* active space without opening a viewer, and can optionally build an `ALTER` block from the top-ranked π orbitals directly.

```
python get_pi_orbitals.py \
    --target <target.molden|target.h5> \
    --atoms <atom_list> \
    [--top_n <n>] \
    [--planarity_threshold <value>] \
    (--active_space <active_orb_list> | --act_elect <n> --act_orb <n>) \
    [--alter <alter_file>]
```

### Arguments

| Argument | Required | Description |
|---|---|---|
| `--target` | yes | Target Molden or HDF5 file. |
| `--atoms` | yes | 1-based atom indices defining the planar aromatic subset. Example: `1,3-8`. |
| `--top_n` | no | Number of orbitals to display, ranked by π-score (default 20). |
| `--planarity_threshold` | no | Warn if the max atom-to-plane distance exceeds this value, same units as the input coordinates (default 0.10). |
| `--active_space` | no | Active-space orbital indices (1-based). If given, the script also selects the top *N* π-ranked orbitals (N = active-space size) and writes an `ALTER` block swapping any mismatch. Mutually exclusive with `--act_elect`/`--act_orb`. |
| `--act_elect`, `--act_orb` | no | Same automatic active-space computation as in `active_space_selection.py` — same integer-occupation restriction applies. Must be given together; mutually exclusive with `--active_space`. |
| `--alter` | no | Output path for the generated `ALTER` file (requires `--active_space` or `--act_elect`/`--act_orb`). Defaults to `ALTER.txt` next to `--target`. |

### Example — phenol ring π orbitals

```bash
python get_pi_orbitals.py \
    --target test/phenol_scf/ref.molden \
    --atoms 1-6 \
    --top_n 7
```

The top 7 orbitals by π-score reproduce phenol's known active space `{19,23,24,25,26,27,34}` — this is checked in the test suite as a genuine correctness cross-check, since that active space was chosen independently of this script.

---

## Usage: `combine_alter_files.py`

If distinct molecular regions (e.g. two aromatic rings tracked separately) each produced their own partial `ALTER` file, merge them into one file for the whole active space:

```
python combine_alter_files.py \
    -i <ALTER1.txt> <ALTER2.txt> [<ALTER3.txt> ...] \
    --active_spaces <active_list_1>[:<active_list_2>[:...]] \
    --total_active_space <full_active_list> \
    [-o <ALTER_total.txt>]
```

### Arguments

| Argument | Required | Description |
|---|---|---|
| `-i`, `--input` | yes | List of input `ALTER` files to combine (space-separated). |
| `--active_spaces` | yes | Active-space list used to generate each input file, same order as `--input`, colon-separated. Example: `19-23:30-34`. |
| `--total_active_space` | yes | Total active space of the full molecule. Example: `19-23,30-34`. |
| `-o`, `--output` | no | Output path for the merged `ALTER` block (default `ALTER.txt`). |

### Example

```bash
python combine_alter_files.py \
    -i ALTER1.txt ALTER2.txt \
    --active_spaces 19-23:30-34 \
    --total_active_space 19-23,30-34 \
    -o ALTER_total.txt
```

---

## How it works

### HDF5 mode internals

OpenMolcas HDF5 checkpoint files (`.h5`) contain, among other things:

| Dataset | Contents |
|---|---|
| `CENTER_COORDINATES` | Atomic coordinates (Bohr) |
| `BASIS_FUNCTION_IDS` | Per-AO metadata: center index, shell index, *l*, *m* |
| `MO_VECTORS` | MO coefficient matrix, stored row-major (MO × AO) |
| `AO_OVERLAP_MATRIX` | Pre-computed AO overlap matrix |

`active_space_selection.py` uses these directly: coordinates for the Kabsch alignment, `BASIS_FUNCTION_IDS` to identify one rotation block per `(atom, l, radial shell)` triplet (rotated the same way as for Molden), and the pre-computed `AO_OVERLAP_MATRIX` to evaluate `S_MO = C_rot · S_AO · C_tar^T` without ever calling orbkit. The Molden and HDF5 pathways produce numerically identical orbital mappings and `ALTER.txt` files.

### Angular momentum support

The Wigner-D rotation of AO shells is implemented for **s, p, d, f, and g functions (l = 0–4)**. Basis sets with h functions or higher (l ≥ 5) are not currently supported.

---

## Tests

A single command runs the entire suite — the recommended way to confirm right after installing that everything (including the orbkit build) actually works:

```bash
python -m pytest test/test_suite.py -v
```

All 60+ tests run against the one real QM calculation checked into `test/phenol_scf/` (a closed-shell SCF wavefunction for phenol, s/p/d basis) — no external QM software is required. Coverage includes:

- HDF5 file-type detection and coordinate extraction (HDF5 vs Molden agreement)
- Rotation-block coverage (all AOs covered exactly once) and Wigner-D unitarity
- HDF5 and Molden workflows produce the same orbital mapping and `ALTER.txt`
- Mixed-type inputs (one Molden, one HDF5) are rejected with a clear error
- The `--act_elect`/`--act_orb` automatic active-space feature
- `get_pi_orbitals.py`: its purely-geometric π-character ranking is checked against the phenol active space, which was chosen independently of that script — a genuine correctness cross-check, not a self-consistency tautology
- `combine_alter_files.py`: splitting the active space into two disjoint pieces, running the main script on each, and combining the results must match a single full-active-space run
- CLI / argument-validation edge cases across all three scripts
- Determinism: repeated runs on identical inputs give byte-identical output files

This single closed-shell SCF/s-p-d test case can't exercise everything — genuine CASSCF active spaces with fractional natural-orbital occupations, f/g basis functions, multi-reference RMSD selection, and `--atoms`-subset alignment are planned but need purpose-built QM calculations not yet part of the repository.

---

## Notes and caveats

- Molden files must be in **OpenMolcas** format. Files from other codes (Gaussian, ORCA, …) may need minor adjustments to the `[Atoms]` section units and the MO block header keywords.
- Orbital indices in `--ref_orbitals` / `--active_space` / `--atoms` are **1-based**, matching OpenMolcas output and standard MO viewers.
- With multiple reference files, the tool picks the one with the **lowest RMSD** to the target and prints all references' RMSD values so you can check alignment quality.
- Overlaps with `|S| < 0.8` or `|S| > 1.2` are flagged as warnings in `ALTER.txt` and on stdout — always inspect these manually before submitting the next calculation.
- All `--ref` and `--target` files must be the same type (all Molden **or** all HDF5); mixing formats is rejected with an error.
- `--act_elect`/`--act_orb` (both scripts) only work when every input orbital is fully occupied or fully virtual (occupation exactly 0 or 2, e.g. SCF/localized orbitals) — genuine CASSCF natural orbitals with fractional occupations must use `--active_space` with an explicit orbital list instead.

---

## Troubleshooting

### `ImportError: cannot import name '...' from 'orbkit....' (unknown location)`

This happens if orbkit was installed *without* `--config-settings editable_mode=compat` (step 3) and a script is then run from the repository root — exactly what the usage examples above do.

The repository checks out the orbkit submodule into a directory that is itself called `orbkit/`, so the layout is `active_space_selection/orbkit/orbkit/…`. Modern setuptools (≥ 64) makes editable installs work through a `sys.meta_path` finder that is registered *after* Python's normal path-based import machinery. Because the outer `orbkit/` submodule directory sits right next to `active_space_selection.py` and has no `__init__.py`, Python's normal import machinery treats it as a *namespace package* called `orbkit` and resolves imports against it **before** the editable finder gets a chance to point at the real, compiled package.

Fix: reinstall with the compat flag (uninstall first):

```bash
pip uninstall orbkit
pip install --no-build-isolation --config-settings editable_mode=compat -e orbkit/
```

`editable_mode=compat` falls back to the older, simpler editable-install mechanism (a `.pth` file that adds the real package directory to `sys.path`), which doesn't have this problem regardless of the current working directory. This flag needs a reasonably recent pip (≥ 23); if it's rejected as an unknown option, run `pip install --upgrade pip` first.

Verify the fix:

```bash
cd active_space_selection   # repository root
python -c "import orbkit.tools; print(orbkit.tools.__file__)"
```

This must print a path ending in `.../orbkit/orbkit/tools.py`. A shorter path (missing the second `orbkit/`) or an `ImportError` means the broken namespace-package resolution is still happening.

### `ModuleNotFoundError: No module named 'Cython'` while installing orbkit

`orbkit/setup.py` imports Cython at module level, but modern pip (≥ 21.3) builds packages in a fresh, isolated environment by default (PEP 517) that doesn't have Cython. Install with `--no-build-isolation` (as in step 3 above) so pip uses the current environment, which already has Cython from step 2.

### `ModuleNotFoundError: No module named 'matplotlib'` when importing sphecerix

`sphecerix` 0.5.0 imports `matplotlib` unconditionally at import time (`sphecerix/__init__.py` → `matrixplot.py` → `import matplotlib.pyplot`) without declaring it as a dependency. Installing orbkit (step 3) happens to pull in matplotlib as a side effect, which is normally the only reason this doesn't fail — installing matplotlib explicitly in step 4 removes that hidden ordering dependency.

### macOS: `-fopenmp` rejected by clang

Apple's default `clang` doesn't support OpenMP, which orbkit's `detci` submodule requires. Set `CC` to a real GCC before installing (see the macOS instructions in step 3) — GCC ships with built-in OpenMP support, so the `-fopenmp` flag orbkit passes when compiling `detci/cy_ci.pyx` is accepted.
