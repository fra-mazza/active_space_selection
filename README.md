# active_space_selection

A tool for automatically selecting and tracking active spaces in multi-reference quantum chemistry calculations across different molecular geometries.

---

## Description

`active_space_selection.py` automates the orbital mapping problem that arises when running **CASSCF** (or similar multi-reference) calculations along a reaction path or across a series of related geometries. When the molecular geometry changes, the ordering and shape of the molecular orbitals (MOs) can change too, making it difficult to maintain a consistent active space.

The workflow implemented by this tool is:

1. **Geometry alignment** – Given one or more *reference* Molden files and a single *target* Molden file, the tool finds the optimal rigid-body rotation that superimposes the reference geometry onto the target geometry using the [Kabsch algorithm](https://en.wikipedia.org/wiki/Kabsch_algorithm). When multiple reference files are provided, the one with the lowest RMSD is selected automatically.

2. **MO coefficient rotation** – The MO coefficients of the best-matching reference are rotated shell-by-shell in the AO basis using Wigner D-matrices (via the **sphecerix** library), so that the rotated reference MOs are expressed in the same orientation as the target.

3. **Orbital overlap calculation** – The overlap matrix between the rotated reference MOs and the target MOs is computed using **orbkit**. Each reference active-space orbital is matched to the target orbital with the highest overlap.

4. **ALTER file generation** – An `ALTER.txt` file is written to the target directory containing an **OpenMolcas** `ALTER` keyword block that swaps orbitals in the target's active space to match the reference ordering. Low-overlap matches are flagged with warnings.

### Output files (written to the target file's directory)

| File | Contents |
|---|---|
| `rotate.csv` | 3×3 rotation matrix (CSV) used to align the reference onto the target |
| `rotated.molden` | *(Molden mode only)* Molden file of the reference with rotated MO coefficients and target coordinates |
| `rotated.h5` | *(HDF5 mode only)* HDF5 file of the reference with rotated MO coefficients and target coordinates |
| `ALTER.txt` | OpenMolcas `ALTER` keyword block for reordering the target active space |

---

## Requirements

- **Python ≥ 3.8**
- [NumPy](https://numpy.org/) **≥ 1.20**
- [SciPy](https://scipy.org/) **≥ 1.7, < 1.15** – SciPy ≥ 1.15 changes the `sph_harm` API and breaks sphecerix 0.5.0
- [sphecerix](https://github.com/ifilot/sphecerix) **== 0.5.0** – Wigner D-matrix library for real (tesseral) spherical harmonics
- **orbkit** (modified) – included in this repository as a Git submodule; the upstream version has been patched to correctly parse Molden files produced by **OpenMolcas**. Only required when using Molden input files.
- [h5py](https://www.h5py.org/) **≥ 3.0** – required only when using OpenMolcas HDF5 (`.h5`) input files

---

## Installation

### 1. Clone the repository (with the orbkit submodule)

```bash
git clone --recurse-submodules https://github.com/fra-mazza/active_space_selection.git
cd active_space_selection
```

If you already cloned without `--recurse-submodules`, initialise the submodule afterwards:

```bash
git submodule update --init --recursive
```

### 2. Install the modified orbkit

The modified orbkit lives in the `orbkit/` subdirectory. Install it in editable mode so that any local patches are picked up automatically:

```bash
pip install -e orbkit/
```

### 3. Install the remaining Python dependencies

```bash
pip install "numpy>=1.20" "scipy>=1.7,<1.15" "sphecerix==0.5.0" "h5py>=3.0"
```

---

## Usage

```
python active_space_selection.py \
    --ref   <ref1.molden|ref1.h5> [<ref2.molden|ref2.h5> ...] \
    --target <target.molden|target.h5> \
    --ref_orbitals <orb_list_1> [<orb_list_2> ...] \
    --active_space <active_orb_list> \
    [--atoms <atom_list>]
```

### Arguments

| Argument | Required | Description |
|---|---|---|
| `--ref` | yes | One or more reference Molden or HDF5 files (space-separated). All files must be of the same type. |
| `--target` | yes | Single target Molden or HDF5 file. Output files are written to the same directory. |
| `--ref_orbitals` | yes | Active-space orbital indices (1-based) for each reference file. Use commas for individual orbitals, hyphens for ranges, and colons to separate lists for different reference files. Example: `1-5,7:2-6,9` supplies `[1,2,3,4,5,7]` for the first reference and `[2,3,4,5,6,9]` for the second. |
| `--active_space` | yes | Active-space orbital indices (1-based) of the target system. Example: `4,6-9` → `[4,6,7,8,9]`. |
| `--atoms` | no | Subset of atom indices (1-based) to use for the Kabsch alignment. Useful when only part of the molecule is rigid (e.g., exclude a flexible substituent). If omitted, all atoms are used. Example: `1,3-5,8`. |

### Example – phenol test case (Molden)

The `test/phenol_scf/` directory contains a ready-to-use example with reference and target Molden files for phenol. The active-space orbitals are listed in `test/active_orbitals.txt` and the reference orbitals in `test/ref_orbitals.txt`.

```bash
# Read the orbital lists from the provided text files
REF_ORBS=$(cat test/ref_orbitals.txt)   # e.g. 19,23-27,34
ACTIVE=$(cat test/active_orbitals.txt)  # e.g. 22-28

python active_space_selection.py \
    --ref    test/phenol_scf/ref.molden \
    --target test/phenol_scf/target.molden \
    --ref_orbitals "$REF_ORBS" \
    --active_space "$ACTIVE"
```

After the run, the files `rotate.csv`, `rotated.molden`, and `ALTER.txt` will appear inside `test/phenol_scf/`.

### Example – phenol test case (HDF5)

The same test case can be run with OpenMolcas HDF5 files.  When HDF5 input is used the AO overlap matrix is read directly from the file, **skipping the expensive analytical-integral step** performed by orbkit in the Molden workflow.

```bash
REF_ORBS=$(cat test/ref_orbitals.txt)
ACTIVE=$(cat test/active_orbitals.txt)

python active_space_selection.py \
    --ref    test/phenol_scf/ref.h5 \
    --target test/phenol_scf/target.h5 \
    --ref_orbitals "$REF_ORBS" \
    --active_space "$ACTIVE"
```

The output `rotate.csv`, `rotated.h5`, and `ALTER.txt` are written to `test/phenol_scf/`.

---

## HDF5 mode – how it works

OpenMolcas writes an HDF5 checkpoint file (`.h5`) that contains, among other things:

| Dataset | Contents |
|---|---|
| `CENTER_COORDINATES` | Atomic coordinates (Bohr) |
| `BASIS_FUNCTION_IDS` | Per-AO metadata: center index, shell index, *l*, *m* |
| `MO_VECTORS` | MO coefficient matrix, stored row-major (MO × AO) |
| `AO_OVERLAP_MATRIX` | Pre-computed AO overlap matrix |

The tool exploits these datasets as follows:

1. **Geometry alignment** – coordinates are read from `CENTER_COORDINATES` and fed to the Kabsch algorithm (same as for Molden).
2. **MO rotation** – `BASIS_FUNCTION_IDS` is parsed to identify rotation blocks (one per `(atom, l, radial shell)` triplet). The same Wigner D-matrix rotation used for Molden is then applied shell-by-shell to the `MO_VECTORS` coefficients.
3. **Overlap calculation** – the pre-computed `AO_OVERLAP_MATRIX` from the target file is used directly to evaluate `S_MO = C_rot · S_AO · C_tar^T`, avoiding any call to orbkit.

The Molden and HDF5 pathways produce numerically identical orbital mappings and `ALTER.txt` files.

---

## Tests

```bash
python -m pytest test/test_h5_support.py -v
```

The test suite verifies:

- HDF5 file-type detection
- Coordinate extraction (HDF5 vs Molden agreement)
- Rotation-block coverage (all AOs are covered exactly once)
- Identity rotation is a no-op on MO coefficients
- Reference MOs are orthonormal under the AO overlap
- Wigner D matrices are unitary
- HDF5 and Molden workflows produce the same orbital mapping and `ALTER.txt`
- Mixed-type inputs (one Molden, one HDF5) are rejected with a clear error

---

## Notes

- The Molden files must be in **OpenMolcas** format. Molden files generated by other codes (Gaussian, ORCA, …) may require minor adjustments to the `[Atoms]` section units and the MO block header keywords.
- The orbital indices supplied to `--ref_orbitals` and `--active_space` are **1-based** (as printed by OpenMolcas and as shown in standard MO visualisation tools).
- When multiple reference files are supplied, the tool picks the one with the **lowest RMSD** to the target and prints the RMSD values for all references to standard output, allowing you to verify the quality of the alignment.
- Overlaps with `|S| < 0.8` or `|S| > 1.2` are flagged as warnings in `ALTER.txt` and on standard output; always inspect these manually before submitting the next calculation.
- All `--ref` and `--target` files must be of the same type (all Molden **or** all HDF5); mixing the two formats is not supported.
