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
| `rotated.molden` | Molden file of the reference with rotated MO coefficients and target coordinates |
| `ALTER.txt` | OpenMolcas `ALTER` keyword block for reordering the target active space |

---

## Requirements

- **Python ≥ 3.8**
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)
- [sphecerix](https://github.com/ifilot/sphecerix) – Wigner D-matrix library for real (tesseral) spherical harmonics
- **orbkit** (modified) – included in this repository as a Git submodule; the upstream version has been patched to correctly parse Molden files produced by **OpenMolcas**

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
pip install numpy scipy sphecerix
```

---

## Usage

```
python active_space_selection.py \
    --ref   <ref1.molden> [<ref2.molden> ...] \
    --target <target.molden> \
    --ref_orbitals <orb_list_1> [<orb_list_2> ...] \
    --active_space <active_orb_list> \
    [--atoms <atom_list>]
```

### Arguments

| Argument | Required | Description |
|---|---|---|
| `--ref` | yes | One or more reference Molden files (space-separated). |
| `--target` | yes | Single target Molden file. Output files are written to the same directory. |
| `--ref_orbitals` | yes | Active-space orbital indices (1-based) for each reference file. Use commas for individual orbitals, hyphens for ranges, and colons to separate lists for different reference files. Example: `1-5,7:2-6,9` supplies `[1,2,3,4,5,7]` for the first reference and `[2,3,4,5,6,9]` for the second. |
| `--active_space` | yes | Active-space orbital indices (1-based) of the target system. Example: `4,6-9` → `[4,6,7,8,9]`. |
| `--atoms` | no | Subset of atom indices (1-based) to use for the Kabsch alignment. Useful when only part of the molecule is rigid (e.g., exclude a flexible substituent). If omitted, all atoms are used. Example: `1,3-5,8`. |

### Example – phenol test case

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

---

## Notes

- The Molden files must be in **OpenMolcas** format. Molden files generated by other codes (Gaussian, ORCA, …) may require minor adjustments to the `[Atoms]` section units and the MO block header keywords.
- The orbital indices supplied to `--ref_orbitals` and `--active_space` are **1-based** (as printed by OpenMolcas and as shown in standard MO visualisation tools).
- When multiple reference files are supplied, the tool picks the one with the **lowest RMSD** to the target and prints the RMSD values for all references to standard output, allowing you to verify the quality of the alignment.
- Overlaps with `|S| < 0.8` or `|S| > 1.2` are flagged as warnings in `ALTER.txt` and on standard output; always inspect these manually before submitting the next calculation.
