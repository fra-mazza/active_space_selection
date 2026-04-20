#!/usr/bin/env python3
"""
Identify pi-like molecular orbitals for a (quasi-)planar atom subset.

Workflow:
1) Fit best plane through selected atoms.
2) Build per-atom p orbital(s) perpendicular to that plane in AO basis.
3) Compute |<MO|p_perp>| and accumulate over all selected-atom p projectors.
4) Rank MOs by accumulated pi-character and print top N.
"""

import argparse
import re
import numpy as np

try:
    import h5py
    _H5PY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _H5PY_AVAILABLE = False


L_TO_NFUNCS = {"s": 1, "p": 3, "d": 5, "f": 7, "g": 9, "h": 11}


def parse_range_or_value(item_str):
    item_str = item_str.strip()
    if "-" in item_str:
        start, end = map(int, item_str.split("-"))
        if end < start:
            raise argparse.ArgumentTypeError(f"Invalid range: '{item_str}'")
        return list(range(start, end + 1))
    return [int(item_str)]


def parse_mixed_list(input_str):
    s = input_str.strip().strip("[]")
    if not s:
        return []
    items = re.split(r"\s*,\s*", s)
    out = []
    for item in items:
        out.extend(parse_range_or_value(item))
    return out


class ParseMixedListAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        joined = " ".join(values)
        setattr(namespace, self.dest, parse_mixed_list(joined))


def is_h5_file(filepath):
    return filepath.lower().endswith(".h5") or filepath.lower().endswith(".hdf5")


def parse_molden_sections(filename):
    with open(filename, "r") as f:
        content = f.read()
    sections = re.split(r"(?m)(?=^\[)", content)
    parsed = []
    for sec in sections:
        sec = sec.strip()
        if sec:
            header = sec.splitlines()[0].strip()
            parsed.append((header, sec))
    return parsed


def extract_atom_coords_molden(filename):
    sections = parse_molden_sections(filename)
    for header, content in sections:
        if header.upper().startswith("[ATOMS]"):
            coords = []
            for line in content.splitlines()[1:]:
                parts = line.strip().split()
                if len(parts) >= 6:
                    _, x, y, z = parts[2:6]
                    coords.append([float(x), float(y), float(z)])
            return np.array(coords, dtype=float)
    raise ValueError(f"No [Atoms] section found in {filename}")


def extract_atom_coords_h5(filename):
    if not _H5PY_AVAILABLE:
        raise ImportError("h5py is required to read HDF5 files: pip install h5py")
    with h5py.File(filename, "r") as f:
        return np.array(f["CENTER_COORDINATES"], dtype=float)


def extract_atom_coords(filename):
    return extract_atom_coords_h5(filename) if is_h5_file(filename) else extract_atom_coords_molden(filename)


def parse_mo_block(mo_text):
    lines = mo_text.splitlines()
    blocks = []
    current = []
    for line in lines:
        if line.strip().upper() == "[MO]":
            continue
        if line.strip().upper().startswith("SYM="):
            if current:
                blocks.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        blocks.append(current)

    parsed = []
    for block in blocks:
        header = block[:4]
        coeffs = []
        for line in block[4:]:
            txt = line.strip()
            if re.match(r"^\d+", txt):
                parts = txt.split()
                if len(parts) >= 2:
                    coeffs.append((parts[0], parts[1]))
        parsed.append({"header": header, "coeffs": coeffs, "raw": block})
    return parsed


def parse_molden_gto_for_p_blocks(filename):
    with open(filename, "r") as f:
        content = f.read()
    match = re.search(r"(?si)\[GTO\].*?(?=\n\[|$)", content)
    if not match:
        raise ValueError("GTO section not found in Molden file.")
    lines = match.group(0).splitlines()[1:]

    p_blocks_by_atom = {}
    ao_idx = 0
    current_atom = None
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue

        # Molden [GTO] atom header can be "atom_index" or "atom_index 0".
        atom_header = re.match(r"^(\d+)(?:\s+(\d+))?$", line)
        if atom_header:
            current_atom = int(atom_header.group(1))
            continue

        shell_match = re.match(r"^([spdfghSPDFGH])\s+(\d+)\s*$", line)
        if shell_match:
            orb = shell_match.group(1).lower()
            nprim = int(shell_match.group(2))
            nfunc = L_TO_NFUNCS[orb]
            if current_atom is None:
                raise ValueError("Found shell data before atom index in [GTO] section.")
            if orb == "p":
                # Molden p ordering is assumed as [m=+1, m=-1, m=0], i.e. [px, py, pz]
                # in the OpenMolcas-style convention used in this repository.
                p_blocks_by_atom.setdefault(current_atom, []).append([ao_idx, ao_idx + 1, ao_idx + 2])
            ao_idx += nfunc
            i += nprim
            continue

    return p_blocks_by_atom, ao_idx


def load_molden_mo_coeff_matrix(filename):
    sections = parse_molden_sections(filename)
    mo_text = None
    for header, sec in sections:
        if header.strip().upper().startswith("[MO]"):
            mo_text = sec
            break
    if mo_text is None:
        raise ValueError("MO section not found in Molden file.")
    mo_blocks = parse_mo_block(mo_text)
    coeff_rows = []
    for block in mo_blocks:
        coeff_rows.append([float(v) for _, v in block["coeffs"]])
    return np.array(coeff_rows, dtype=float)


def load_h5_data(filename):
    if not _H5PY_AVAILABLE:
        raise ImportError("h5py is required to read HDF5 files: pip install h5py")
    with h5py.File(filename, "r") as f:
        n = int(np.sqrt(len(f["MO_VECTORS"])))
        C = np.array(f["MO_VECTORS"], dtype=float).reshape(n, n)
        S = np.array(f["AO_OVERLAP_MATRIX"], dtype=float).reshape(n, n)
        bf_ids = np.array(f["BASIS_FUNCTION_IDS"], dtype=int)
    return C, S, bf_ids


def infer_h5_atom_index_base(bf_ids, natoms):
    centers = bf_ids[:, 0]
    cmin, cmax = int(np.min(centers)), int(np.max(centers))
    if cmin <= 0 and cmax <= natoms - 1:
        return 0
    return 1


def h5_p_blocks_by_atom(bf_ids, natoms):
    p_blocks = {}
    base = infer_h5_atom_index_base(bf_ids, natoms)
    groups = {}
    for ao_idx, row in enumerate(bf_ids):
        center, shell, l, m = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        if l != 1:
            continue
        atom_1based = center + (1 - base)
        key = (atom_1based, shell)
        groups.setdefault(key, []).append((ao_idx, m))

    for (atom_1based, _shell), items in groups.items():
        m_to_idx = {m: idx for idx, m in items}
        if all(m in m_to_idx for m in (1, -1, 0)):
            # Map normal components [nx, ny, nz] onto [m=+1, m=-1, m=0] ~= [px, py, pz].
            p_blocks.setdefault(atom_1based, []).append([m_to_idx[1], m_to_idx[-1], m_to_idx[0]])
    return p_blocks


def best_fit_plane(coords):
    centroid = np.mean(coords, axis=0)
    centered = coords - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    normal = normal / np.linalg.norm(normal)
    distances = np.abs(centered @ normal)
    return centroid, normal, distances


def build_pi_projectors(nbasis, p_blocks_by_atom, selected_atoms, normal):
    projectors = []
    for atom in selected_atoms:
        for block in p_blocks_by_atom.get(atom, []):
            vec = np.zeros(nbasis, dtype=float)
            vec[block[0]] = normal[0]
            vec[block[1]] = normal[1]
            vec[block[2]] = normal[2]
            projectors.append((atom, vec))
    return projectors


def normalize_projectors(projectors, ao_overlap):
    normed = []
    for atom, vec in projectors:
        n2 = float(vec @ ao_overlap @ vec)
        if n2 > 1e-14:
            normed.append((atom, vec / np.sqrt(n2)))
    return normed


def rank_pi_orbitals(C_mo, ao_overlap, projectors, top_n):
    # score[m] = sum_i |<MO_m|p_i>| where <MO|p> = C_mo[m] @ ao_overlap @ p
    if not projectors:
        return []
    proj_mat = np.stack([p for _, p in projectors], axis=1)  # (nbasis, nproj)
    overlaps = C_mo @ ao_overlap @ proj_mat                   # (nmo, nproj)
    scores = np.sum(np.abs(overlaps), axis=1)
    order = np.argsort(scores)[::-1]
    nout = min(top_n, len(order))
    return [(int(order[i]) + 1, float(scores[order[i]])) for i in range(nout)]


def get_molden_ao_overlap(filename):
    from orbkit import read, analytical_integrals
    qc = read.main_read(filename, itype="molden", all_mo=True)
    return analytical_integrals.get_ao_overlap(qc.geo_spec, qc.geo_spec, qc.ao_spec)


def main():
    parser = argparse.ArgumentParser(
        description="Rank molecular orbitals by pi character relative to a best-fit plane through selected atoms."
    )
    parser.add_argument("--target", required=True, help="Target Molden or HDF5 file.")
    parser.add_argument(
        "--atoms",
        required=True,
        nargs="+",
        action=ParseMixedListAction,
        metavar="ATOMS",
        help="1-based atom indices defining the planar aromatic subset (e.g. 1,3-8).",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=20,
        help="Number of orbitals to display (default: 20).",
    )
    parser.add_argument(
        "--planarity_threshold",
        type=float,
        default=0.10,
        help="Warn if max atom-to-plane distance exceeds this value (same coordinate units as input).",
    )
    args = parser.parse_args()

    if args.top_n <= 0:
        raise ValueError("--top_n must be a positive integer.")
    if not args.atoms:
        raise ValueError("--atoms must not be empty.")

    coords = extract_atom_coords(args.target)
    natoms = coords.shape[0]
    selected = sorted(set(args.atoms))
    if min(selected) < 1 or max(selected) > natoms:
        raise ValueError(f"Atom indices out of range: valid interval is [1, {natoms}].")

    sel_coords = coords[np.array(selected) - 1]
    _, normal, distances = best_fit_plane(sel_coords)

    print("Target file:", args.target)
    print("Selected atoms (1-based):", selected)
    print(f"Best-fit plane normal: [{normal[0]: .6f}, {normal[1]: .6f}, {normal[2]: .6f}]")
    print(f"Planarity check: max distance = {np.max(distances):.6f}, RMS distance = {np.sqrt(np.mean(distances**2)):.6f}")
    if np.max(distances) > args.planarity_threshold:
        print(
            "WARNING: selected atoms are not close to a single plane "
            f"(max distance {np.max(distances):.6f} > threshold {args.planarity_threshold:.6f})."
        )

    if is_h5_file(args.target):
        C_mo, S_ao, bf_ids = load_h5_data(args.target)
        p_blocks = h5_p_blocks_by_atom(bf_ids, natoms)
    else:
        C_mo = load_molden_mo_coeff_matrix(args.target)
        S_ao = get_molden_ao_overlap(args.target)
        p_blocks, _ = parse_molden_gto_for_p_blocks(args.target)

    projectors = build_pi_projectors(C_mo.shape[1], p_blocks, selected, normal)
    projectors = normalize_projectors(projectors, S_ao)
    if not projectors:
        raise ValueError(
            "No valid perpendicular p projectors were built for the selected atoms. "
            "Check atom selection and basis content."
        )

    used_atoms = sorted(set(atom for atom, _ in projectors))
    missing = [a for a in selected if a not in used_atoms]
    if missing:
        print(f"WARNING: no p-shells found for selected atoms: {missing}")

    ranked = rank_pi_orbitals(C_mo, S_ao, projectors, args.top_n)

    print("\nTop orbitals by pi character:")
    print("  MO   PiScore")
    for mo_idx, score in ranked:
        print(f"{mo_idx:4d}  {score:.6f}")


if __name__ == "__main__":
    main()
