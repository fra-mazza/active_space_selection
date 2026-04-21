#!/usr/bin/env python3
"""
Identify pi-like molecular orbitals for a (quasi-)planar atom subset.

Workflow:
1) Fit best plane through selected atoms.
2) Build per-atom p orbital(s) perpendicular to that plane in AO basis,
   weighting each contracted p-shell by Σ c_i² · α_i so that inner
   (compact) shells contribute more than diffuse outer shells.
3) Compute |<MO|p_perp>| and accumulate over all selected-atom p projectors.
4) Rank MOs by accumulated pi-character and print top N.
5) Optionally build an OpenMolcas ALTER block from top-ranked pi orbitals.
"""

import argparse
import os
import re
import numpy as np

try:
    import h5py
    _H5PY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _H5PY_AVAILABLE = False


# Number of real AO components per shell (2*l+1): s=1, p=3, d=5, ...
L_TO_NFUNCS = {"s": 1, "p": 3, "d": 5, "f": 7, "g": 9, "h": 11}
# Numerical guard for near-zero projector norms in AO-overlap metric.
PROJECTOR_NORM_EPS = 1e-14
# Heuristic threshold requested by user for "high pi-score" orbitals.
PI_SCORE_HIGH_THRESHOLD = 1.0


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
    """
    Parse the [GTO] section of a Molden file to extract contracted p-shell data.

    For each atom, the function records:
      - The AO indices (x, y, z) for every contracted p-shell.
      - A compactness weight for each contraction, defined as
        ``Σ_i c_i² · α_i`` (sum of squared contraction coefficient times
        exponent over all primitives in the shell).  This is larger for tight
        (inner) shells than for diffuse (outer) shells, and is used as an
        unnormalized weight when building the pi projector.

    Returns
    -------
    p_blocks_by_atom : dict  {atom_1based: [[ao_x, ao_y, ao_z], ...]}
        AO indices for each contracted p-shell, in order of appearance.
    p_shell_weights : dict  {atom_1based: [w_shell1, ...]}
        Compactness weight for each p-shell (same ordering as above).
    ao_idx : int
        Total number of AO basis functions encountered.
    """
    with open(filename, "r") as f:
        content = f.read()
    match = re.search(r"(?si)\[GTO\].*?(?=\n\[|$)", content)
    if not match:
        raise ValueError("GTO section not found in Molden file.")
    lines = match.group(0).splitlines()[1:]

    p_blocks_by_atom = {}
    p_shell_weights = {}
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
                # Canonical real-p component ordering used by this script/repository:
                # [component_x, component_y, component_z].
                p_blocks_by_atom.setdefault(current_atom, []).append([ao_idx, ao_idx + 1, ao_idx + 2])
                # Compute the compactness weight Σ c_i² · α_i for this shell.
                # This equals the kinetic-energy-like contribution of the contraction
                # and is systematically larger for compact (inner) shells than for
                # diffuse (outer) shells, even when the same primitives appear in
                # multiple contractions (general contraction scheme).
                weight = 0.0
                for k in range(nprim):
                    parts = lines[i + k].strip().split()
                    if len(parts) >= 2:
                        try:
                            alpha = abs(float(parts[0]))
                            coeff = float(parts[1])
                            weight += coeff ** 2 * alpha
                        except ValueError:
                            pass
                # Fallback: if parsing yields zero, assign a positive placeholder.
                p_shell_weights.setdefault(current_atom, []).append(weight if weight > 0.0 else 1.0)
            ao_idx += nfunc
            i += nprim
            continue

    return p_blocks_by_atom, p_shell_weights, ao_idx


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


def h5_p_blocks_by_atom(bf_ids, natoms, filename=None):
    """
    Build a per-atom list of contracted p-shell AO index groups.

    Parameters
    ----------
    bf_ids : ndarray, shape (n_basis, 4)
        ``BASIS_FUNCTION_IDS`` array from the OpenMolcas HDF5 file.
        Columns: (center, shell_index, l, m).
    natoms : int
        Total number of atoms (used to infer the center-index base).
    filename : str or None
        Path to the same HDF5 file.  When provided, the ``PRIMITIVES`` and
        ``PRIMITIVE_IDS`` datasets are read to compute a compactness weight
        ``Σ_i c_i² · α_i`` for each contracted p-shell (identical to the
        metric used in the Molden code path).  This is systematically larger
        for tight (inner) contractions than for diffuse ones, even in general
        contraction schemes where multiple shells share the same primitives.
        When *None*, a rank-based fallback is used: the first p-shell for each
        atom receives weight 1, the second 1/2, the third 1/3, etc.

    Returns
    -------
    p_blocks : dict  {atom_1based: [[ao_x, ao_y, ao_z], ...]}
        AO index triples for each contracted p-shell.
    p_shell_weights : dict  {atom_1based: [weight, ...]}
        Compactness weight for each p-shell, in the same order as *p_blocks*.
    """
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

    # Build compactness-weight lookup from PRIMITIVE_IDS / PRIMITIVES (if available).
    # PRIMITIVE_IDS columns: (center, l, contracted_shell_1based).
    # PRIMITIVES columns:    (exponent, contraction_coefficient).
    # Weight = Σ_i c_i² · α_i — identical to the metric used in the Molden path.
    # This is larger for compact (inner) contractions and correctly handles general
    # contraction schemes where multiple contracted shells share the same primitives.
    weight_lookup = {}  # key: (center_in_prim, shell_in_prim) for l==1
    if filename is not None and _H5PY_AVAILABLE:
        try:
            with h5py.File(filename, "r") as f:
                if "PRIMITIVE_IDS" in f and "PRIMITIVES" in f:
                    prim_ids = np.array(f["PRIMITIVE_IDS"], dtype=int)
                    primitives = np.array(f["PRIMITIVES"], dtype=float)
            for i in range(len(prim_ids)):
                c_p, l_p, s_p = int(prim_ids[i, 0]), int(prim_ids[i, 1]), int(prim_ids[i, 2])
                if l_p == 1:
                    key = (c_p, s_p)
                    alpha = abs(float(primitives[i, 0]))
                    coeff = float(primitives[i, 1])
                    weight_lookup[key] = weight_lookup.get(key, 0.0) + coeff ** 2 * alpha
        except Exception:
            # If reading fails for any reason, fall back to rank-based weights.
            weight_lookup = {}

    p_shell_weights = {}
    for (atom_1based, shell), items in groups.items():
        m_to_idx = {m: idx for idx, m in items}
        if all(m in m_to_idx for m in (1, -1, 0)):
            # Reconstruct the same real-p component order used in the Molden code path.
            p_blocks.setdefault(atom_1based, []).append([m_to_idx[1], m_to_idx[-1], m_to_idx[0]])
            if weight_lookup:
                # Convert atom_1based back to the center index used in PRIMITIVE_IDS.
                # PRIMITIVE_IDS uses the same raw center stored in BASIS_FUNCTION_IDS
                # (before the 1-base correction), so we reverse the correction here.
                c_prim = atom_1based - (1 - base)
                w = weight_lookup.get((c_prim, shell), 0.0)
                # Fallback: if the lookup yielded zero (e.g. pure-zero contraction),
                # assign a small positive placeholder so the shell is not silently dropped.
                w = w if w > 0.0 else 1e-6
            else:
                # Rank-based fallback: rank 0 (first p-shell = most compact) → 1.0,
                # rank 1 → 0.5, rank 2 → 0.333, ...
                rank = len(p_blocks.get(atom_1based, [])) - 1
                w = 1.0 / (rank + 1)
            p_shell_weights.setdefault(atom_1based, []).append(w)

    return p_blocks, p_shell_weights


def best_fit_plane(coords):
    centroid = np.mean(coords, axis=0)
    centered = coords - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    normal = normal / np.linalg.norm(normal)
    distances = np.abs(centered @ normal)
    return centroid, normal, distances


def build_pi_projectors(nbasis, p_blocks_by_atom, selected_atoms, normal, shell_weights=None):
    """
    Build one perpendicular-p projector vector per selected atom.

    For each atom the projector is a weighted sum of all contracted p-shells:

        p_⊥ = Σ_shells  w_shell · (n_x · p_x + n_y · p_y + n_z · p_z)

    where (n_x, n_y, n_z) is the plane normal and w_shell is the shell weight.

    Parameters
    ----------
    nbasis : int
        Total number of AO basis functions.
    p_blocks_by_atom : dict  {atom_1based: [[ao_x, ao_y, ao_z], ...]}
        AO index triples for each contracted p-shell (from the Molden/HDF5
        parsing functions).
    selected_atoms : list of int
        1-based atom indices to include.
    normal : ndarray, shape (3,)
        Unit normal to the best-fit molecular plane.
    shell_weights : dict  {atom_1based: [w0, w1, ...]} or None
        Unnormalized weight for each contracted p-shell of each atom.  A larger
        weight makes that shell's contribution dominate the projector.  The
        natural choice is ``Σ_i c_i² · α_i`` (sum of squared contraction
        coefficient times exponent), which is systematically larger for
        compact (inner) shells than for diffuse (outer) shells.  When *None*
        all shells are treated equally (original behaviour).

    Returns
    -------
    projectors : list of (atom_1based, ndarray)
        Raw (un-normalized) projector vectors in the AO basis.
    """
    projectors = []
    for atom in selected_atoms:
        atom_blocks = p_blocks_by_atom.get(atom, [])
        if not atom_blocks:
            continue
        vec = np.zeros(nbasis, dtype=float)

        if shell_weights and atom in shell_weights:
            weights = shell_weights[atom]
        else:
            # Equal weighting: every contracted p-shell contributes the same amount.
            weights = [1.0] * len(atom_blocks)

        for block, w in zip(atom_blocks, weights):
            # Scale the normal-direction p-components of this shell by its weight.
            vec[block] = np.asarray(normal) * w

        projectors.append((atom, vec))
    return projectors


def normalize_projectors(projectors, ao_overlap):
    normed = []
    for atom, vec in projectors:
        n2 = float(vec @ ao_overlap @ vec)
        if n2 > PROJECTOR_NORM_EPS:
            normed.append((atom, vec / np.sqrt(n2)))
    return normed


def write_projectors_molden(target_molden, output_molden, projectors):
    sections = parse_molden_sections(target_molden)
    mo_lines = ["[MO]"]
    for atom, vec in projectors:
        mo_lines.extend(
            [
                f" Sym= P_ATOM_{atom}",
                " Ene= 0.00000000",
                " Spin= Alpha",
                " Occup= 0.00000000",
            ]
        )
        for i, coeff in enumerate(vec, start=1):
            mo_lines.append(f" {i:5d} {coeff: .12e}")
    mo_section = "\n".join(mo_lines)

    with open(output_molden, "w") as f:
        for header, sec_text in sections:
            if header.strip().upper().startswith("[MO]"):
                f.write(mo_section + "\n")
            else:
                f.write(sec_text + "\n")


def write_projectors_h5(target_h5, output_h5, projectors):
    if not _H5PY_AVAILABLE:
        raise ImportError("h5py is required to write HDF5 files: pip install h5py")
    projector_matrix = np.stack([vec for _, vec in projectors], axis=0)
    projector_atoms = np.array([atom for atom, _ in projectors], dtype=int)

    with h5py.File(target_h5, "r") as src, h5py.File(output_h5, "w") as dst:
        for attr_name, attr_val in src.attrs.items():
            dst.attrs[attr_name] = attr_val
        for key in src:
            if key == "MO_VECTORS":
                dst.create_dataset("MO_VECTORS", data=projector_matrix)
            else:
                src.copy(key, dst)
        dst.create_dataset("PI_PROJECTOR_ATOMS", data=projector_atoms)
        dst.attrs["N_PI_PROJECTORS"] = int(projector_matrix.shape[0])


def save_projector_orbitals(target_file, projectors):
    out_dir = os.path.dirname(os.path.abspath(target_file))
    if is_h5_file(target_file):
        out_file = os.path.join(out_dir, "pi_projectors.h5")
        write_projectors_h5(target_file, out_file, projectors)
    else:
        out_file = os.path.join(out_dir, "pi_projectors.molden")
        write_projectors_molden(target_file, out_file, projectors)
    return out_file


def compute_pi_scores(C_mo, ao_overlap, projectors):
    """
    Compute the pi-score for every MO.

    score[k] = sum_i |<MO_k|p_i>|, where p_i is the i-th normalized projector
    and <MO|p> = C_mo[k] @ ao_overlap @ p.
    """
    if not projectors:
        return np.array([], dtype=float)
    proj_mat = np.stack([p for _, p in projectors], axis=1)  # (nbasis, nproj)
    overlaps = C_mo @ ao_overlap @ proj_mat                   # (nmo, nproj)
    return np.sum(np.abs(overlaps), axis=1)


def rank_pi_orbitals(scores, top_n):
    """Return top-N (1-based mo_index, score) pairs sorted by descending score."""
    if len(scores) == 0:
        return []
    order = np.argsort(scores)[::-1]
    nout = min(top_n, len(order))
    ranked = []
    for i in range(nout):
        mo_zero_based = order[i]
        ranked.append((int(mo_zero_based) + 1, float(scores[mo_zero_based])))
    return ranked


def resolve_alter_path(target_file, alter_arg):
    """
    Resolve ALTER output path using the same conventions as active_space_selection.py.

    - If --alter is omitted, write ALTER.txt next to --target.
    - If --alter is a bare filename, write it next to --target.
    - If --alter contains a directory, use it as provided.
    """
    target_dir = os.path.dirname(os.path.abspath(target_file))

    if alter_arg:
        if os.path.dirname(alter_arg):
            return alter_arg
        return os.path.join(target_dir, alter_arg)

    return os.path.join(target_dir, "ALTER.txt")


def write_alter_from_pi_selection(selected_pi_orbitals, active_space, alter_path):
    """
    Build and write an OpenMolcas ALTER block from selected pi-like orbitals.

    This mirrors the target/active-space swap logic used in active_space_selection.py:
    every orbital in selected_pi_orbitals that is not in active_space is swapped with
    the corresponding orbital in active_space that is not in selected_pi_orbitals.
    """
    target_not_in_active = [t for t in selected_pi_orbitals if t not in active_space]
    active_not_in_target = [a for a in active_space if a not in selected_pi_orbitals]
    if len(target_not_in_active) != len(active_not_in_target):
        raise ValueError(
            "Cannot build ALTER block: selected pi orbitals and active space produce "
            "a different number of swap candidates."
        )

    with open(alter_path, "w") as alterfile:
        swaps = "".join(
            f"1 {target_orb} {active_orb}; "
            for target_orb, active_orb in zip(target_not_in_active, active_not_in_target)
        )
        line = f"ALTER = {len(active_not_in_target)}; {swaps} * Generated automatically\n"
        alterfile.write(line)


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
    parser.add_argument(
        "--active_space",
        required=False,
        nargs="+",
        action=ParseMixedListAction,
        default=[],
        metavar="ACTIVE",
        help=(
            "(Optional) Active-space orbital indices (1-based) used to auto-build ALTER.\n"
            "If provided, the script selects the top N orbitals by PiScore where N is\n"
            "the number of active orbitals."
        ),
    )
    parser.add_argument(
        "--alter",
        required=False,
        default=None,
        metavar="ALTER_FILE",
        help=(
            "(Optional) Name/path for generated ALTER file when --active_space is used.\n"
            "Defaults to ALTER.txt in the target-file directory."
        ),
    )
    args = parser.parse_args()

    if args.top_n <= 0:
        raise ValueError("--top_n must be a positive integer.")
    if not args.atoms:
        raise ValueError("--atoms must not be empty.")
    if args.alter and not args.active_space:
        raise ValueError("--alter requires --active_space.")

    # --- 1) Read geometry and verify selected atoms ---
    coords = extract_atom_coords(args.target)
    natoms = coords.shape[0]
    selected = sorted(set(args.atoms))
    if min(selected) < 1 or max(selected) > natoms:
        raise ValueError(f"Atom indices out of range: valid interval is [1, {natoms}].")

    # --- 2) Build best-fit plane from selected atoms and print diagnostics ---
    sel_coords = coords[np.array(selected) - 1]
    _, normal, distances = best_fit_plane(sel_coords)

    print("Target file:", args.target)
    print("Selected atoms (1-based):", selected)
    print(f"Best-fit plane normal: [{normal[0]:+.6f}, {normal[1]:+.6f}, {normal[2]:+.6f}]")
    print(f"Planarity check: max distance = {np.max(distances):.6f}, RMS distance = {np.sqrt(np.mean(distances**2)):.6f}")
    if np.max(distances) > args.planarity_threshold:
        print(
            "WARNING: selected atoms are not close to a single plane "
            f"(max distance {np.max(distances):.6f} > threshold {args.planarity_threshold:.6f})."
        )

    # --- 3) Load MO coefficients / AO overlap and identify p-shell blocks ---
    if is_h5_file(args.target):
        C_mo, S_ao, bf_ids = load_h5_data(args.target)
        # Pass the filename so that PRIMITIVE_IDS / PRIMITIVES are read to
        # compute Σ c²·α compactness weights.  Falls back to rank-based
        # weights automatically if those datasets are absent.
        p_blocks, p_shell_exps = h5_p_blocks_by_atom(bf_ids, natoms, filename=args.target)
    else:
        C_mo = load_molden_mo_coeff_matrix(args.target)
        S_ao = get_molden_ao_overlap(args.target)
        # parse_molden_gto_for_p_blocks also returns Σ c²·α compactness
        # weights for each contracted p-shell.
        p_blocks, p_shell_exps, _ = parse_molden_gto_for_p_blocks(args.target)

    # --- 4) Build and normalize per-atom perpendicular p projectors ---
    # shell_weights = Σ c²·α compactness weights so that inner (compact)
    # p-shells contribute more to the projector than diffuse outer shells.
    projectors = build_pi_projectors(C_mo.shape[1], p_blocks, selected, normal, shell_weights=p_shell_exps)
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

    projector_file = save_projector_orbitals(args.target, projectors)
    print(f"Saved pi projectors to: {projector_file}")

    # --- 5) Score all MOs and print ranked list ---
    scores = compute_pi_scores(C_mo, S_ao, projectors)
    ranked = rank_pi_orbitals(scores, args.top_n)

    print("\nTop orbitals by pi character:")
    print("  MO   PiScore")
    for mo_idx, score in ranked:
        print(f"{mo_idx:4d}  {score:.6f}")

    # If an active space was provided, automatically create an ALTER file using
    # the top-N pi orbitals, where N is the active-space size.
    if args.active_space:
        if min(args.active_space) < 1 or max(args.active_space) > C_mo.shape[0]:
            raise ValueError(
                f"Active-space orbital indices out of range: valid interval is [1, {C_mo.shape[0]}]."
            )
        n_active = len(args.active_space)
        top_for_active = rank_pi_orbitals(scores, n_active)
        selected_pi_orbitals = [mo_idx for mo_idx, _ in top_for_active]
        alter_path = resolve_alter_path(args.target, args.alter)
        write_alter_from_pi_selection(selected_pi_orbitals, args.active_space, alter_path)

        print(f"\nActive-space size: {n_active}")
        print("Top pi orbitals used for ALTER:", selected_pi_orbitals)
        print(f"ALTER file written to: {alter_path}")

        high_score_count = np.sum(scores > PI_SCORE_HIGH_THRESHOLD)
        if high_score_count > n_active:
            print(
                "WARNING: More than N orbitals have high PiScore "
                f"(>{PI_SCORE_HIGH_THRESHOLD:.1f}): {high_score_count} orbitals vs N={n_active}. "
                "Important pi orbitals may be excluded; consider increasing the active space."
            )


if __name__ == "__main__":
    main()
