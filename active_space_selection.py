#!/usr/bin/env python3
"""
Merged script for molecular orbital alignment workflow:
1. Compute optimal rotation between geometries (Kabsch algorithm)
2. Rotate reference MO coefficients
3. Calculate orbital overlaps with target
"""

import os
import numpy as np
import argparse
import re
from scipy.spatial.transform import Rotation as R
from sphecerix import tesseral_wigner_D
from orbkit import read, analytical_integrals

# ========================== Kabsch Algorithm Functions ========================

def compute_centroid(coords):
    """Calculate geometric center of coordinates"""
    return np.mean(coords, axis=0)

def kabsch_rotation(P, Q):
    """Compute optimal rotation matrix using Kabsch algorithm"""
    # Covariance matrix
    C = np.dot(P.T, Q)
    
    # Singular Value Decomposition
    V, _, Wt = np.linalg.svd(C)
    
    # Ensure proper rotation (handle reflection)
    if np.linalg.det(V) * np.linalg.det(Wt) < 0:
        V[:,-1] *= -1
    
    return V @ Wt


def compute_rmsd(P_rotated, Q):
    """
    Compute the Root Mean Square Deviation (RMSD) between two sets of coordinates.

    Parameters:
        P_rotated (np.ndarray): Nx3 matrix of the rotated initial geometry.
        Q (np.ndarray): Nx3 matrix of the final geometry (centered).
    
    Returns:
        float: The RMSD value.
    """
    diff = P_rotated - Q
    return np.sqrt(np.sum(diff**2) / len(P_rotated))

# ======================== Molden File Processing ==============================

def parse_molden_sections(filename):
    """
    Reads a Molden file and splits it into sections.
    
    Data Structure:
      Returns a list of tuples:
        [(section_header, section_text), ...]
      where section_header is a string (e.g. "[MO]", "[GTO]", etc.) and section_text is the full text of that section.
    
    Parameters:
      filename (str): Path to the Molden file.
      
    Returns:
      list of (str, str)
    """
    with open(filename, 'r') as f:
        content = f.read()
    # Split sections at lines beginning with "["
    sections = re.split(r'(?m)(?=^\[)', content)
    parsed = []
    for sec in sections:
        sec = sec.strip()
        if sec:
            header = sec.splitlines()[0].strip()
            parsed.append((header, sec))
    return parsed

def process_atom_block(lines):
    """
    Processes lines for one atom block in the [GTO] section.
    
    Data Structure:
      Returns a list of tuples. Each tuple has the form (orbital_label, n_funcs) where:
         - orbital_label (str): e.g., 's', 'p', 'd', etc.
         - n_funcs (int): expected number of functions (e.g., 1 for s, 3 for p, 5 for d, etc.)
    
    Parameters:
      lines (list of str): Lines belonging to an atom block.
      
    Returns:
      list of tuple (str, int)
    """
    shells = []
    for line in lines:
        parts = line.split()
        if parts and parts[0].lower() in ['s', 'p', 'd', 'f', 'g', 'h']:
            orb = parts[0].lower()
            if orb == 's':
                n_funcs = 1
            elif orb == 'p':
                n_funcs = 3
            elif orb == 'd':
                n_funcs = 5
            elif orb == 'f':
                n_funcs = 7
            elif orb == 'g':
                n_funcs = 9
            elif orb == 'h':
                n_funcs = 11
            else:
                n_funcs = 1
            shells.append((orb, n_funcs))
    return shells

def parse_gto_basis(filename):
    """
    Parses the [GTO] section of the Molden file to extract the basis set information.
    
    Data Structure:
      Returns a list of tuples, e.g.:
         [('s', 1), ('p', 3), ('d', 5), ...]
      The order corresponds to the ordering of shells in the file.
    
    Parameters:
      filename (str): Path to the Molden file.
      
    Returns:
      list of tuple (str, int)
    """
    with open(filename, 'r') as f:
        content = f.read()
    match = re.search(r'(?si)\[GTO\].*?(?=\n\[)', content)
    if not match:
        raise ValueError("GTO section not found in the file.")
    gto_text = match.group(0)
    lines = gto_text.splitlines()[1:]  # Skip the [GTO] header
    shells = []
    current_atom_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # A line with only a number indicates the start of a new atom block
        if re.match(r'^\d+$', line):
            if current_atom_lines:
                shells.extend(process_atom_block(current_atom_lines))
            current_atom_lines = []
        else:
            current_atom_lines.append(line)
    if current_atom_lines:
        shells.extend(process_atom_block(current_atom_lines))
    return shells

def parse_mo_block(mo_text):
    """
    Parses the [MO] section into individual MO blocks.
    
    Data Structure:
      Returns a list of dictionaries. Each dictionary represents one molecular orbital (MO) block with:
        'header' : list of str (typically the first four lines, e.g., "Sym=...", "Ene=...", etc.)
        'coeffs' : list of tuples (index, value) where both are strings.
        'raw'    : list of all lines for that MO block.
    
    Parameters:
      mo_text (str): Text of the [MO] section.
      
    Returns:
      list of dict
    """
    lines = mo_text.splitlines()
    blocks = []
    current_block = []
    for line in lines:
        if line.strip().upper() == "[MO]":
            continue
        if line.strip().upper().startswith("SYM="):
            if current_block:
                blocks.append(current_block)
            current_block = [line]
        else:
            current_block.append(line)
    if current_block:
        blocks.append(current_block)
    
    parsed_blocks = []
    for block in blocks:
        header = block[:4]  # The first 4 lines are taken as the header.
        coeff_lines = block[4:]
        coeffs = []
        for cl in coeff_lines:
            cl = cl.strip()
            if re.match(r'^\d+', cl):
                parts = cl.split()
                if len(parts) >= 2:
                    index = parts[0]
                    value = parts[1]
                    coeffs.append((index, value))
        parsed_blocks.append({'header': header, 'coeffs': coeffs, 'raw': block})
    return parsed_blocks

def format_mo_block(mo_block, new_coeffs):
    """
    Formats one MO block with the new (rotated) coefficients.
    
    Parameters:
      mo_block (dict): An MO block dictionary with keys 'header' and 'coeffs'.
      new_coeffs (list of float): List of new coefficients for the MO block.
      
    Returns:
      str: The formatted MO block as a string.
    """
    lines = []
    for h in mo_block['header']:
        lines.append(h)
    for i, coeff in enumerate(new_coeffs):
        lines.append(f"  {i+1:>3}   {coeff: .8e}")
    return "\n".join(lines)

def update_mo_section(mo_text, new_mo_coeff_blocks):
    """
    Rebuilds the [MO] section by replacing the original coefficients with the new ones.
    
    Parameters:
      mo_text (str): The original [MO] section text.
      new_mo_coeff_blocks (list of list of float): A list containing new coefficients for each MO block.
      
    Returns:
      str: The updated [MO] section as a string.
    """
    mo_blocks = parse_mo_block(mo_text)
    if len(mo_blocks) != len(new_mo_coeff_blocks):
        raise ValueError("Mismatch between number of MO blocks and new coefficient blocks.")
    updated_section = "[MO]\n"
    block_texts = []
    for block, new_coeffs in zip(mo_blocks, new_mo_coeff_blocks):
        block_texts.append(format_mo_block(block, new_coeffs))
    updated_section += "\n".join(block_texts)
    return updated_section

def write_new_molden_file(input_file, output_file, new_mo_coeff_blocks, new_coords=None):
    """
    Writes a new Molden file with updated [MO] section and optional coordinates.

    Parameters:
      input_file (str): Path to original Molden file.
      output_file (str): Path to save modified Molden file.
      new_mo_coeff_blocks (str): Updated [MO] section as a string.
      new_coords (list of tuples): Optional list of either:
          - (element, x, y, z)
          - or just (x, y, z), in which case element info is taken from the input file.
    """
    sections = parse_molden_sections(input_file)
    updated_content = ""

    # Store original atom info to reuse if only (x, y, z) are given
    original_atoms = []

    for header, sec_text in sections:
        if header.strip().upper().startswith("[MO]"):
            updated_sec = update_mo_section(sec_text, new_mo_coeff_blocks)
            updated_content += updated_sec + "\n"

        elif header.strip().upper().startswith("[ATOMS]"):
            updated_content += header + "\n"

            # Parse original atom lines to extract element, index, Z
            original_atoms = []
            for line in sec_text.strip().splitlines()[1:]:  # skip header
                tokens = line.strip().split()
                if len(tokens) >= 6:
                    element, idx, atomic_num = tokens[0], int(tokens[1]), int(tokens[2])
                    original_atoms.append((element, idx, atomic_num))

            if new_coords is not None:
                # If numpy array is passed, treat it as list of xyz
                if isinstance(new_coords, np.ndarray):
                    if new_coords.shape[1] != 3:
                        raise ValueError("new_coords numpy array must have shape (N, 3)")
                    new_coords = new_coords.tolist()

                for i, coord in enumerate(new_coords):
                    if len(coord) == 4:
                        # Full (element, x, y, z)
                        element, x, y, z = coord
                        idx = i + 1
                        atomic_num = next((atm[2] for atm in original_atoms if atm[0] == element), 0)
                    elif len(coord) == 3:
                        # Only (x, y, z), fetch atom info from original
                        x, y, z = coord
                        try:
                            element, idx, atomic_num = original_atoms[i]
                        except IndexError:
                            raise ValueError("More coordinates provided than atoms in the original Molden file.")
                    else:
                        raise ValueError("Each coordinate must be (x, y, z) or (element, x, y, z)")


                    updated_content += f"{element:4s} {idx:3d} {atomic_num:3d} {x:12.6f} {y:12.6f} {z:12.6f}\n"
            else:
                # Keep original atom section as-is
                updated_content += sec_text.strip().split('\n', 1)[1] + "\n"

        else:
            updated_content += sec_text + "\n"

    with open(output_file, 'w') as f:
        f.write(updated_content)


#def write_new_molden_file(input_file, output_file, new_mo_coeff_blocks):
#    """
#    Writes a new Molden file with the updated [MO] section.
#    
#    The file is divided into sections; only the [MO] section is replaced.
#    
#    Parameters:
#      input_file (str): Path to the original Molden file.
#      output_file (str): Path where the new Molden file will be written.
#      new_mo_coeff_blocks (list of list of float): The updated MO coefficients for each MO block.
#    """
#    sections = parse_molden_sections(input_file)
#    updated_content = ""
#    for header, sec_text in sections:
#        if header.strip().upper().startswith("[MO]"):
#            updated_sec = update_mo_section(sec_text, new_mo_coeff_blocks)
#            updated_content += updated_sec + "\n"
#        else:
#            updated_content += sec_text + "\n"
#    with open(output_file, 'w') as f:
#        f.write(updated_content)

# -----------------------------------------------------------------------------
# Function to Rotate MO Coefficients Shell-by-Shell
# -----------------------------------------------------------------------------

def get_permutation(l):
    if l == 0:
        molden_m = [1]
        sphecerix_m = [1]
    elif l == 1:
        molden_m = [1, -1, 0]
        sphecerix_m = [-1, 0, 1]
    elif l == 2:
        molden_m = [0, 1, -1, 2, -2]
        sphecerix_m = [-2, -1, 0, 1, 2]
    elif l == 3:
        molden_m = [0, 1, -1, 2, -2, 3, -3]
        sphecerix_m = [-3, -2, -1, 0, 1, 2, 3]
    elif l == 4:
        molden_m = [0, 1, -1, 2, -2, 3, -3, 4, -4]
        sphecerix_m = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    else:
        return [], []
    permutation = [molden_m.index(m) for m in sphecerix_m]
    inverse_perm = [0]*len(permutation)
    for i, p in enumerate(permutation):
        inverse_perm[p] = i
    return permutation, inverse_perm


def rotate_mo_coefficients_from_molden(mo_coeffs, shells, rotation_matrix):
    """
    Rotates the MO coefficients for one molecular orbital block.
    
    Parameters:
      mo_coeffs (list of float): The MO coefficients for one molecular orbital.
      shells (list of tuple): A list of shells obtained from the [GTO] section.
                              Each element is a tuple (orbital_label, n_funcs), e.g., ('s', 1), ('p', 3), etc.
    
    Process:
      For each shell:
        1. Determine the angular momentum l from the orbital label:
             's' -> l = 0, 'p' -> l = 1, 'd' -> l = 2, 'f' -> l = 3, etc.
        2. Extract the corresponding block of coefficients (n_funcs coefficients).
        3. Get the rotation matrix (here the identity matrix via D_matrix) of dimension (2*l+1).
        4. Multiply the block by the rotation matrix (which here leaves it unchanged).
    
    Returns:
      list of float: The new (rotated) coefficients for the molecular orbital.
    """
    new_coeffs = []
    idx = 0
    for orb, n_funcs in shells:
        # Determine angular momentum l based on orbital label
        if orb == 's':
            l = 0
        elif orb == 'p':
            l = 1
        elif orb == 'd':
            l = 2
        elif orb == 'f':
            l = 3
        elif orb == 'g':
            l = 4
        elif orb == 'h':
            l = 5
        else:
            l = 0
        
        # Check that n_funcs matches 2*l+1
        if n_funcs != 2 * l + 1:
            raise ValueError(f"For orbital '{orb}', expected {2*l+1} functions but got {n_funcs}.")
        
        # Extract coefficients for this shell
        block = mo_coeffs[idx: idx + n_funcs]
        if len(block) != n_funcs:
            raise ValueError(f"Expected {n_funcs} coefficients for orbital '{orb}', got {len(block)}.")
        block = np.array(block, dtype=float)

        Robj = R.from_matrix(rotation_matrix)
        D =  tesseral_wigner_D(l, Robj)

        # Permuta, ruota 
        perm, inv_perm = get_permutation(l)
       # print(l, block, perm)
        permuted = block[perm]
       # print(permuted)
        rotated_block = D @ permuted
        # riordina nell'ordine iniziale
        rotated_block = rotated_block[inv_perm]

        new_coeffs.extend(rotated_block.tolist())
        idx += n_funcs
    return new_coeffs

def create_rotated_molden(input_molden, output_molden, R_mat, tar_coords):

    # Parse the [GTO] section to extract basis shell information
    shells = parse_gto_basis(input_molden)
    total_basis = sum(n for orb, n in shells)
    print("Total number of basis functions (from [GTO] shells):", total_basis)
    
    # Parse the [MO] section from the input file
    sections = parse_molden_sections(input_molden)
    mo_text = None
    for header, sec in sections:
        if header.strip().upper().startswith("[MO]"):
            mo_text = sec
            break
    if mo_text is None:
        raise ValueError("MO section not found in the Molden file.")
    
    # Parse the MO section into MO blocks (each MO block is a dictionary)
    mo_blocks = parse_mo_block(mo_text)
    
    new_mo_coeff_blocks = []
    for block in mo_blocks:
        try:
            # Convert coefficient strings to floats.
            # Each block['coeffs'] is a list of tuples (index, value) as strings.
            coeffs = [float(val) for idx, val in block['coeffs']]
        except ValueError as e:
            raise ValueError(f"Error converting coefficient to float in block with header {block['header'][0]}: {e}")
        
        if len(coeffs) != total_basis:
            raise ValueError(f"Mismatch: MO block has {len(coeffs)} coefficients but expected {total_basis} based on [GTO] shells.")
        
        # Rotate the coefficients shell-by-shell using our simple D_matrix (identity rotation)
        rotated_coeffs = rotate_mo_coefficients_from_molden(coeffs, shells, R_mat)
        new_mo_coeff_blocks.append(rotated_coeffs)
    
    # Write the new Molden file with the updated [MO] section
    write_new_molden_file(input_molden, output_molden, new_mo_coeff_blocks, tar_coords)
    print(f"Rotated Molden file written to {output_molden}")

def extract_atom_coords(filename):
    """Extract atomic coordinates from [Atoms] section"""
    sections = parse_molden_sections(filename)
    for header, content in sections:
        if header.upper().startswith("[ATOMS]"):
            coords = []
            for line in content.splitlines()[1:]:  # Skip header
                parts = line.strip().split()
                if len(parts) >= 4:
                    _, x, y, z = parts[2:6]
                    coords.append([float(x), float(y), float(z)])
            return np.array(coords)
    raise ValueError(f"No [Atoms] section found in {filename}")


# ======================== Overlap Calculation =================================

def compute_orbital_overlaps(rotated_file, target_file, ref_orbitals, active_orbitals):
    """Calculate and print orbital overlaps"""
    qc_rot = read.main_read(rotated_file, itype='molden', all_mo=True)
    qc_tar = read.main_read(target_file, itype='molden', all_mo=True)
    
    # Compute AO overlap matrix
    ao_overlap = analytical_integrals.get_ao_overlap(
        qc_rot.geo_spec, qc_tar.geo_spec, qc_rot.ao_spec
    )
    
    # Compute MO overlap matrix
    mo_overlap = analytical_integrals.get_mo_overlap_matrix(
        qc_rot.mo_spec, qc_tar.mo_spec, ao_overlap
    )

    target_orbitals = []
    second_best_target_orbitals = []
    warnings = []
    
    # Print mapping for specified orbitals
    print("Orbital Mapping Results:")
    print("Reference -> Target (Overlap)")
    for orb in ref_orbitals:
        idx = orb - 1  # Convert to 0-based index
        # best_match = np.argmax(np.abs(mo_overlap[idx]))
        best_matches  = np.argsort(np.abs(mo_overlap[idx]))[::-1][:len(active_orbitals)]
        for j, match in enumerate(best_matches):
            if match +1 in target_orbitals:
                 continue
            else:
                 best_match = match
                 second_best_match = best_matches[j+1]
                 target_orbitals.append(match+1)
                 break

             
             
        # target_orbitals.append(best_match + 1)
        overlap_value = mo_overlap[idx, best_match]
        if abs(overlap_value) < 0.8 or abs(overlap_value) > 1.2 :
             warnings.append([orb, best_match+1, overlap_value])
        try:
                print(f"  {orb:3d}    -> {best_match+1:3d}    ({overlap_value:.4f}) [second best match: {second_best_match+1:3d} ({mo_overlap[idx,second_best_match]:.4f})] [original MO overlap: ({mo_overlap[idx,idx]:.4f}) {list(best_matches).index(idx)+1:3d}]")
        except:
             print(f"  {orb:3d}    -> {best_match+1:3d}    ({overlap_value:.4f}) [second best match: {second_best_match+1:3d} ({mo_overlap[idx,second_best_match]:.4f})] [original MO overlap: ({mo_overlap[idx,idx]:.4f}) not in list]")

    target_not_in_active_space = []
    active_space_not_in_target = []
    for targetMO in target_orbitals:
       if targetMO not in active_orbitals:
           target_not_in_active_space.append(targetMO)
           
    for activeMO in active_orbitals:
       if activeMO not in target_orbitals:
           active_space_not_in_target.append(activeMO)
       
    # Save alter file in the same dir as the target molden
    target_dir = os.path.dirname(os.path.abspath(target_file))
    alter_path = os.path.join(target_dir, 'ALTER.txt')
    with open(alter_path, 'w') as alterfile:
       line = 'ALTER = '+str(len(active_space_not_in_target))+'; '
       for i in range(len(active_space_not_in_target)):
            line = line + '1 '+str(target_not_in_active_space[i])+' '+str(active_space_not_in_target[i])+'; '
       line = line +' * Generated automatically\n'
       if warnings != []:
            line  = line + '* WARNING some orbitals do not match. Check manually\n'
            for w in warnings:
                line  = line + '* REF '+str(w[0])+'--> TARGET '+str(w[1])+'  ('+str(w[2])+')\n'
            print(line)
       alterfile.write(line)
     
#===========================PARSING================================

def parse_range_or_value(item_str):
    """
    Converte "7" in [7] o "3-5" in [3,4,5], ignorando eventuali spazi.
    """
    item_str = item_str.strip()
    if '-' in item_str:
        try:
            start, end = map(int, item_str.split('-'))
            return list(range(start, end + 1))
        except ValueError:
            raise argparse.ArgumentTypeError(f"Intervallo non valido: '{item_str}'")
    else:
        try:
            return [int(item_str)]
        except ValueError:
            raise argparse.ArgumentTypeError(f"Valore non valido: '{item_str}'")

def parse_mixed_list(input_str):
    """
    Converte "1,3-5,7" in [1,3,4,5,7], ignorando spazi e parentesi.
    """
    s = input_str.strip().strip('[]')
    if not s:
        return []
    items = re.split(r'\s*,\s*', s)
    result = []
    for it in items:
        result.extend(parse_range_or_value(it))
    return result

def parse_list_of_mo_lists(input_str):
    """
    Converte "1-3:5,7-8" in [[1,2,3], [5,7,8]],
    usando ':' per separare le sottoliste e ignorando gli spazi.
    """
    s = input_str.strip().strip('[]')
    if not s:
        return []
    groups = re.split(r'\s*:\s*', s)
    result = []
    for g in groups:
        items = re.split(r'\s*,\s*', g)
        sub = []
        for it in items:
            sub.extend(parse_range_or_value(it))
        result.append(sub)
    return result

# --- custom actions ------------------------

class ParseMixedListAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        # values è una lista di token, es ['4,6-9']
        joined = ' '.join(values)
        parsed = parse_mixed_list(joined)
        setattr(namespace, self.dest, parsed)

class ParseMoListListAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        joined = ' '.join(values)
        parsed = parse_list_of_mo_lists(joined)
        setattr(namespace, self.dest, parsed)

# --- main() ------------------------
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Molecular Orbital Alignment Tool\n\n"
            "This script takes one or more reference Molden files and a single target Molden file, "
            "computes the optimal rotational alignment (using a subset of atoms if requested), rotates "
            "the MO coefficients of the best-matching reference, and then calculates orbital overlaps "
            "between the rotated reference and the target. Output files (rotate.csv, rotated.molden, ALTER.txt) "
            "are saved in the same directory as the target file and a summary is printed to standard output."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Example Usage:\n"
            "  python active_space_selection.py \\\n"
            "    --ref ref1.molden ref2.molden \\\n"
            "    --ref_orbitals 1-5: 7, 2-6 :8,10 \\\n"
            "    --target target.molden \\\n"
            "    --active_space 3,4-6,9 \\\n"
            "    --atoms 1,3-5,8\n\n"
            "Notes on Formats:\n"
            "  --ref               Provide one or more reference Molden files (separated by spaces).\n"
            "  --target            Provide exactly one target Molden file.\n"
            "  --ref_orbitals      For each reference file, specify a list of orbitals (1-based indices):\n"
            "                       Use commas to separate individual orbitals, hyphens for ranges, and\n"
            "                       colons to separate different sublists when supplying multiple groups.\n"
            "                       Example: 1-3:5,7-8 means orbitals [1,2,3] and [5,7,8].\n"
            "  --active_space      Specify active orbitals (1-based indices) for the system. Use\n"
            "                       commas for individual orbitals and hyphens for ranges. Example: 4,6-9.\n"
            "  --atoms             (Optional) Specify which atom indices (1-based) to include in the alignment\n"
            "                       algorithm. Supports commas and ranges (e.g., 1,3-5,8). If omitted,\n"
            "                       all atoms in each file are used.\n"
        )
    )

    parser.add_argument(
        '--ref',
        required=True,
        nargs='+',
        metavar='REF_MOLDEN',
        help=(
            "List of one or more reference Molden files.\n"
            "Example: --ref ref1.molden ref2.molden ref3.molden"
        )
    )

    parser.add_argument(
        '--target',
        required=True,
        metavar='TARGET_MOLDEN',
        help=(
            "Single target Molden file (input).\n"
            "All output files (rotate.csv, rotated.molden, ALTER.txt) will be saved in the same directory\n"
            "where this target file resides."
        )
    )

    parser.add_argument(
        '--ref_orbitals',
        required=True,
        nargs='+',
        action=ParseMoListListAction,
        metavar='REF_ORBS',
        help=(
            "Orbital indices for each reference file, using 1-based numbering.\n"
            "Use commas to separate orbitals, hyphens for ranges, and colons to separate sublists:\n"
            "Example: 1-3:5,7-8 means orbitals [1,2,3] and [5,7,8].\n"
            "Provide exactly as many lists here as there are files in --ref, in the same order.\n"
            "Example: --ref ref1.molden ref2.molden --ref_orbitals 1-5,7: 2-6, 9 "
        )
    )

    parser.add_argument(
        '--active_space',
        required=True,
        nargs='+',
        action=ParseMixedListAction,
        metavar='ACTIVE',
        help=(
            "List of active orbitals (1-based indices) for the system.\n"
            "Use commas to separate single orbitals and hyphens for ranges.\n"
            "Example: --active_space 4,6-9 means orbitals [4,6,7,8,9]."
        )
    )

    parser.add_argument(
        '--atoms',
        nargs='+',
        required=False,
        action=ParseMixedListAction,
        default=[],
        metavar='ATOMS',
        help=(
            "(Optional) Specify which atoms (1-based indices) to include in the Kabsch alignment.\n"
            "Use commas and hyphens for ranges. Example: --atoms 1,3-5,8.\n"
            "If omitted, all atoms in each file are used for the alignment."
        )
    )

    args = parser.parse_args()

    target_dir = os.path.dirname(os.path.abspath(args.target))
    if not os.path.isdir(target_dir):
        # Se viene passato solo il nome del file (senza path), uso la cartella corrente
        target_dir = os.getcwd()



    if len(args.ref) != len(args.ref_orbitals):
        raise ValueError("Il numero di file in --ref deve corrispondere al numero di liste in --ref_orbitals")

    print("Reference molden files:", args.ref)
    print("Reference MOs:")
    for i, (ref_file, orbitals) in enumerate(zip(args.ref, args.ref_orbitals)):
        print(f"  {ref_file}: {orbitals}")

    print("Target molden file:", args.target)
    print("Active space:", args.active_space)

    if args.atoms:
        print("Atoms selected for alignment:", args.atoms)
        atom_list = np.array(args.atoms) -1
    else:
        print("All atoms used for alignment.")


    
    # Step 1: Compute optimal rotation among the references:
    print("\nEvaluating file: "+args.target )
    print("Calculating optimal rotation..." )

    rmsd_list = []
    rotation_list = []
    for ref in args.ref:

        ref_coords = extract_atom_coords(ref)
        # print('Ref coords:', ref_coords)
        tar_coords = extract_atom_coords(args.target)
        # print('Tar coords:', tar_coords)

        if args.atoms:
            ref_coords_trimmed = ref_coords[atom_list, :]
            tar_coords_trimmed = tar_coords[atom_list, :]
        else:
            ref_coords_trimmed = ref_coords
            tar_coords_trimmed = tar_coords

        # print('Ref coords trimmed:', ref_coords_trimmed)
        # print('Tar coords trimmed:', tar_coords_trimmed)
        
        # Center coordinates
        ref_centroid = compute_centroid(ref_coords_trimmed)
        tar_centroid = compute_centroid(tar_coords_trimmed)
        P = ref_coords_trimmed - ref_centroid
        Q = tar_coords_trimmed - tar_centroid
        
        # Compute rotation matrix
        rotation = kabsch_rotation(P, Q)
        rotation_list.append(rotation)
        # Rotate the initial geometry using the rotation matrix U
        P_rotated = np.dot(P, rotation)
        # Compute the RMSD between the rotated initial geometry and the final geometry (both centered)
        rmsd_value = compute_rmsd(P_rotated, Q)
        rmsd_list.append(rmsd_value)

    best_ref_index = rmsd_list.index(min(rmsd_list))
    best_ref = args.ref[best_ref_index]
    best_ref_orbitals = args.ref_orbitals[best_ref_index]
    best_rmsd = rmsd_list[best_ref_index]
    rotation = rotation_list[best_ref_index]

    rotate_path = os.path.join(target_dir, 'rotate.csv')
    np.savetxt(rotate_path, rotation, delimiter=',')
    print("Rotation matrix saved to rotate.csv --> RMSD: "+str(best_rmsd)+' A.U. ('+best_ref+')')
    
    # Step 2: Create rotated Molden file
    # print("Generating rotated Molden file...")
    rotated_path = os.path.join(target_dir, 'rotated.molden')
    create_rotated_molden(best_ref, rotated_path, rotation.T, tar_coords)
    # print("Rotated file saved to rotated.molden")
    
    # Step 3: Compute orbital overlaps
    print("Computing orbital overlaps...")
    compute_orbital_overlaps(rotated_path, args.target, best_ref_orbitals, args.active_space)

if __name__ == "__main__":
    main()
