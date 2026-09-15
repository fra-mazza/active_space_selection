"""
Full test suite for active_space_selection.py, get_pi_orbitals.py, and
combine_alter_files.py.

Everything here runs against the single real QM calculation checked into
test/phenol_scf/ (a closed-shell SCF wavefunction for phenol, s/p/d basis) --
no external QM software is required to run this suite. It is intended as the
one command a user runs right after installing the package to confirm the
install works end to end:

    python -m pytest test/test_suite.py -v

Coverage:
  - Molden and HDF5 workflows for active_space_selection.py, and that they
    agree with each other (orbital mapping, ALTER.txt, rotation matrices)
  - Unit tests for the HDF5 rotation-block / Wigner-D machinery
  - The --act_elect/--act_orb automatic active-space feature
  - get_pi_orbitals.py, including a genuine correctness cross-check: its
    purely-geometric pi-character ranking is compared against the phenol
    active space (test/ref_orbitals.txt / test/active_orbitals.txt), which
    was chosen independently of that script
  - combine_alter_files.py, including a round-trip check: splitting the
    active space into two disjoint pieces, running the main script on each,
    and combining the results must match a single full-active-space run
  - CLI / argument-validation edge cases across all three scripts
  - Determinism: repeated runs on identical inputs give byte-identical
    output files

This single closed-shell SCF/s-p-d test case cannot exercise everything --
genuine CASSCF active spaces with fractional natural-orbital occupations,
f/g basis functions, multi-reference RMSD selection, and --atoms-subset
alignment all need purpose-built QM calculations that are not yet part of
the repository (see the project's test-suite plan).
"""

import filecmp
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import pytest

# Ensure the repo root is on the path so we can import the modules directly.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

TEST_DIR = os.path.join(REPO_ROOT, "test", "phenol_scf")
REF_MOLDEN = os.path.join(TEST_DIR, "ref.molden")
TAR_MOLDEN = os.path.join(TEST_DIR, "target.molden")
REF_H5 = os.path.join(TEST_DIR, "ref.h5")
TAR_H5 = os.path.join(TEST_DIR, "target.h5")

REF_ORBS_FILE = os.path.join(REPO_ROOT, "test", "ref_orbitals.txt")
ACTIVE_FILE = os.path.join(REPO_ROOT, "test", "active_orbitals.txt")

with open(REF_ORBS_FILE) as fh:
    REF_ORBS_STR = fh.read().strip()           # "19,23-27,34"
with open(ACTIVE_FILE) as fh:
    ACTIVE_STR = fh.read().strip()              # "22-28"

# Known-good active space for phenol, as a plain int set. This is the
# *independently chosen* set of CASSCF active orbitals for this test
# molecule (picked by whoever set up test/phenol_scf, not derived from
# get_pi_orbitals.py), which is what makes comparing get_pi_orbitals.py's
# purely-geometric ranking against it a meaningful check rather than a
# tautology.
PHENOL_ACTIVE_SET = {19, 23, 24, 25, 26, 27, 34}

# Ring atoms (C1-C6) of the phenol test molecule, 1-based, as they appear in
# the [Atoms] section of test/phenol_scf/ref.molden / ref.h5.
PHENOL_RING_ATOMS = "1-6"

MAIN_SCRIPT = os.path.join(REPO_ROOT, "active_space_selection.py")
GET_PI_ORBITALS = os.path.join(REPO_ROOT, "get_pi_orbitals.py")
COMBINE_ALTER = os.path.join(REPO_ROOT, "combine_alter_files.py")


# ═══════════════════════════ shared helpers ═══════════════════════════════

def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)


def run_main_script(ref, target, ref_orbitals, active_space, tmpdir, extra_args=None):
    """
    Run active_space_selection.py with the given ref/target files.

    The script writes output files to the directory of the ``--target``
    file, so we copy the original target file into *tmpdir* and pass the
    copy as ``--target`` -- output files land in *tmpdir* and the checked-in
    test data directory is never touched.

    Returns (CompletedProcess, path_to_target_copy).
    """
    target_copy = shutil.copy(target, tmpdir)
    cmd = [
        sys.executable, MAIN_SCRIPT,
        "--ref", ref,
        "--target", target_copy,
        "--ref_orbitals", ref_orbitals,
    ]
    if active_space is not None:
        cmd.extend(["--active_space", active_space])
    if extra_args:
        cmd.extend(extra_args)
    return run(cmd), target_copy


def run_get_pi_orbitals(target, atoms, tmpdir, extra_args=None):
    target_copy = shutil.copy(target, tmpdir)
    cmd = [
        sys.executable, GET_PI_ORBITALS,
        "--target", target_copy,
        "--atoms", atoms,
    ]
    if extra_args:
        cmd.extend(extra_args)
    return run(cmd), target_copy


def parse_overlap_lines(stdout):
    """
    Return a dict {ref_orb: (target_orb, overlap)} parsed from the
    "Reference -> Target (Overlap)" section of active_space_selection.py's
    stdout.
    """
    mapping = {}
    for line in stdout.splitlines():
        line = line.strip()
        if "->" in line and "(" in line:
            parts = line.split("->")
            try:
                ref_orb = int(parts[0].strip())
                rest = parts[1].split("(")[1].split(")")[0]
                overlap = float(rest)
                tar_part = parts[1].split("(")[0].strip()
                tar_orb = int(tar_part)
                mapping[ref_orb] = (tar_orb, overlap)
            except (ValueError, IndexError):
                pass
    return mapping


def parse_pi_scores(stdout):
    """Parse the 'MO    PiScore[    In Active Space?]' table from get_pi_orbitals.py's stdout."""
    scores = {}
    for line in stdout.splitlines():
        m = re.match(r"^\s*(\d+)\s+([-+0-9.eE]+)(\s+(Yes|No))?\s*$", line)
        if m:
            scores[int(m.group(1))] = float(m.group(2))
    return scores


def read_alter(tmpdir):
    with open(os.path.join(tmpdir, "ALTER.txt")) as fh:
        return fh.read()


def read_rotation_csv(tmpdir):
    return np.loadtxt(os.path.join(tmpdir, "rotate.csv"), delimiter=",")


# ═══════════════════════════ shared fixtures ══════════════════════════════

@pytest.fixture(scope="module")
def molden_result(tmp_path_factory):
    tmpdir = str(tmp_path_factory.mktemp("molden"))
    result, target_copy = run_main_script(REF_MOLDEN, TAR_MOLDEN, REF_ORBS_STR, ACTIVE_STR, tmpdir)
    assert result.returncode == 0, (
        f"Molden script failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return result.stdout, os.path.dirname(target_copy)


@pytest.fixture(scope="module")
def h5_result(tmp_path_factory):
    tmpdir = str(tmp_path_factory.mktemp("h5"))
    result, target_copy = run_main_script(REF_H5, TAR_H5, REF_ORBS_STR, ACTIVE_STR, tmpdir)
    assert result.returncode == 0, (
        f"HDF5 script failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return result.stdout, os.path.dirname(target_copy)


@pytest.fixture(scope="module")
def full_run_alter(tmp_path_factory):
    tmpdir = str(tmp_path_factory.mktemp("full"))
    result, target_copy = run_main_script(REF_MOLDEN, TAR_MOLDEN, REF_ORBS_STR, ACTIVE_STR, tmpdir)
    assert result.returncode == 0, result.stderr
    with open(os.path.join(os.path.dirname(target_copy), "ALTER.txt")) as f:
        return f.read()


@pytest.fixture(scope="module")
def combined_alter(tmp_path_factory):
    """
    Splits the real phenol active space {19,23-27,34} -> {22-28} into two
    arbitrary, disjoint sub-spaces, runs active_space_selection.py once per
    sub-space to get two genuine ALTER files, and combines them with
    combine_alter_files.py.
    """
    tmpdir = str(tmp_path_factory.mktemp("combine"))
    part_a = os.path.join(tmpdir, "a")
    part_b = os.path.join(tmpdir, "b")
    os.makedirs(part_a)
    os.makedirs(part_b)

    res_a, target_a = run_main_script(REF_MOLDEN, TAR_MOLDEN, "19,23,24", "22,23,24", part_a)
    assert res_a.returncode == 0, res_a.stderr
    res_b, target_b = run_main_script(REF_MOLDEN, TAR_MOLDEN, "25,26,27,34", "25,26,27,28", part_b)
    assert res_b.returncode == 0, res_b.stderr

    alter_a = os.path.join(os.path.dirname(target_a), "ALTER.txt")
    alter_b = os.path.join(os.path.dirname(target_b), "ALTER.txt")
    combined_path = os.path.join(tmpdir, "ALTER_combined.txt")

    result = run([
        sys.executable, COMBINE_ALTER,
        "-i", alter_a, alter_b,
        "--active_spaces", "22-24:25-28",
        "--total_active_space", "22-28",
        "-o", combined_path,
    ])
    assert result.returncode == 0, result.stderr
    with open(combined_path) as f:
        return f.read()


# ═══════════════════ active_space_selection.py: unit tests ═══════════════

class TestH5FileDetection:
    """Unit tests for the is_h5_file helper."""

    def test_h5_extension_detected(self):
        from active_space_selection import is_h5_file
        assert is_h5_file("file.h5")
        assert is_h5_file("FILE.H5")
        assert is_h5_file("path/to/file.hdf5")

    def test_molden_not_detected_as_h5(self):
        from active_space_selection import is_h5_file
        assert not is_h5_file("file.molden")
        assert not is_h5_file("file.txt")


class TestCoordExtraction:
    """Unit tests for coordinate extraction from HDF5 files."""

    def test_h5_coords_match_molden(self):
        from active_space_selection import extract_atom_coords
        coords_h5 = extract_atom_coords(REF_H5)
        coords_molden = extract_atom_coords(REF_MOLDEN)
        np.testing.assert_allclose(coords_h5, coords_molden, atol=1e-6,
                                   err_msg="H5 and Molden coordinates differ")

    def test_h5_coords_shape(self):
        from active_space_selection import extract_atom_coords
        coords = extract_atom_coords(REF_H5)
        assert coords.ndim == 2
        assert coords.shape[1] == 3


class TestRotationBlocks:
    """Unit tests for rotation-block extraction from HDF5 files."""

    def test_blocks_cover_all_aos(self):
        from active_space_selection import get_rotation_blocks_from_h5
        import h5py
        blocks = get_rotation_blocks_from_h5(REF_H5)
        with h5py.File(REF_H5, 'r') as f:
            n_basis = len(f['BASIS_FUNCTION_IDS'])
        covered = set()
        for ao_indices, m_values, l in blocks:
            assert len(ao_indices) == 2 * l + 1
            assert len(m_values) == 2 * l + 1
            covered.update(ao_indices)
        assert covered == set(range(n_basis)), "Rotation blocks do not cover all AOs"

    def test_m_values_complete(self):
        from active_space_selection import get_rotation_blocks_from_h5
        blocks = get_rotation_blocks_from_h5(REF_H5)
        for ao_indices, m_values, l in blocks:
            assert sorted(m_values) == list(range(-l, l + 1)), (
                f"m values for l={l} are not complete: {m_values}"
            )


class TestMORotation:
    """Unit tests for the HDF5 MO-coefficient rotation."""

    def test_identity_rotation_leaves_coefficients_unchanged(self):
        from active_space_selection import (
            get_rotation_blocks_from_h5,
            rotate_mo_coefficients_h5,
        )
        import h5py
        blocks = get_rotation_blocks_from_h5(REF_H5)
        with h5py.File(REF_H5, 'r') as f:
            n = int(np.sqrt(len(f['MO_VECTORS'])))
            C_raw = np.array(f['MO_VECTORS']).reshape(n, n)
        I3 = np.eye(3)
        C_rot = rotate_mo_coefficients_h5(C_raw, blocks, I3)
        np.testing.assert_allclose(C_rot, C_raw, atol=1e-10,
                                   err_msg="Identity rotation changed MO coefficients")

    def test_wigner_d_matrices_are_unitary(self):
        """The Wigner D matrices produced by sphecerix must be unitary."""
        from sphecerix import tesseral_wigner_D
        from scipy.spatial.transform import Rotation as R_scipy
        rot = R_scipy.from_euler('xyz', [15, -10, 25], degrees=True)
        for l in range(4):
            D = tesseral_wigner_D(l, rot)
            np.testing.assert_allclose(D @ D.T, np.eye(2 * l + 1), atol=1e-12,
                                       err_msg=f"D matrix for l={l} is not unitary")

    def test_ref_mos_orthonormal_in_ref_ao_overlap(self):
        """
        The reference MOs should already satisfy C @ S_AO @ C^T = I in the
        reference geometry (sanity-check that MO_VECTORS is correctly oriented).
        """
        import h5py
        with h5py.File(REF_H5, 'r') as f:
            n = int(np.sqrt(len(f['MO_VECTORS'])))
            C = np.array(f['MO_VECTORS']).reshape(n, n)
            S = np.array(f['AO_OVERLAP_MATRIX']).reshape(n, n)
        S_MO = C @ S @ C.T
        np.testing.assert_allclose(S_MO, np.eye(n), atol=1e-8,
                                   err_msg="Reference MOs are not orthonormal")


class TestLazyOrbkitImport:
    """Verify that orbkit is not imported at module load time."""

    def test_orbkit_not_imported_at_module_level(self):
        """
        active_space_selection should be importable without orbkit being
        installed, as long as no Molden workflow function is called.
        """
        import sys
        orbkit_mods = [k for k in sys.modules if k == 'orbkit' or k.startswith('orbkit.')]
        saved = {k: sys.modules.pop(k) for k in orbkit_mods}
        mod_name = 'active_space_selection'
        if mod_name in sys.modules:
            del sys.modules[mod_name]
        try:
            import active_space_selection  # noqa: F401 -- should not raise
        except ImportError as exc:
            raise AssertionError(
                f"active_space_selection raised ImportError on import (orbkit missing): {exc}"
            ) from exc
        finally:
            sys.modules.update(saved)


# ═══════════════ active_space_selection.py: integration tests ════════════

class TestH5VsMoldenConsistency:
    """Integration tests: HDF5 and Molden workflows must give identical results."""

    def test_orbital_mapping_identical(self, molden_result, h5_result):
        stdout_m, _ = molden_result
        stdout_h, _ = h5_result
        mapping_m = parse_overlap_lines(stdout_m)
        mapping_h = parse_overlap_lines(stdout_h)
        assert mapping_m.keys() == mapping_h.keys(), "Different reference orbitals reported"
        for orb in mapping_m:
            tar_m, ovlp_m = mapping_m[orb]
            tar_h, ovlp_h = mapping_h[orb]
            assert tar_m == tar_h, (
                f"Orbital {orb}: Molden maps to {tar_m}, HDF5 maps to {tar_h}"
            )
            assert abs(ovlp_m - ovlp_h) < 1e-4, (
                f"Orbital {orb}: overlap differs Molden={ovlp_m:.6f} HDF5={ovlp_h:.6f}"
            )

    def test_alter_files_identical(self, molden_result, h5_result):
        _, tmpdir_m = molden_result
        _, tmpdir_h = h5_result
        alter_line_m = read_alter(tmpdir_m).splitlines()[0]
        alter_line_h = read_alter(tmpdir_h).splitlines()[0]
        assert alter_line_m == alter_line_h, (
            f"ALTER.txt first lines differ:\n  Molden: {alter_line_m}\n  HDF5:   {alter_line_h}"
        )

    def test_rotation_matrices_close(self, molden_result, h5_result):
        _, tmpdir_m = molden_result
        _, tmpdir_h = h5_result
        rot_m = read_rotation_csv(tmpdir_m)
        rot_h = read_rotation_csv(tmpdir_h)
        np.testing.assert_allclose(rot_m, rot_h, atol=1e-5,
                                   err_msg="Rotation matrices differ between Molden and HDF5 runs")

    def test_h5_script_mode_reported(self, h5_result):
        stdout_h, _ = h5_result
        assert "Mode" in stdout_h and "HDF5" in stdout_h

    def test_molden_script_mode_reported(self, molden_result):
        stdout_m, _ = molden_result
        assert "Mode" in stdout_m and "Molden" in stdout_m


class TestMixedInputRejected:
    """Mixing HDF5 and Molden files must raise an error."""

    def test_mixed_ref_h5_target_molden_rejected(self, tmp_path):
        result, _ = run_main_script(REF_H5, TAR_MOLDEN, REF_ORBS_STR, ACTIVE_STR, str(tmp_path))
        assert result.returncode != 0
        assert "same type" in (result.stdout + result.stderr)

    def test_mixed_ref_molden_target_h5_rejected(self, tmp_path):
        result, _ = run_main_script(REF_MOLDEN, TAR_H5, REF_ORBS_STR, ACTIVE_STR, str(tmp_path))
        assert result.returncode != 0
        assert "same type" in (result.stdout + result.stderr)


class TestWriteRotatedH5:
    """Tests for the write_rotated_h5 function."""

    def test_rotated_h5_written_by_script(self, h5_result):
        _, tmpdir = h5_result
        assert os.path.isfile(os.path.join(tmpdir, "rotated.h5")), \
            "rotated.h5 was not written by the HDF5 workflow"

    def test_rotated_h5_mo_vectors_differ_from_ref(self, h5_result):
        import h5py
        _, tmpdir = h5_result
        rotated_h5 = os.path.join(tmpdir, "rotated.h5")
        with h5py.File(rotated_h5, 'r') as f_rot, h5py.File(REF_H5, 'r') as f_ref:
            C_rot = np.array(f_rot['MO_VECTORS'])
            C_ref = np.array(f_ref['MO_VECTORS'])
        assert not np.allclose(C_rot, C_ref, atol=1e-8), (
            "rotated.h5 MO_VECTORS are identical to the unrotated reference"
        )

    def test_rotated_h5_coords_match_target(self, h5_result):
        import h5py
        _, tmpdir = h5_result
        rotated_h5 = os.path.join(tmpdir, "rotated.h5")
        with h5py.File(rotated_h5, 'r') as f_rot, h5py.File(TAR_H5, 'r') as f_tar:
            coords_rot = np.array(f_rot['CENTER_COORDINATES'])
            coords_tar = np.array(f_tar['CENTER_COORDINATES'])
        np.testing.assert_allclose(coords_rot, coords_tar, atol=1e-8,
                                   err_msg="rotated.h5 coordinates do not match target")

    def test_write_rotated_h5_unit(self, tmp_path):
        import h5py
        from active_space_selection import (
            write_rotated_h5,
            get_rotation_blocks_from_h5,
            rotate_mo_coefficients_h5,
            extract_atom_coords_h5,
        )
        blocks = get_rotation_blocks_from_h5(REF_H5)
        with h5py.File(REF_H5, 'r') as f:
            n = int(np.sqrt(len(f['MO_VECTORS'])))
            C_raw = np.array(f['MO_VECTORS']).reshape(n, n)
        I3 = np.eye(3)
        C_rot = rotate_mo_coefficients_h5(C_raw, blocks, I3)
        tar_coords = extract_atom_coords_h5(TAR_H5)

        out_path = str(tmp_path / "rotated_test.h5")
        write_rotated_h5(REF_H5, out_path, C_rot, tar_coords)

        assert os.path.isfile(out_path)
        with h5py.File(out_path, 'r') as f, h5py.File(REF_H5, 'r') as f_ref:
            assert 'MO_VECTORS' in f
            assert 'CENTER_COORDINATES' in f
            np.testing.assert_allclose(np.array(f['MO_VECTORS']), C_rot.ravel(), atol=1e-12)
            np.testing.assert_allclose(np.array(f['CENTER_COORDINATES']), tar_coords, atol=1e-12)
            for attr in f_ref.attrs:
                assert attr in f.attrs, f"Root attribute '{attr}' missing from rotated.h5"
            for ds_name in ('MO_VECTORS', 'CENTER_COORDINATES'):
                for attr in f_ref[ds_name].attrs:
                    assert attr in f[ds_name].attrs, (
                        f"Attribute '{attr}' of dataset '{ds_name}' missing from rotated.h5"
                    )


class TestAutoActiveSpace:
    """Tests for the --act_elect/--act_orb automatic active-space feature."""

    def test_compute_active_space_unit(self):
        from active_space_selection import compute_active_space
        # 10 occupied, 10 virtual. HOMO=10, LUMO=11.
        # 8 electrons in 7 orbitals -> 4 occupied {7,8,9,10}, 3 virtual {11,12,13}.
        occupations = [2.0] * 10 + [0.0] * 10
        active = compute_active_space(occupations, 8, 7)
        assert active == [7, 8, 9, 10, 11, 12, 13]

    def test_compute_active_space_casscf_fails(self):
        """CASSCF-like occupations (not 2.0 or 0.0) must raise SystemExit."""
        from active_space_selection import compute_active_space
        occupations = [2.0] * 8 + [1.8, 0.2] + [0.0] * 10
        with pytest.raises(SystemExit):
            compute_active_space(occupations, 4, 4)

    def test_auto_active_space_integration_h5(self, tmp_path):
        res, target_copy = run_main_script(REF_H5, TAR_H5, REF_ORBS_STR, None, tmp_path,
                                            extra_args=["--act_elect", "8", "--act_orb", "7"])
        assert res.returncode == 0, res.stderr
        assert "Target Active Space: 22-28" in res.stdout
        alter_content = read_alter(os.path.dirname(target_copy))
        assert "ALTER = 2;" in alter_content

    def test_auto_active_space_integration_molden(self, tmp_path):
        res, _ = run_main_script(REF_MOLDEN, TAR_MOLDEN, REF_ORBS_STR, None, tmp_path,
                                  extra_args=["--act_elect", "8", "--act_orb", "7"])
        assert res.returncode == 0, res.stderr
        assert "Target Active Space: 22-28" in res.stdout

    def test_mutually_exclusive_validation(self, tmp_path):
        res, _ = run_main_script(REF_H5, TAR_H5, REF_ORBS_STR, ACTIVE_STR, tmp_path,
                                  extra_args=["--act_elect", "8", "--act_orb", "7"])
        assert res.returncode != 0
        assert "mutually exclusive" in res.stderr


# ═══════════════════════════ get_pi_orbitals.py ═══════════════════════════

class TestGetPiOrbitalsUnit:
    """Unit tests for pure-Python helpers in get_pi_orbitals.py (no QM I/O)."""

    def test_format_integer_list_roundtrip(self):
        from get_pi_orbitals import format_integer_list, parse_mixed_list
        assert format_integer_list([1, 2, 3, 5, 7, 8]) == "1-3,5,7-8"
        assert parse_mixed_list("1-3,5,7-8") == [1, 2, 3, 5, 7, 8]

    def test_format_integer_list_empty(self):
        from get_pi_orbitals import format_integer_list
        assert format_integer_list([]) == ""

    def test_best_fit_plane_of_planar_points_has_zero_distance(self):
        from get_pi_orbitals import best_fit_plane
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        centroid, normal, distances = best_fit_plane(coords)
        assert np.allclose(distances, 0.0, atol=1e-10)
        assert np.allclose(np.abs(normal), [0.0, 0.0, 1.0], atol=1e-10)

    def test_compute_pi_scores_picks_aligned_mo(self):
        from get_pi_orbitals import compute_pi_scores, normalize_projectors
        S = np.eye(3)
        C = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        projectors = normalize_projectors([(1, np.array([1.0, 0.0, 0.0]))], S)
        scores = compute_pi_scores(C, S, projectors)
        assert scores[0] > scores[1]
        assert np.isclose(scores[0], 1.0)
        assert np.isclose(scores[1], 0.0)

    def test_compute_active_space_matches_active_space_selection(self):
        """
        get_pi_orbitals.py re-implements compute_active_space() rather than
        importing it from active_space_selection.py. Guard against the two
        copies drifting apart by checking they agree on the same input.
        """
        import get_pi_orbitals
        import active_space_selection
        occupations = [2.0] * 10 + [0.0] * 10
        a = get_pi_orbitals.compute_active_space(occupations, 8, 7)
        b = active_space_selection.compute_active_space(occupations, 8, 7)
        assert a == b


class TestGetPiOrbitalsIntegration:
    """
    Integration tests against the real phenol MOs. The aromatic ring (atoms
    1-6) has a well-defined plane, and the phenol test case's active space
    ({19,23,24,25,26,27,34}) was chosen independently of get_pi_orbitals.py.
    If the top-7 orbitals by pi-score reproduce that exact set, it is a
    genuine cross-check of the pi-character metric, not a tautology.
    """

    def test_molden_top_orbitals_match_known_active_space(self, tmp_path):
        result, _ = run_get_pi_orbitals(REF_MOLDEN, PHENOL_RING_ATOMS, tmp_path,
                                         extra_args=["--top_n", "7"])
        assert result.returncode == 0, result.stderr
        assert set(parse_pi_scores(result.stdout).keys()) == PHENOL_ACTIVE_SET

    def test_h5_top_orbitals_match_known_active_space(self, tmp_path):
        result, _ = run_get_pi_orbitals(REF_H5, PHENOL_RING_ATOMS, tmp_path,
                                         extra_args=["--top_n", "7"])
        assert result.returncode == 0, result.stderr
        assert set(parse_pi_scores(result.stdout).keys()) == PHENOL_ACTIVE_SET

    def test_molden_and_h5_pi_scores_agree(self, tmp_path):
        molden_dir = tmp_path / "m"
        h5_dir = tmp_path / "h"
        molden_dir.mkdir()
        h5_dir.mkdir()
        res_molden, _ = run_get_pi_orbitals(REF_MOLDEN, PHENOL_RING_ATOMS, molden_dir,
                                             extra_args=["--top_n", "15"])
        res_h5, _ = run_get_pi_orbitals(REF_H5, PHENOL_RING_ATOMS, h5_dir,
                                         extra_args=["--top_n", "15"])
        assert res_molden.returncode == 0, res_molden.stderr
        assert res_h5.returncode == 0, res_h5.stderr
        scores_m = parse_pi_scores(res_molden.stdout)
        scores_h = parse_pi_scores(res_h5.stdout)
        assert set(scores_m.keys()) == set(scores_h.keys())
        for mo in scores_m:
            assert scores_m[mo] == pytest.approx(scores_h[mo], abs=1e-4)

    def test_active_space_matching_pi_orbitals_needs_no_swaps(self, tmp_path):
        """When --active_space is exactly the top pi-character orbitals, the
        generated ALTER block should require zero swaps (ALTER = 0;)."""
        active_str = ",".join(str(i) for i in sorted(PHENOL_ACTIVE_SET))
        result, target_copy = run_get_pi_orbitals(
            REF_MOLDEN, PHENOL_RING_ATOMS, tmp_path,
            extra_args=["--active_space", active_str],
        )
        assert result.returncode == 0, result.stderr
        content = read_alter(os.path.dirname(target_copy))
        assert "ALTER = 0;" in content

    def test_active_space_out_of_range_rejected(self, tmp_path):
        result, _ = run_get_pi_orbitals(
            REF_MOLDEN, PHENOL_RING_ATOMS, tmp_path,
            extra_args=["--active_space", "9999"],
        )
        assert result.returncode != 0

    def test_planarity_warning_for_nonplanar_atom_selection(self, tmp_path):
        """Including a clearly out-of-plane atom (a ring hydrogen) should
        trigger the planarity warning against a tight threshold."""
        result, _ = run_get_pi_orbitals(
            REF_MOLDEN, "1-6,8", tmp_path,
            extra_args=["--planarity_threshold", "0.001"],
        )
        assert result.returncode == 0, result.stderr
        assert "not close to a single plane" in result.stdout

    def test_pi_projector_file_written(self, tmp_path):
        result, target_copy = run_get_pi_orbitals(REF_MOLDEN, PHENOL_RING_ATOMS, tmp_path)
        assert result.returncode == 0, result.stderr
        proj_path = os.path.join(os.path.dirname(target_copy), "pi_projectors.molden")
        assert os.path.isfile(proj_path)


class TestGetPiOrbitalsArgValidation:

    def test_top_n_must_be_positive(self, tmp_path):
        result, _ = run_get_pi_orbitals(REF_MOLDEN, PHENOL_RING_ATOMS, tmp_path,
                                         extra_args=["--top_n", "0"])
        assert result.returncode != 0

    def test_atom_index_out_of_range_rejected(self, tmp_path):
        result, _ = run_get_pi_orbitals(REF_MOLDEN, "1-6,999", tmp_path)
        assert result.returncode != 0

    def test_alter_without_active_space_rejected(self, tmp_path):
        result, _ = run_get_pi_orbitals(REF_MOLDEN, PHENOL_RING_ATOMS, tmp_path,
                                         extra_args=["--alter", "out.txt"])
        assert result.returncode != 0

    def test_active_space_and_act_elect_mutually_exclusive(self, tmp_path):
        result, _ = run_get_pi_orbitals(
            REF_MOLDEN, PHENOL_RING_ATOMS, tmp_path,
            extra_args=["--active_space", "19,23-27,34", "--act_elect", "8", "--act_orb", "7"],
        )
        assert result.returncode != 0
        assert "mutually exclusive" in result.stderr

    def test_act_elect_requires_act_orb(self, tmp_path):
        result, _ = run_get_pi_orbitals(
            REF_MOLDEN, PHENOL_RING_ATOMS, tmp_path,
            extra_args=["--act_elect", "8"],
        )
        assert result.returncode != 0
        assert "together" in result.stderr

    def test_missing_target_file(self, tmp_path):
        cmd = [sys.executable, GET_PI_ORBITALS, "--target", str(tmp_path / "does_not_exist.molden"),
               "--atoms", "1-6"]
        assert run(cmd).returncode != 0


# ═══════════════════════════ combine_alter_files.py ═══════════════════════

class TestCombineAlterFilesUnit:

    def test_parse_list_of_mo_lists(self):
        from combine_alter_files import parse_list_of_mo_lists
        assert parse_list_of_mo_lists("19-23:30-34") == [[19, 20, 21, 22, 23], [30, 31, 32, 33, 34]]

    def test_read_alter_swaps(self, tmp_path):
        from combine_alter_files import read_alter_swaps
        alter_file = tmp_path / "ALTER.txt"
        alter_file.write_text("ALTER = 2; 1 18 23; 1 34 28;  * Generated automatically\n")
        assert read_alter_swaps(str(alter_file)) == [(18, 23), (34, 28)]

    def test_read_alter_swaps_count_mismatch_raises(self, tmp_path):
        from combine_alter_files import read_alter_swaps
        alter_file = tmp_path / "ALTER.txt"
        alter_file.write_text("ALTER = 2; 1 18 23;  * Generated automatically\n")  # declares 2, has 1
        with pytest.raises(ValueError):
            read_alter_swaps(str(alter_file))

    def test_validate_no_overlap_raises_on_shared_orbital(self):
        from combine_alter_files import validate_no_overlap
        with pytest.raises(ValueError):
            validate_no_overlap([[19, 20, 21], [21, 22]])

    def test_validate_no_overlap_passes_for_disjoint(self):
        from combine_alter_files import validate_no_overlap
        validate_no_overlap([[19, 20, 21], [22, 23]])  # must not raise

    def test_write_alter_file_matches_active_space_selection_convention(self, tmp_path):
        from combine_alter_files import write_alter_file
        out = tmp_path / "ALTER.txt"
        write_alter_file(target_orbitals=[18, 24, 34], active_orbitals=[22, 24, 28], alter_path=str(out))
        content = out.read_text()
        assert content.startswith("ALTER = 2;")
        assert "1 18 22;" in content
        assert "1 34 28;" in content


class TestCombineAlterFilesIntegration:
    """
    Splits the real phenol active space into two arbitrary, disjoint
    sub-spaces, runs active_space_selection.py once per sub-space to get two
    genuine ALTER files, combines them with combine_alter_files.py, and
    checks the swap list against a single reference run over the full
    active space (test/ref_orbitals.txt / test/active_orbitals.txt).
    """

    def test_combined_swap_count_matches_full_run(self, full_run_alter, combined_alter):
        assert re.search(r"ALTER = (\d+);", combined_alter).group(1) == \
            re.search(r"ALTER = (\d+);", full_run_alter).group(1)

    def test_combined_swaps_match_full_run(self, full_run_alter, combined_alter, tmp_path):
        from combine_alter_files import read_alter_swaps
        full_file = tmp_path / "full.txt"
        combined_file = tmp_path / "combined.txt"
        full_file.write_text(full_run_alter)
        combined_file.write_text(combined_alter)
        assert set(read_alter_swaps(str(full_file))) == set(read_alter_swaps(str(combined_file)))

    def test_mismatched_input_and_active_spaces_count_rejected(self, tmp_path):
        alter_file = tmp_path / "ALTER.txt"
        alter_file.write_text("ALTER = 0;  * Generated automatically\n")
        result = run([
            sys.executable, COMBINE_ALTER,
            "-i", str(alter_file),
            "--active_spaces", "19-23:30-34",
            "--total_active_space", "19-23,30-34",
        ])
        assert result.returncode != 0

    def test_orbital_missing_from_total_active_space_rejected(self, tmp_path):
        alter_file = tmp_path / "ALTER.txt"
        alter_file.write_text("ALTER = 0;  * Generated automatically\n")
        result = run([
            sys.executable, COMBINE_ALTER,
            "-i", str(alter_file),
            "--active_spaces", "19-23",
            "--total_active_space", "19-22",  # 23 missing
        ])
        assert result.returncode != 0

    def test_duplicate_orbitals_in_total_active_space_rejected(self, tmp_path):
        alter_file = tmp_path / "ALTER.txt"
        alter_file.write_text("ALTER = 0;  * Generated automatically\n")
        result = run([
            sys.executable, COMBINE_ALTER,
            "-i", str(alter_file),
            "--active_spaces", "19-23",
            "--total_active_space", "19-23,20",  # 20 duplicated
        ])
        assert result.returncode != 0

    def test_missing_input_file_rejected(self, tmp_path):
        result = run([
            sys.executable, COMBINE_ALTER,
            "-i", str(tmp_path / "does_not_exist.txt"),
            "--active_spaces", "19-23",
            "--total_active_space", "19-23",
        ])
        assert result.returncode != 0


# ═══════════════════ CLI / argument-parsing edge cases ════════════════════

class TestActiveSpaceSelectionArgValidation:

    def test_mismatched_ref_files_and_ref_orbitals_count(self, tmp_path):
        """Two --ref files but only one --ref_orbitals list must be rejected."""
        target_copy = shutil.copy(TAR_MOLDEN, tmp_path)
        result = run([
            sys.executable, MAIN_SCRIPT,
            "--ref", REF_MOLDEN, TAR_MOLDEN,
            "--target", target_copy,
            "--ref_orbitals", REF_ORBS_STR,
            "--active_space", ACTIVE_STR,
        ])
        assert result.returncode != 0

    def test_malformed_active_space_range_rejected(self, tmp_path):
        target_copy = shutil.copy(TAR_MOLDEN, tmp_path)
        result = run([
            sys.executable, MAIN_SCRIPT,
            "--ref", REF_MOLDEN,
            "--target", target_copy,
            "--ref_orbitals", REF_ORBS_STR,
            "--active_space", "22-",  # malformed range
        ])
        assert result.returncode != 0

    def test_missing_ref_file_rejected(self, tmp_path):
        target_copy = shutil.copy(TAR_MOLDEN, tmp_path)
        result = run([
            sys.executable, MAIN_SCRIPT,
            "--ref", str(tmp_path / "no_such_ref.molden"),
            "--target", target_copy,
            "--ref_orbitals", REF_ORBS_STR,
            "--active_space", ACTIVE_STR,
        ])
        assert result.returncode != 0

    def test_missing_target_file_rejected(self, tmp_path):
        result = run([
            sys.executable, MAIN_SCRIPT,
            "--ref", REF_MOLDEN,
            "--target", str(tmp_path / "no_such_target.molden"),
            "--ref_orbitals", REF_ORBS_STR,
            "--active_space", ACTIVE_STR,
        ])
        assert result.returncode != 0

    def test_atoms_out_of_range_rejected(self, tmp_path):
        result, _ = run_main_script(REF_MOLDEN, TAR_MOLDEN, REF_ORBS_STR, ACTIVE_STR, tmp_path,
                                     extra_args=["--atoms", "9999"])
        assert result.returncode != 0


class TestMixedInputAndArgsAgree:
    """Mixed-input rejection is also exercised as an arg-validation case."""

    def test_active_space_and_act_elect_mutually_exclusive(self, tmp_path):
        result, _ = run_main_script(REF_MOLDEN, TAR_MOLDEN, REF_ORBS_STR, ACTIVE_STR, tmp_path,
                                     extra_args=["--act_elect", "8", "--act_orb", "7"])
        assert result.returncode != 0
        assert "mutually exclusive" in result.stderr


# ═══════════════════════════ determinism ═══════════════════════════════════

class TestDeterminism:
    """Running the tool twice on identical inputs must give identical outputs."""

    def test_molden_run_is_reproducible(self, tmp_path):
        dir1, dir2 = tmp_path / "run1", tmp_path / "run2"
        os.makedirs(dir1)
        os.makedirs(dir2)
        res1, target1 = run_main_script(REF_MOLDEN, TAR_MOLDEN, REF_ORBS_STR, ACTIVE_STR, dir1)
        res2, target2 = run_main_script(REF_MOLDEN, TAR_MOLDEN, REF_ORBS_STR, ACTIVE_STR, dir2)
        assert res1.returncode == 0 and res2.returncode == 0

        out1, out2 = os.path.dirname(target1), os.path.dirname(target2)
        for fname in ("ALTER.txt", "rotate.csv", "rotated.molden"):
            assert filecmp.cmp(os.path.join(out1, fname), os.path.join(out2, fname), shallow=False), \
                f"{fname} differs between two runs with identical inputs"

    def test_h5_run_is_reproducible(self, tmp_path):
        dir1, dir2 = tmp_path / "run1", tmp_path / "run2"
        os.makedirs(dir1)
        os.makedirs(dir2)
        res1, target1 = run_main_script(REF_H5, TAR_H5, REF_ORBS_STR, ACTIVE_STR, dir1)
        res2, target2 = run_main_script(REF_H5, TAR_H5, REF_ORBS_STR, ACTIVE_STR, dir2)
        assert res1.returncode == 0 and res2.returncode == 0

        out1, out2 = os.path.dirname(target1), os.path.dirname(target2)
        for fname in ("ALTER.txt", "rotate.csv"):
            assert filecmp.cmp(os.path.join(out1, fname), os.path.join(out2, fname), shallow=False), \
                f"{fname} differs between two runs with identical inputs"

        import h5py
        with h5py.File(os.path.join(out1, "rotated.h5")) as f1, \
             h5py.File(os.path.join(out2, "rotated.h5")) as f2:
            assert np.array_equal(f1["MO_VECTORS"][()], f2["MO_VECTORS"][()])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
