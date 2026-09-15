"""
Additional tests built on the existing phenol_scf test data, covering
features that test_h5_support.py does not exercise:

  - get_pi_orbitals.py (previously completely untested, and not
    referenced from the main README)
  - combine_alter_files.py (previously completely untested)
  - CLI / argument-parsing edge cases for all three scripts
  - determinism (repeated runs of active_space_selection.py produce
    byte-identical output files)

These tests deliberately reuse only test/phenol_scf/{ref,target}.{molden,h5}
so they can run with no new QM data. See the project's test-suite plan for
the additional scenarios (genuine CASSCF active space, higher angular
momenta, multi-reference RMSD selection, --atoms subsetting) that need
purpose-built QM calculations not yet available in the repository.

Run with:
    python -m pytest test/test_additional_features.py -v
"""

import filecmp
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import pytest

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
    REF_ORBS_STR = fh.read().strip()          # "19,23-27,34"
with open(ACTIVE_FILE) as fh:
    ACTIVE_STR = fh.read().strip()             # "22-28"

# Known-good active space for phenol, expanded to a plain int list. This is
# the *independently chosen* set of CASSCF active orbitals for this test
# molecule (picked by the person who set up test/phenol_scf, not derived
# from get_pi_orbitals.py), which is exactly what makes comparing
# get_pi_orbitals.py's purely-geometric ranking against it a meaningful
# check rather than a tautology.
PHENOL_ACTIVE_SET = {19, 23, 24, 25, 26, 27, 34}

# Ring atoms (C1-C6) of the phenol test molecule, 1-based, as they appear in
# the [Atoms] section of test/phenol_scf/ref.molden / ref.h5.
PHENOL_RING_ATOMS = "1-6"

GET_PI_ORBITALS = os.path.join(REPO_ROOT, "get_pi_orbitals.py")
COMBINE_ALTER = os.path.join(REPO_ROOT, "combine_alter_files.py")
MAIN_SCRIPT = os.path.join(REPO_ROOT, "active_space_selection.py")


# ───────────────────────── helpers ──────────────────────────────────────────

def run(cmd, cwd=None):
    return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd or REPO_ROOT)


def run_main_script(ref, target, ref_orbitals, active_space, tmpdir, extra_args=None):
    target_copy = shutil.copy(target, tmpdir)
    cmd = [
        sys.executable, MAIN_SCRIPT,
        "--ref", ref,
        "--target", target_copy,
        "--ref_orbitals", ref_orbitals,
        "--active_space", active_space,
    ]
    if extra_args:
        cmd.extend(extra_args)
    result = run(cmd)
    return result, target_copy


def run_get_pi_orbitals(target, atoms, tmpdir, extra_args=None):
    target_copy = shutil.copy(target, tmpdir)
    cmd = [
        sys.executable, GET_PI_ORBITALS,
        "--target", target_copy,
        "--atoms", atoms,
    ]
    if extra_args:
        cmd.extend(extra_args)
    result = run(cmd)
    return result, target_copy


def parse_pi_scores(stdout):
    """Parse the 'MO    PiScore[    In Active Space?]' table from stdout."""
    scores = {}
    for line in stdout.splitlines():
        m = re.match(r"^\s*(\d+)\s+([-+0-9.eE]+)(\s+(Yes|No))?\s*$", line)
        if m:
            scores[int(m.group(1))] = float(m.group(2))
    return scores


# ───────────────────────── get_pi_orbitals.py ────────────────────────────────

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
        # A perfect square in the z=0 plane.
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        centroid, normal, distances = best_fit_plane(coords)
        assert np.allclose(distances, 0.0, atol=1e-10)
        # Normal must be along +/- z.
        assert np.allclose(np.abs(normal), [0.0, 0.0, 1.0], atol=1e-10)

    def test_compute_pi_scores_picks_aligned_mo(self):
        from get_pi_orbitals import compute_pi_scores, normalize_projectors

        nbasis = 3
        S = np.eye(nbasis)
        # Two MOs: one purely along basis fn 0, one purely along fn 2.
        C = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        projector_vec = np.array([1.0, 0.0, 0.0])
        projectors = normalize_projectors([(1, projector_vec)], S)
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
    (test/active_orbitals.txt / test/ref_orbitals.txt, orbitals
    {19,23,24,25,26,27,34}) was chosen independently of get_pi_orbitals.py.
    If the top-7 orbitals by pi-score reproduce that exact set, it is a
    genuine cross-check of the pi-character metric, not a tautology.
    """

    def test_molden_top_orbitals_match_known_active_space(self, tmp_path):
        result, _ = run_get_pi_orbitals(REF_MOLDEN, PHENOL_RING_ATOMS, tmp_path,
                                         extra_args=["--top_n", "7"])
        assert result.returncode == 0, result.stderr
        scores = parse_pi_scores(result.stdout)
        assert set(scores.keys()) == PHENOL_ACTIVE_SET

    def test_h5_top_orbitals_match_known_active_space(self, tmp_path):
        result, _ = run_get_pi_orbitals(REF_H5, PHENOL_RING_ATOMS, tmp_path,
                                         extra_args=["--top_n", "7"])
        assert result.returncode == 0, result.stderr
        scores = parse_pi_scores(result.stdout)
        assert set(scores.keys()) == PHENOL_ACTIVE_SET

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
        """
        When --active_space is exactly the set of top pi-character orbitals,
        the generated ALTER block should require zero swaps (ALTER = 0;).
        """
        active_str = ",".join(str(i) for i in sorted(PHENOL_ACTIVE_SET))
        result, target_copy = run_get_pi_orbitals(
            REF_MOLDEN, PHENOL_RING_ATOMS, tmp_path,
            extra_args=["--active_space", active_str],
        )
        assert result.returncode == 0, result.stderr
        alter_path = os.path.join(os.path.dirname(target_copy), "ALTER.txt")
        with open(alter_path) as f:
            content = f.read()
        assert "ALTER = 0;" in content

    def test_active_space_out_of_range_rejected(self, tmp_path):
        result, _ = run_get_pi_orbitals(
            REF_MOLDEN, PHENOL_RING_ATOMS, tmp_path,
            extra_args=["--active_space", "9999"],
        )
        assert result.returncode != 0

    def test_planarity_warning_for_nonplanar_atom_selection(self, tmp_path):
        """Including a clearly out-of-plane atom (an OH/ring hydrogen) should
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
        result = run(cmd)
        assert result.returncode != 0


# ───────────────────────── combine_alter_files.py ────────────────────────────

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
        # Declares 2 swaps but only provides 1.
        alter_file.write_text("ALTER = 2; 1 18 23;  * Generated automatically\n")
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


@pytest.fixture(scope="module")
def full_run_alter(tmp_path_factory):
    tmpdir = str(tmp_path_factory.mktemp("full"))
    result, target_copy = run_main_script(REF_MOLDEN, TAR_MOLDEN, REF_ORBS_STR, ACTIVE_STR, tmpdir)
    assert result.returncode == 0, result.stderr
    with open(os.path.join(os.path.dirname(target_copy), "ALTER.txt")) as f:
        return f.read()


@pytest.fixture(scope="module")
def combined_alter(tmp_path_factory):
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


class TestCombineAlterFilesIntegration:
    """
    Splits the real phenol active space into two arbitrary, disjoint
    sub-spaces, runs active_space_selection.py once per sub-space to get two
    genuine ALTER files, combines them with combine_alter_files.py, and
    checks the swap list against a single reference run over the full
    active space (test/ref_orbitals.txt / test/active_orbitals.txt).
    """

    def test_combined_swap_count_matches_full_run(self, full_run_alter, combined_alter):
        from combine_alter_files import read_alter_swaps
        # read_alter_swaps just needs a file path; write both contents out via tmp files
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


# ───────────────────────── CLI / argument-parsing edge cases ─────────────────

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
            "--active_space", "22-",   # malformed range
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
        target_copy = shutil.copy(TAR_MOLDEN, tmp_path)
        result = run([
            sys.executable, MAIN_SCRIPT,
            "--ref", REF_MOLDEN,
            "--target", target_copy,
            "--ref_orbitals", REF_ORBS_STR,
            "--active_space", ACTIVE_STR,
            "--atoms", "9999",
        ])
        assert result.returncode != 0

    def test_active_space_and_act_elect_mutually_exclusive(self, tmp_path):
        target_copy = shutil.copy(TAR_MOLDEN, tmp_path)
        result = run([
            sys.executable, MAIN_SCRIPT,
            "--ref", REF_MOLDEN,
            "--target", target_copy,
            "--ref_orbitals", REF_ORBS_STR,
            "--active_space", ACTIVE_STR,
            "--act_elect", "8",
            "--act_orb", "7",
        ])
        assert result.returncode != 0
        assert "mutually exclusive" in result.stderr


# ───────────────────────── determinism ────────────────────────────────────────

class TestDeterminism:
    """Running the tool twice on identical inputs must give identical outputs."""

    def test_molden_run_is_reproducible(self, tmp_path):
        dir1 = tmp_path / "run1"
        dir2 = tmp_path / "run2"
        os.makedirs(dir1)
        os.makedirs(dir2)
        res1, target1 = run_main_script(REF_MOLDEN, TAR_MOLDEN, REF_ORBS_STR, ACTIVE_STR, dir1)
        res2, target2 = run_main_script(REF_MOLDEN, TAR_MOLDEN, REF_ORBS_STR, ACTIVE_STR, dir2)
        assert res1.returncode == 0 and res2.returncode == 0

        out1 = os.path.dirname(target1)
        out2 = os.path.dirname(target2)
        for fname in ("ALTER.txt", "rotate.csv", "rotated.molden"):
            assert filecmp.cmp(os.path.join(out1, fname), os.path.join(out2, fname), shallow=False), \
                f"{fname} differs between two runs with identical inputs"

    def test_h5_run_is_reproducible(self, tmp_path):
        dir1 = tmp_path / "run1"
        dir2 = tmp_path / "run2"
        os.makedirs(dir1)
        os.makedirs(dir2)
        res1, target1 = run_main_script(REF_H5, TAR_H5, REF_ORBS_STR, ACTIVE_STR, dir1)
        res2, target2 = run_main_script(REF_H5, TAR_H5, REF_ORBS_STR, ACTIVE_STR, dir2)
        assert res1.returncode == 0 and res2.returncode == 0

        out1 = os.path.dirname(target1)
        out2 = os.path.dirname(target2)
        for fname in ("ALTER.txt", "rotate.csv"):
            assert filecmp.cmp(os.path.join(out1, fname), os.path.join(out2, fname), shallow=False), \
                f"{fname} differs between two runs with identical inputs"

        import h5py
        with h5py.File(os.path.join(out1, "rotated.h5")) as f1, \
             h5py.File(os.path.join(out2, "rotated.h5")) as f2:
            assert np.array_equal(f1["MO_VECTORS"][()], f2["MO_VECTORS"][()])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
