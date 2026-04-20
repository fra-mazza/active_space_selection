"""
Tests to verify that the HDF5 workflow produces results consistent with the
Molden workflow for the phenol test case.

Run with:
    python -m pytest test/test_h5_support.py -v
or:
    python test/test_h5_support.py
"""

import sys
import os
import subprocess
import shutil
import numpy as np
import pytest

# Ensure the repo root is on the path so we can import the module directly
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
    REF_ORBS_STR = fh.read().strip()
with open(ACTIVE_FILE) as fh:
    ACTIVE_STR = fh.read().strip()

# ───────────────────────── helpers ──────────────────────────────────────────

def run_script(ref, target, tmpdir):
    """
    Run active_space_selection.py with the given ref / target files.

    The script writes output files to the directory of the ``--target`` file.
    We copy the original target file into *tmpdir* and pass the copy as
    ``--target`` so that output files land in *tmpdir* and do not pollute the
    test data directory.

    Returns (CompletedProcess, output_dir).
    """
    target_copy = shutil.copy(target, tmpdir)
    cmd = [
        sys.executable,
        os.path.join(REPO_ROOT, "active_space_selection.py"),
        "--ref", ref,
        "--target", target_copy,
        "--ref_orbitals", REF_ORBS_STR,
        "--active_space", ACTIVE_STR,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    return result, tmpdir


def parse_overlap_lines(stdout):
    """
    Return a dict {ref_orb: (target_orb, overlap)} parsed from the
    "Reference -> Target (Overlap)" section of the script's stdout.
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


def read_alter(tmpdir):
    alter_path = os.path.join(tmpdir, "ALTER.txt")
    with open(alter_path) as fh:
        return fh.read()


def read_rotation_csv(tmpdir):
    rotate_path = os.path.join(tmpdir, "rotate.csv")
    return np.loadtxt(rotate_path, delimiter=",")


# ───────────────────────── fixtures ─────────────────────────────────────────

@pytest.fixture(scope="module")
def molden_result(tmp_path_factory):
    tmpdir = str(tmp_path_factory.mktemp("molden"))
    result, out_dir = run_script(REF_MOLDEN, TAR_MOLDEN, tmpdir)
    assert result.returncode == 0, (
        f"Molden script failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return result.stdout, out_dir


@pytest.fixture(scope="module")
def h5_result(tmp_path_factory):
    tmpdir = str(tmp_path_factory.mktemp("h5"))
    result, out_dir = run_script(REF_H5, TAR_H5, tmpdir)
    assert result.returncode == 0, (
        f"HDF5 script failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return result.stdout, out_dir


# ───────────────────────── tests ─────────────────────────────────────────────

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


class TestMoldenRepresentationDetection:
    """Unit tests for spherical/cartesian tag detection in Molden files."""

    def test_missing_spherical_tags_means_cartesian_df(self, tmp_path):
        from active_space_selection import parse_molden_angular_representation
        p = tmp_path / "cart_df.molden"
        p.write_text(
            "[Molden Format]\n"
            "[Atoms] (AU)\nH 1 1 0.0 0.0 0.0\n"
            "[GTO]\n1\n d 1 1.00\n 1.0 1.0\n f 1 1.00\n 1.0 1.0\n"
            "[MO]\n"
        )
        rep = parse_molden_angular_representation(str(p))
        assert rep[2] == "cartesian"
        assert rep[3] == "cartesian"

    def test_presence_of_5d7f_tag_means_spherical_df(self, tmp_path):
        from active_space_selection import parse_molden_angular_representation
        p = tmp_path / "sph_df.molden"
        p.write_text(
            "[Molden Format]\n"
            "[5D7F]\n"
            "[Atoms] (AU)\nH 1 1 0.0 0.0 0.0\n"
            "[GTO]\n1\n d 1 1.00\n 1.0 1.0\n f 1 1.00\n 1.0 1.0\n"
            "[MO]\n"
        )
        rep = parse_molden_angular_representation(str(p))
        assert rep[2] == "spherical"
        assert rep[3] == "spherical"


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
        for ao_indices, m_values, l, representation in blocks:
            expected = 2 * l + 1 if representation == "spherical" else (l + 1) * (l + 2) // 2
            assert len(ao_indices) == expected
            if representation == "spherical":
                assert len(m_values) == 2 * l + 1
            else:
                assert len(m_values) == 0
            covered.update(ao_indices)
        assert covered == set(range(n_basis)), "Rotation blocks do not cover all AOs"

    def test_m_values_complete(self):
        from active_space_selection import get_rotation_blocks_from_h5
        blocks = get_rotation_blocks_from_h5(REF_H5)
        for ao_indices, m_values, l, representation in blocks:
            if representation == "spherical":
                assert sorted(m_values) == list(range(-l, l + 1)), (
                    f"m values for l={l} are not complete: {m_values}"
                )


class TestCartesianToSphericalTransform:
    """Unit tests for Schlegel–Frisch Cartesian/spherical transformations."""

    def test_transform_orthonormality_d_shell(self):
        from active_space_selection import get_cartesian_spherical_transforms
        C, C_inv = get_cartesian_spherical_transforms(2)
        I = C @ C_inv
        np.testing.assert_allclose(I, np.eye(5), atol=1e-12,
                                   err_msg="d-shell Cartesian/spherical transform/inverse mismatch")

    def test_transform_orthonormality_f_shell(self):
        from active_space_selection import get_cartesian_spherical_transforms
        C, C_inv = get_cartesian_spherical_transforms(3)
        I = C @ C_inv
        np.testing.assert_allclose(I, np.eye(7), atol=1e-12,
                                   err_msg="f-shell Cartesian/spherical transform/inverse mismatch")


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
        alter_m = read_alter(tmpdir_m)
        alter_h = read_alter(tmpdir_h)
        # Compare the ALTER= line (first line) which contains the orbital swaps
        alter_line_m = alter_m.splitlines()[0]
        alter_line_h = alter_h.splitlines()[0]
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
        assert "Mode: HDF5" in stdout_h

    def test_molden_script_mode_reported(self, molden_result):
        stdout_m, _ = molden_result
        assert "Mode: Molden" in stdout_m


class TestMixedInputRejected:
    """Mixing HDF5 and Molden files must raise an error."""

    def test_mixed_ref_h5_target_molden_rejected(self, tmp_path):
        result, _ = run_script(REF_H5, TAR_MOLDEN, str(tmp_path))
        assert result.returncode != 0
        combined = result.stdout + result.stderr
        assert "same type" in combined

    def test_mixed_ref_molden_target_h5_rejected(self, tmp_path):
        result, _ = run_script(REF_MOLDEN, TAR_H5, str(tmp_path))
        assert result.returncode != 0
        combined = result.stdout + result.stderr
        assert "same type" in combined


class TestWriteRotatedH5:
    """Tests for the write_rotated_h5 function."""

    def test_rotated_h5_written_by_script(self, h5_result):
        """The HDF5 workflow must produce a rotated.h5 output file."""
        _, tmpdir = h5_result
        rotated_h5 = os.path.join(tmpdir, "rotated.h5")
        assert os.path.isfile(rotated_h5), "rotated.h5 was not written by the HDF5 workflow"

    def test_rotated_h5_mo_vectors_differ_from_ref(self, h5_result):
        """MO_VECTORS in rotated.h5 must differ from the unrotated reference."""
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
        """CENTER_COORDINATES in rotated.h5 must match the target geometry."""
        import h5py
        _, tmpdir = h5_result
        rotated_h5 = os.path.join(tmpdir, "rotated.h5")
        with h5py.File(rotated_h5, 'r') as f_rot, h5py.File(TAR_H5, 'r') as f_tar:
            coords_rot = np.array(f_rot['CENTER_COORDINATES'])
            coords_tar = np.array(f_tar['CENTER_COORDINATES'])
        np.testing.assert_allclose(coords_rot, coords_tar, atol=1e-8,
                                   err_msg="rotated.h5 coordinates do not match target")

    def test_write_rotated_h5_unit(self, tmp_path):
        """Unit test for write_rotated_h5: output file contains expected datasets."""
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
            # Root-level attributes must be preserved
            for attr in f_ref.attrs:
                assert attr in f.attrs, f"Root attribute '{attr}' missing from rotated.h5"
            # Dataset attributes must be preserved for replaced datasets
            for ds_name in ('MO_VECTORS', 'CENTER_COORDINATES'):
                for attr in f_ref[ds_name].attrs:
                    assert attr in f[ds_name].attrs, (
                        f"Attribute '{attr}' of dataset '{ds_name}' missing from rotated.h5"
                    )


class TestLazyOrbkitImport:
    """Verify that orbkit is not imported at module load time."""

    def test_orbkit_not_imported_at_module_level(self):
        """
        active_space_selection should be importable without orbkit being installed,
        as long as no Molden workflow function is called.
        """
        import importlib
        import sys
        # Remove orbkit from sys.modules if present to simulate it being absent
        orbkit_mods = [k for k in sys.modules if k == 'orbkit' or k.startswith('orbkit.')]
        saved = {k: sys.modules.pop(k) for k in orbkit_mods}
        # Re-import the module; this should not raise even without orbkit
        mod_name = 'active_space_selection'
        if mod_name in sys.modules:
            del sys.modules[mod_name]
        try:
            import active_space_selection  # noqa: F401 – should not raise
        except ImportError as exc:
            raise AssertionError(
                f"active_space_selection raised ImportError on import (orbkit missing): {exc}"
            ) from exc
        finally:
            sys.modules.update(saved)


# ───────────────────────── standalone runner ─────────────────────────────────

if __name__ == "__main__":
    # Allow running without pytest for a quick sanity check
    import traceback
    passed = failed = 0
    unit_test_classes = [
        TestH5FileDetection,
        TestCoordExtraction,
        TestRotationBlocks,
        TestMORotation,
    ]
    for cls in unit_test_classes:
        obj = cls()
        for name in [m for m in dir(cls) if m.startswith("test_")]:
            method = getattr(obj, name)
            try:
                method()
                print(f"  PASS  {cls.__name__}::{name}")
                passed += 1
            except Exception as exc:
                print(f"  FAIL  {cls.__name__}::{name}: {exc}")
                traceback.print_exc()
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
