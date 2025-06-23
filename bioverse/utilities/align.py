import os
import shutil
import subprocess
import tempfile

import awkward as ak
import numpy as np
from biopandas.pdb import PandasPdb
from scipy.spatial.distance import cdist

from . import config
from .io import to_pdb


def find_closest_points(source, target):
    """Find the closest points in target for each point in source."""
    distances = cdist(source, target)
    return np.argmin(distances, axis=1)


def compute_rotation_translation(source, target, correspondences):
    """Compute optimal rotation and translation using SVD."""
    # Center the points
    source_centered = source - np.mean(source, axis=0)
    target_centered = target[correspondences] - np.mean(target[correspondences], axis=0)
    # Compute covariance matrix
    H = source_centered.T @ target_centered
    # SVD
    U, _, Vt = np.linalg.svd(H)
    # Compute rotation matrix
    R = Vt.T @ U.T
    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    # Compute translation
    t = np.mean(target[correspondences], axis=0) - R @ np.mean(source, axis=0)
    return R, t


def alignment(coords1, coords2, max_iterations=50, tolerance=1e-6):
    """
    Align two sets of 3D coordinates using the Iterative Closest Point algorithm.

    Args:
        coords1: Source coordinates array of shape (n, 3)
        coords2: Target coordinates array of shape (m, 3)
        max_iterations: Maximum number of iterations
        tolerance: Convergence threshold for mean squared error

    Returns:
        tuple: (aligned_coords1, coords2) where aligned_coords1 is the transformed source coordinates
    """
    # Convert to numpy arrays if they aren't already
    source = np.asarray(coords1)
    target = np.asarray(coords2)
    # Initialize transformation
    R = np.eye(3)
    t = np.zeros(3)
    prev_error = float("inf")
    for iteration in range(max_iterations):
        # Find closest points
        correspondences = find_closest_points(source, target)
        # Compute rotation and translation
        R_new, t_new = compute_rotation_translation(source, target, correspondences)
        # Update transformation
        R = R_new @ R
        t = R_new @ t + t_new
        # Transform source points
        source = (R_new @ source.T).T + t_new
        # Compute error
        error = np.mean(np.sum((source - target[correspondences]) ** 2, axis=1))
        # Check convergence
        if abs(prev_error - error) < tolerance:
            break
        prev_error = error
    return source, coords2


def _alignment(X1, X2):
    pdb1 = to_pdb(**{field: X1[field] for field in X1.fields})
    pdb2 = to_pdb(**{field: X2[field] for field in X2.fields})
    print(pdb1)
    print(pdb2)
    assert (
        not shutil.which("USalign") is None
    ), "Please install USalign from https://zhanggroup.org/US-align/"
    with tempfile.TemporaryDirectory(dir=config.scratch_path) as tmpdir:
        with open(f"{tmpdir}/pdb1.pdb", "w") as f:
            f.write(pdb1)
        with open(f"{tmpdir}/pdb2.pdb", "w") as f:
            f.write(pdb2)
        subprocess.run(
            [
                "USalign",
                "-o",
                f"{tmpdir}/superposition",
                f"{tmpdir}/pdb1.pdb",
                f"{tmpdir}/pdb2.pdb",
            ],
            stdout=subprocess.PIPE,
        )
        A = PandasPdb().read_pdb(f"{tmpdir}/superposition.pdb").df["ATOM"]
        B = PandasPdb().read_pdb(f"{tmpdir}/pdb2.pdb").df["ATOM"]
        coordsA = ak.Array(A[["x_coord", "y_coord", "z_coord"]].to_numpy())
        coordsB = ak.Array(B[["x_coord", "y_coord", "z_coord"]].to_numpy())
        # reshape to residues and chains
        residue_num = ak.run_lengths(A["residue_number"])
        chain_num = ak.run_lengths(ak.unflatten(A["chain_id"], residue_num).firsts())
        coordsA = ak.unflatten(ak.unflatten(coordsA, residue_num), chain_num)
        coordsB = ak.unflatten(ak.unflatten(coordsB, residue_num), chain_num)
    return coordsA, coordsB
