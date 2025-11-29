import numpy as np


def rotation_matrix_to_quaternion(R):
    """
    Convert a batch of 3x3 rotation matrices to quaternions.
    Input: R of shape (..., 3, 3)
    Output: quaternions of shape (..., 4) in (x, y, z, w) format
    """
    # Based on: https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
    batch_dims = R.shape[:-2]
    m00 = R[..., 0, 0]
    m01 = R[..., 0, 1]
    m02 = R[..., 0, 2]
    m10 = R[..., 1, 0]
    m11 = R[..., 1, 1]
    m12 = R[..., 1, 2]
    m20 = R[..., 2, 0]
    m21 = R[..., 2, 1]
    m22 = R[..., 2, 2]

    qw = np.sqrt(np.clip(1.0 + m00 + m11 + m22, a_min=0, a_max=None)) / 2
    qx = np.sqrt(np.clip(1.0 + m00 - m11 - m22, a_min=0, a_max=None)) / 2
    qy = np.sqrt(np.clip(1.0 - m00 + m11 - m22, a_min=0, a_max=None)) / 2
    qz = np.sqrt(np.clip(1.0 - m00 - m11 + m22, a_min=0, a_max=None)) / 2

    cond = np.stack([qw, qx, qy, qz], axis=-1)
    max_idx = np.argmax(cond, axis=-1)

    quat = np.zeros(batch_dims + (4,))

    for i in range(4):
        mask = max_idx == i
        if mask.any():
            if i == 0:
                quat[mask, 0] = (m21 - m12)[mask] / (4 * qw[mask])
                quat[mask, 1] = (m02 - m20)[mask] / (4 * qw[mask])
                quat[mask, 2] = (m10 - m01)[mask] / (4 * qw[mask])
                quat[mask, 3] = qw[mask]
            elif i == 1:
                quat[mask, 0] = qx[mask]
                quat[mask, 1] = (m01 + m10)[mask] / (4 * qx[mask])
                quat[mask, 2] = (m02 + m20)[mask] / (4 * qx[mask])
                quat[mask, 3] = (m21 - m12)[mask] / (4 * qx[mask])
            elif i == 2:
                quat[mask, 0] = (m01 + m10)[mask] / (4 * qy[mask])
                quat[mask, 1] = qy[mask]
                quat[mask, 2] = (m12 + m21)[mask] / (4 * qy[mask])
                quat[mask, 3] = (m02 - m20)[mask] / (4 * qy[mask])
            elif i == 3:
                quat[mask, 0] = (m02 + m20)[mask] / (4 * qz[mask])
                quat[mask, 1] = (m12 + m21)[mask] / (4 * qz[mask])
                quat[mask, 2] = qz[mask]
                quat[mask, 3] = (m10 - m01)[mask] / (4 * qz[mask])
    return quat
