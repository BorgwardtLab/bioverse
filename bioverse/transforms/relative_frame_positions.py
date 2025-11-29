import awkward as ak
import numpy as np

from ..transform import Transform
from ..utilities import rotation_matrix_to_quaternion


class RelativeFramePositions(Transform):

    def __init__(self):
        pass

    def transform_batch(self, batch):
        q = []
        for R, (i, j) in zip(
            batch.molecules.residue_frame_R, batch.molecules.molecule_graph
        ):  # todo: vectorize
            R = R.to_numpy()
            R_T = np.transpose(R, axes=(0, 2, 1))
            RR = R[:, np.newaxis] @ R_T[np.newaxis, :]
            q.append(rotation_matrix_to_quaternion(RR)[i, j])
        batch.molecules.molecule_graph_values = q
        return batch
