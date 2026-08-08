import awkward as ak
import numpy as np

from ..transform import Transform


def _first_atom_mask(labels):
    labels = np.asarray(labels)
    if len(labels) == 0:
        return np.array([], dtype=bool)
    _, idx = np.unique(labels, return_index=True)
    mask = np.zeros(len(labels), dtype=bool)
    mask[np.sort(idx)] = True
    return mask


class DeduplicateAtoms(Transform):

    def transform_batch(self, batch):
        mask = ak.Array(
            [
                _first_atom_mask(labels)
                for labels in ak.to_list(batch.residues.atom_label)
            ]
        )

        def apply_mask(values):
            return ak.Array(
                [
                    np.asarray(value)[atom_mask]
                    for value, atom_mask in zip(ak.to_list(values), ak.to_list(mask))
                ]
            )

        for column in list(batch.data.keys()):
            if column.startswith("atom_"):
                setattr(
                    batch.residues,
                    column,
                    apply_mask(getattr(batch.residues, column)),
                )

        batch.toc["atom"] = ak.num(
            batch.data["atom_label"], axis=batch.prefixes.index("atom")
        )
        return batch
