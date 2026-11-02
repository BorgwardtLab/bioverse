import awkward as ak

from ..transform import Transform
from ..utilities import element_from_atom_name


def _map_chains(trees, depth, target, fn):
    if depth == target:
        return fn(trees)
    n = len(next(iter(trees.values())))
    children = [
        _map_chains({key: tree[i] for key, tree in trees.items()}, depth + 1, target, fn)
        for i in range(n)
    ]
    return {key: [child[key] for child in children] for key in trees}


class ExplodeLigandAtoms(Transform):
    """Turn each ligand atom into its own residue.

    Protein chains are unchanged. On ligand chains, ``residue_label`` becomes
    the atom element and each new residue holds exactly one atom, so a
    residue-level k-NN graph can connect protein residues to ligand atoms.
    """

    def __init__(self, chain_type="ligand"):
        self.chain_type = chain_type

    def transform_batch(self, batch):
        if "chain_type" not in batch or "atom_label" not in batch:
            return batch

        residue_cols = [key for key in batch.data if key.startswith("residue_")]
        atom_cols = [key for key in batch.data if key.startswith("atom_")]
        trees = {
            key: ak.to_list(batch.data[key])
            for key in residue_cols + atom_cols + ["chain_type"]
        }
        chain_depth = batch.prefixes.index("chain")

        def explode_chains(nodes):
            out = {key: [] for key in nodes}
            out["chain_type"] = nodes["chain_type"]
            for chain_idx, chain_type in enumerate(nodes["chain_type"]):
                if chain_type != self.chain_type:
                    for key in residue_cols + atom_cols:
                        out[key].append(nodes[key][chain_idx])
                    continue
                exploded = {key: [] for key in residue_cols + atom_cols}
                atom_labels = nodes["atom_label"][chain_idx]
                atom_pos = nodes["atom_pos"][chain_idx]
                for res_idx, labels in enumerate(atom_labels):
                    for atom_idx, label in enumerate(labels):
                        element = element_from_atom_name(label)
                        for key in residue_cols:
                            if key == "residue_label":
                                exploded[key].append(element)
                            elif key == "residue_pos":
                                exploded[key].append(atom_pos[res_idx][atom_idx])
                            else:
                                exploded[key].append(nodes[key][chain_idx][res_idx])
                        for key in atom_cols:
                            value = nodes[key][chain_idx][res_idx][atom_idx]
                            exploded[key].append(
                                [element if key == "atom_label" else value]
                            )
                for key in residue_cols + atom_cols:
                    out[key].append(exploded[key])
            return out

        out = _map_chains(trees, 0, chain_depth, explode_chains)
        for key in residue_cols + atom_cols:
            batch.data[key] = ak.Array(out[key])
        batch.toc["residue"] = ak.num(
            batch.data["residue_label"], axis=batch.prefixes.index("residue")
        )
        batch.toc["atom"] = ak.num(
            batch.data["atom_label"], axis=batch.prefixes.index("atom")
        )
        return batch
