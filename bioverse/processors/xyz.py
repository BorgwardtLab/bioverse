from pathlib import Path

import awkward as ak

from ..processor import Processor


class XyzProcessor(Processor):

    valid_extensions = [".xyz"]

    @classmethod
    def process_file(cls, path: str | Path) -> ak.Record:
        path = Path(path)
        with open(path, "r") as file:
            lines = file.readlines()
            name = path.name.split(".")[0]
            values = [float(x) for x in lines[1].split()[2:]]
            smiles = lines[-2].split()[0]
            lines = lines[2:-3]
            coords = []
            atom_type = []
            for line in lines:
                split = line.split()
                v = split[0]
                x = float(split[1].replace("*^", "E"))
                y = float(split[2].replace("*^", "E"))
                z = float(split[3].replace("*^", "E"))
                atom_type.append(v)
                coords.append([x, y, z])

        data = {
            "molecule_smiles": [[smiles]],
            "molecule_id": [[name]],
            "molecule_labels": [[values]],
            "atom_label": [[[[atom_type]]]],
            "atom_pos": [[[[coords]]]],
        }
        data = {k: ak.Array(v) for k, v in data.items()}
        return ak.Record(data)
