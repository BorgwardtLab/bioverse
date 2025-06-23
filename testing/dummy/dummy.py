import awkward as ak

from bioverse.utilities.io import batched


def dummy_scene():
    return {
        "frame_id": ak.Array([0]),
        "molecule_id": ak.Array([["dummy"]]),
        "chain_label": ak.Array([[["A"]]]),
        "residue_label": ak.Array([[[["A", "R", "N"]]]]),
        "atom_label": ak.Array(
            [[[[["CA", "CB", "N"], ["CA", "CB", "N"], ["CA", "CB", "N"]]]]]
        ),
        "atom_pos": ak.Array(
            [
                [
                    [
                        [
                            [
                                [0.2, 0.3, -0.9],
                                [0.2, 0.9, 0.1],
                                [0.7, 0.5, 0.2],
                            ],
                            [
                                [0.4, 0.5, 0.4],
                                [0.6, 0.7, 0.3],
                                [-0.2, 0.1, 0.1],
                            ],
                            [
                                [0.5, -0.6, 0.2],
                                [0.1, -0.7, 0.0],
                                [0.3, 0.3, 0.4],
                            ],
                        ]
                    ]
                ]
            ]
        ),
    }


def dummy_batches(n=10):
    return batched(dummy_scene() for _ in range(n))
