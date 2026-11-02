import awkward as ak
import numpy as np

from bioverse.adapters.protein_gym import centered_effects, keep_assay
from bioverse.collaters.long import LongCollater
from bioverse.data import Assets, Batch
from bioverse.tasks.pocket_contact_prediction import PocketContactPredictionTask
from bioverse.transforms.annotate_pocket_contacts import AnnotatePocketContacts
from bioverse.transforms.drop_peptide_residues import DropPeptideResidues
from bioverse.transforms.tokenize_residues import TokenizeResidues
from bioverse.utilities.peptide import pocket_distances, pocket_mask
from tests.dummy.dummy import dummy_scene
from tests.test_crop_to_pocket import _far_protein_scene, _IndexBatch


def test_pocket_distances_inf_on_peptide():
    labels = [["A", "R"], ["G", "K"]]
    types = ["protein", "peptide"]
    pos = [[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], [[4.0, 4.0, 0.0], [8.0, 4.0, 0.0]]]
    dist = pocket_distances(labels, types, pos)
    assert np.isinf(dist[1][0]) and np.isinf(dist[1][1])
    assert dist[0][1] < dist[0][0]
    assert pocket_mask(labels, types, pos, radius=8.0) == [[True, True], [False, False]]


def test_annotate_drop_leaves_protein_contacts():
    batch = TokenizeResidues().transform_batch(Batch(ak.Array([_far_protein_scene()])))
    batch = AnnotatePocketContacts(radius=8.0).transform_batch(batch)
    contacts = [int(x) for x in ak.to_list(ak.ravel(batch.residue_contact))]
    assert contacts == [1, 1, 0, 0, 0]
    dropped = DropPeptideResidues().transform_batch(batch)
    labels = [str(x) for x in ak.to_list(ak.ravel(dropped.residue_label))]
    assert labels == ["A", "R", "D"]
    kept = [int(x) for x in ak.to_list(ak.ravel(dropped.residue_contact))]
    assert kept == [1, 1, 0]


def test_pocket_contact_task_drops_peptide():
    batch = TokenizeResidues().transform_batch(Batch(ak.Array([_far_protein_scene()])))
    task = PocketContactPredictionTask(radius=8.0, drop_peptide=True)
    index = {
        "scene": ak.Array([0]),
        "frame": ak.Array([[0]]),
        "molecule": ak.Array([[[0]]]),
    }
    X, y = task(_IndexBatch(batch), assets=Assets({}), index=index)
    labels = [str(x) for x in ak.to_list(ak.ravel(X.residue_label))]
    assert "G" not in labels and "K" not in labels
    target = np.asarray(ak.to_numpy(ak.ravel(y["target"])))
    assert target.tolist() == [1.0, 1.0, 0.0]


def test_pocket_task_collate():
    batch = TokenizeResidues().transform_batch(Batch(ak.Array([_far_protein_scene()])))
    task = PocketContactPredictionTask(radius=8.0, drop_peptide=True)
    index = {
        "scene": ak.Array([0]),
        "frame": ak.Array([[0]]),
        "molecule": ak.Array([[[0]]]),
    }
    X, y = task(_IndexBatch(batch), assets=Assets({}), index=index)
    data = LongCollater.collate(X, y, assets=Assets({}))
    assert int(np.asarray(data.y).reshape(-1).sum()) == 2


def test_keep_assay_filters_tev_and_fitness():
    import pandas as pd

    ref = pd.DataFrame(
        {
            "DMS_id": ["TEVP_VIRAL", "KINASE_HUMAN", "FIT_ORG"],
            "selection_type": ["Activity", "Activity", "OrganismalFitness"],
            "UniProt_ID": ["P04517", "P12345", "Q9ZZZ9"],
        }
    )
    assert not keep_assay("TEVP_VIRAL", ["Activity"], ["TEV"], ref)
    assert keep_assay("KINASE_HUMAN", ["Activity"], ["TEV"], ref)
    assert not keep_assay("FIT_ORG", ["Activity", "Binding"], ["TEV"], ref)


def test_centered_effects_subtract_median_without_wt():
    import pandas as pd

    df = pd.DataFrame({"mutant": ["A2C", "A2D", "A2E"], "DMS_score": [1.0, 2.0, 3.0]})
    out = centered_effects(df, how="wt")
    assert out == [-1.0, 0.0, 1.0]


def test_centered_effects_mad_scale():
    import numpy as np
    import pandas as pd

    df = pd.DataFrame(
        {
            "mutant": ["A1A", "A2C", "A2D", "A2E"],
            "DMS_score": [10.0, 9.0, 10.0, 11.0],
        }
    )
    out = np.asarray(centered_effects(df, how="wt", scale="mad"))
    mad = 1.4826 * np.median(np.abs(np.array([0.0, -1.0, 0.0, 1.0])))
    np.testing.assert_allclose(out, np.array([0.0, -1.0, 0.0, 1.0]) / mad)


def test_factories_resolve_pocket_transforms():
    from bioverse.factory import TransformFactory, _load_module

    TransformFactory([{"AnnotatePocketContacts": {"radius": 8.0}}])
    TransformFactory([{"DropPeptideResidues": {}}])
    _load_module(
        {"PocketContactPredictionTask": {"radius": 8.0, "drop_peptide": True}},
        "tasks",
    )
