import awkward as ak
import numpy as np

from bioverse.collaters.long import LongCollater
from bioverse.data import Assets, Batch
from bioverse.samplers.mutation import MutationSampler
from bioverse.tasks.mutation_effect_prediction import MutationEffectPredictionTask
from bioverse.virtual import first_unique_rows
from tests.dummy.dummy import dummy_scene


def test_as_float_accepts_numpy_scalars():
    from bioverse.tasks.mutation_effect_prediction import _as_float

    assert _as_float(np.float32(-0.25)) == -0.25
    assert _as_float(np.array(0.5)) == 0.5
    assert _as_float(ak.Array([[1.25]])) == 1.25


def test_first_unique_rows_keeps_first_occurrence():
    uniq, inverse = first_unique_rows(np.array([[3, 0], [3, 0], [5, 0], [3, 0]]))
    assert uniq.tolist() == [0, 2]
    assert inverse.tolist() == [0, 0, 1, 0]


def test_mutation_sampler_group_scenes_keeps_blocks():
    sampler = MutationSampler(group_scenes=True)
    sampler.rng = np.random.default_rng(0)
    n0, n1 = 5, 7
    n = n0 + n1
    mixed = np.array([0, 5, 1, 6, 2, 7, 3, 8, 4, 9, 10, 11])
    index = ak.Array(
        {
            "scene": np.concatenate(
                [np.zeros(n0, dtype=int), np.ones(n1, dtype=int)]
            )[mixed],
            "mutation": np.concatenate([np.arange(n0), np.arange(n1)])[mixed],
            "frame": ak.Array([[0] for _ in range(n)])[mixed],
            "molecule": ak.Array([[0] for _ in range(n)])[mixed],
        }
    )
    out = sampler._shuffle_grouped(index)
    scenes = np.asarray(ak.to_numpy(out["scene"]))
    assert int((np.diff(scenes) != 0).sum()) == 1
    assert set(scenes.tolist()) == {0, 1}


def _scene_with_mutations():
    scene = dummy_scene()
    scene["molecule_mutation_label"] = ak.Array([[[["G"], ["K"]]]])
    scene["molecule_mutation_pos"] = ak.Array([[[[0], [2]]]])
    scene["molecule_mutation_effect"] = ak.Array([[[[0.5], [-0.25]]]])
    return scene


def test_wt_task_fetches_unique_structures():
    batch = Batch(ak.Array([_scene_with_mutations(), _scene_with_mutations()]))

    class _VBatch:
        def __init__(self):
            self.n = 0

        def __getitem__(self, index):
            self.n += 1
            return batch[index]

    vbatch = _VBatch()
    index = {
        "scene": ak.Array([0, 0, 1, 0]),
        "frame": ak.Array([[0], [0], [0], [0]]),
        "molecule": ak.Array([[0], [0], [0], [0]]),
        "mutation": ak.Array([0, 1, 0, 1]),
    }
    X, y = MutationEffectPredictionTask(mutate_structure=False)(
        vbatch, Assets({}), index
    )
    assert vbatch.n == 1
    assert len(X) == 2
    assert np.asarray(ak.to_numpy(y["graph"])).tolist() == [0, 0, 1, 0]
    assert len(y["target"]) == 4
    data = LongCollater.collate(X, y, assets=Assets({}))
    assert int(np.asarray(data.num_molecules).reshape(-1)[0]) == 2
    assert len(np.asarray(data.y).reshape(-1)) == 4
    assert len(np.asarray(data.mutation_graph).reshape(-1)) == 4
