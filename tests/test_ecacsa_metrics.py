import awkward as ak
import numpy as np

from bioverse.metrics import (
    MeanCatalyticRankMetric,
    TopKPrecisionMetric,
    TopKRecallMetric,
)


def test_mean_catalytic_rank_perfect_ranking():
    metric = MeanCatalyticRankMetric(property="target")
    y_true = ak.Array({"target": [[False, True, False, True]]})
    y_pred = ak.Array({"target": [[0.1, 0.9, 0.2, 0.8]]})
    metric.update(y_true, y_pred)
    value = list(metric.result().to_dict()["Ours"].values())[0]
    np.testing.assert_allclose(value, 1.5)


def test_topk_recall_and_precision():
    y_true = ak.Array({"target": [[False, True, False, True, False]]})
    y_pred = ak.Array({"target": [[0.1, 0.9, 0.2, 0.8, 0.3]]})

    recall = TopKRecallMetric(k=2, property="target")
    recall.update(y_true, y_pred)
    recall_value = list(recall.result().to_dict()["Ours"].values())[0]
    np.testing.assert_allclose(recall_value, 1.0)

    precision = TopKPrecisionMetric(k=2, property="target")
    precision.update(y_true, y_pred)
    precision_value = list(precision.result().to_dict()["Ours"].values())[0]
    np.testing.assert_allclose(precision_value, 1.0)


def test_ranking_metrics_average_per_protein():
    proteins = [
        ([[False, True, False]], [[0.1, 0.9, 0.2]]),
        ([[True, False, False, True]], [[0.8, 0.1, 0.0, 0.7]]),
    ]

    rank = MeanCatalyticRankMetric(property="target")
    recall = TopKRecallMetric(k=2, property="target")
    precision = TopKPrecisionMetric(k=2, property="target")
    for y_true, y_pred in proteins:
        rank.update(ak.Array({"target": y_true}), ak.Array({"target": y_pred}))
        recall.update(ak.Array({"target": y_true}), ak.Array({"target": y_pred}))
        precision.update(ak.Array({"target": y_true}), ak.Array({"target": y_pred}))

    np.testing.assert_allclose(list(rank.result().to_dict()["Ours"].values())[0], 1.25)
    np.testing.assert_allclose(list(recall.result().to_dict()["Ours"].values())[0], 1.0)
    np.testing.assert_allclose(
        list(precision.result().to_dict()["Ours"].values())[0], 0.75
    )


def test_ranking_compute_is_vectorized_over_jagged_proteins():
    y_true = ak.Array([[False, True, False], [True, False, False, True]])
    y_pred = ak.Array([[0.1, 0.9, 0.2], [0.8, 0.1, 0.0, 0.7]])

    ranks = MeanCatalyticRankMetric().compute(y_true, y_pred)
    recalls = TopKRecallMetric(k=2).compute(y_true, y_pred)
    precisions = TopKPrecisionMetric(k=2).compute(y_true, y_pred)

    np.testing.assert_allclose(ak.to_numpy(ranks), [1.0, 1.5])
    np.testing.assert_allclose(ak.to_numpy(recalls), [1.0, 1.0])
    np.testing.assert_allclose(ak.to_numpy(precisions), [0.5, 1.0])
