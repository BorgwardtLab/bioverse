import awkward as ak
import numpy as np
import pytest
from sklearn import metrics as sk_metrics

from bioverse.metrics import *


@pytest.fixture
def binary_data():
    # Create batch of binary predictions and labels with nested structure
    y_true = ak.Array({"label": [[0, 1, 1, 0], [1, 0, 1, 1]]})
    y_pred = ak.Array({"label": [[0.1, 0.9, 0.8, 0.3], [0.7, 0.3, 0.9, 0.8]]})
    return y_true, y_pred


@pytest.fixture
def multiclass_data():
    # Create batch of multiclass predictions and labels with nested structure
    y_true = ak.Array({"label": [[0, 1, 2, 1], [2, 0, 1, 2]]})
    y_pred = ak.Array(
        {
            "label": [
                [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.2, 0.6, 0.2]],
                [[0.1, 0.1, 0.8], [0.6, 0.2, 0.2], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]],
            ]
        }
    )
    return y_true, y_pred


@pytest.fixture
def multilabel_data():
    # Create batch of multilabel predictions and labels
    y_true = ak.Array({"label": [[1, 1, 0], [0, 1, 1], [1, 0, 1]]})
    y_pred = ak.Array({"label": [[0.9, 0.8, 0.2], [0.1, 0.7, 0.8], [0.7, 0.3, 0.9]]})
    return y_true, y_pred


@pytest.fixture
def regression_data():
    # Create batch of regression predictions and labels
    y_true = ak.Array({"label": [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]})
    y_pred = ak.Array({"label": [[1.1, 2.2, 2.8], [2.1, 3.1, 3.9]]})
    return y_true, y_pred


def test_binary_accuracy(binary_data):
    y_true, y_pred = binary_data
    metric = BinaryAccuracyMetric(on=2, per=1, reduction="mean")
    metric.update(y_true, y_pred)
    bio_acc = list(metric.result().to_dict()["Ours"].values())[0]
    sk_accs = []
    for yt, yp in zip(y_true["label"], y_pred["label"]):
        sk_accs.append(sk_metrics.accuracy_score(yt, yp > 0.5))
    sk_acc = np.mean(sk_accs)
    np.testing.assert_allclose(bio_acc, sk_acc, rtol=1e-5)


def test_multi_class_accuracy(multiclass_data):
    y_true, y_pred = multiclass_data
    metric = MultiClassAccuracyMetric(on=2, per=1, reduction="mean")
    metric.update(y_true, y_pred)
    bio_acc = list(metric.result().to_dict()["Ours"].values())[0]
    sk_accs = []
    for yt, yp in zip(y_true["label"], y_pred["label"]):
        yp_classes = np.argmax(yp, axis=-1)
        sk_accs.append(sk_metrics.accuracy_score(yt, yp_classes))
    sk_acc = np.mean(sk_accs)
    np.testing.assert_allclose(bio_acc, sk_acc, rtol=1e-5)


def test_balanced_binary_accuracy(binary_data):
    y_true, y_pred = binary_data
    metric = BalancedBinaryAccuracyMetric(on=2, per=1, reduction="mean")
    metric.update(y_true, y_pred)
    bio_acc = list(metric.result().to_dict()["Ours"].values())[0]
    sk_accs = []
    for yt, yp in zip(y_true["label"], y_pred["label"]):
        sk_accs.append(sk_metrics.balanced_accuracy_score(yt, yp > 0.5))
    sk_acc = np.mean(sk_accs)
    np.testing.assert_allclose(bio_acc, sk_acc, rtol=1e-5)


def test_precision(binary_data):
    y_true, y_pred = binary_data
    metric = PrecisionMetric(on=2, per=1, reduction="mean")
    metric.update(y_true, y_pred)
    bio_prec = list(metric.result().to_dict()["Ours"].values())[0]
    sk_precs = []
    for yt, yp in zip(y_true["label"], y_pred["label"]):
        sk_precs.append(sk_metrics.precision_score(yt, yp > 0.5))
    sk_prec = np.mean(sk_precs)
    np.testing.assert_allclose(bio_prec, sk_prec, rtol=1e-5)


def test_recall(binary_data):
    y_true, y_pred = binary_data
    metric = RecallMetric(on=2, per=1, reduction="mean")
    metric.update(y_true, y_pred)
    bio_rec = list(metric.result().to_dict()["Ours"].values())[0]
    sk_recs = []
    for yt, yp in zip(y_true["label"], y_pred["label"]):
        sk_recs.append(sk_metrics.recall_score(yt, yp > 0.5))
    sk_rec = np.mean(sk_recs)
    np.testing.assert_allclose(bio_rec, sk_rec, rtol=1e-5)


def test_f1_score(binary_data):
    y_true, y_pred = binary_data
    metric = F1ScoreMetric(on=2, per=1, reduction="mean")
    metric.update(y_true, y_pred)
    bio_f1 = list(metric.result().to_dict()["Ours"].values())[0]
    sk_f1s = []
    for yt, yp in zip(y_true["label"], y_pred["label"]):
        sk_f1s.append(sk_metrics.f1_score(yt, yp > 0.5))
    sk_f1 = np.mean(sk_f1s)
    np.testing.assert_allclose(bio_f1, sk_f1, rtol=1e-5)


def test_auroc(binary_data):
    y_true, y_pred = binary_data
    metric = AreaUnderReceiverOperatingCharacteristicCurveMetric(
        on=2, per=1, reduction="mean"
    )
    metric.update(y_true, y_pred)
    bio_auroc = list(metric.result().to_dict()["Ours"].values())[0]
    sk_aurocs = []
    for yt, yp in zip(y_true["label"], y_pred["label"]):
        sk_aurocs.append(sk_metrics.roc_auc_score(yt, yp))
    sk_auroc = np.mean(sk_aurocs)
    np.testing.assert_allclose(bio_auroc, sk_auroc, rtol=1e-5)


def test_auprc(binary_data):
    y_true, y_pred = binary_data
    metric = AreaUnderPrecisionRecallCurveMetric(on=2, per=1, reduction="mean")
    metric.update(y_true, y_pred)
    bio_auprc = list(metric.result().to_dict()["Ours"].values())[0]
    sk_auprcs = []
    for yt, yp in zip(y_true["label"], y_pred["label"]):
        sk_auprcs.append(sk_metrics.average_precision_score(yt, yp))
    sk_auprc = np.mean(sk_auprcs)
    np.testing.assert_allclose(bio_auprc, sk_auprc, rtol=1e-5)


def test_mae(regression_data):
    y_true, y_pred = regression_data
    metric = MeanAbsoluteErrorMetric(on=2, per=1, reduction="mean")
    metric.update(y_true, y_pred)
    bio_mae = list(metric.result().to_dict()["Ours"].values())[0]
    sk_maes = []
    for yt, yp in zip(y_true["label"], y_pred["label"]):
        sk_maes.append(sk_metrics.mean_absolute_error(yt, yp))
    sk_mae = np.mean(sk_maes)
    np.testing.assert_allclose(bio_mae, sk_mae, rtol=1e-5)


def test_mse(regression_data):
    y_true, y_pred = regression_data
    metric = MeanSquaredErrorMetric(on=2, per=1, reduction="mean")
    metric.update(y_true, y_pred)
    bio_mse = list(metric.result().to_dict()["Ours"].values())[0]
    sk_mses = []
    for yt, yp in zip(y_true["label"], y_pred["label"]):
        sk_mses.append(sk_metrics.mean_squared_error(yt, yp))
    sk_mse = np.mean(sk_mses)
    np.testing.assert_allclose(bio_mse, sk_mse, rtol=1e-5)


def test_r2(regression_data):
    y_true, y_pred = regression_data
    metric = CoefficientOfDeterminationMetric(on=2, per=1, reduction="mean")
    metric.update(y_true, y_pred)
    bio_r2 = list(metric.result().to_dict()["Ours"].values())[0]
    sk_r2s = []
    for yt, yp in zip(y_true["label"], y_pred["label"]):
        sk_r2s.append(sk_metrics.r2_score(yt, yp))
    sk_r2 = np.mean(sk_r2s)
    np.testing.assert_allclose(bio_r2, sk_r2, rtol=1e-5)


def test_pearsons_r(regression_data):
    y_true, y_pred = regression_data
    metric = PearsonsRMetric(on=2, per=1, reduction="mean")
    metric.update(y_true, y_pred)
    bio_pearson = list(metric.result().to_dict()["Ours"].values())[0]
    assert bio_pearson >= -1 and bio_pearson <= 1


def test_spearmans_rho(regression_data):
    y_true, y_pred = regression_data
    metric = SpearmansRhoMetric(on=2, per=1, reduction="mean")
    metric.update(y_true, y_pred)
    bio_spearman = list(metric.result().to_dict()["Ours"].values())[0]
    assert bio_spearman >= -1 and bio_spearman <= 1


def test_topk_accuracy(multiclass_data):
    y_true, y_pred = multiclass_data
    k = 2
    metric = TopKAccuracyMetric(k=k, on=2, per=1, reduction="mean")
    metric.update(y_true, y_pred)
    bio_topk = list(metric.result().to_dict()["Ours"].values())[0]
    sk_topks = []
    for yt, yp in zip(y_true["label"], y_pred["label"]):
        sk_topks.append(sk_metrics.top_k_accuracy_score(yt, yp, k=k))
    sk_topk = np.mean(sk_topks)
    np.testing.assert_allclose(bio_topk, sk_topk, rtol=1e-5)


def test_perplexity(multiclass_data):
    y_true, y_pred = multiclass_data
    metric = PerplexityMetric(on=2, per=1, reduction="mean")
    metric.update(y_true, y_pred)
    ppl = list(metric.result().to_dict()["Ours"].values())[0]
    assert ppl > 0


def test_recovery(multiclass_data):
    y_true, y_pred = multiclass_data
    metric = RecoveryMetric(on=2, per=1, reduction="mean")
    metric.update(y_true, y_pred)
    recovery = list(metric.result().to_dict()["Ours"].values())[0]
    assert recovery >= 0 and recovery <= 1


def test_balanced_multiclass_accuracy(multiclass_data):
    y_true, y_pred = multiclass_data
    metric = BalancedMultiClassAccuracyMetric(on=2, per=1, reduction="mean")
    metric.update(y_true, y_pred)
    balanced_accuracy = list(metric.result().to_dict()["Ours"].values())[0]
    sk_balanced_accuracies = []
    for yt, yp in zip(y_true["label"], y_pred["label"]):
        sk_balanced_accuracies.append(
            sk_metrics.balanced_accuracy_score(yt, np.argmax(yp, axis=-1))
        )
    sk_balanced_accuracy = np.mean(sk_balanced_accuracies)
    np.testing.assert_allclose(balanced_accuracy, sk_balanced_accuracy, rtol=1e-5)


def test_multilabel_accuracy(multilabel_data):
    y_true, y_pred = multilabel_data
    threshold = 0.5
    metric = MultiLabelAccuracyMetric(
        on=2, per=1, reduction="mean", threshold=threshold
    )
    metric.update(y_true, y_pred)
    multilabel_accuracy = list(metric.result().to_dict()["Ours"].values())[0]
    sk_multilabel_accuracies = []
    for yt, yp in zip(y_true["label"], y_pred["label"]):
        sk_multilabel_accuracies.append(sk_metrics.accuracy_score(yt, yp > threshold))
    sk_multilabel_accuracy = np.mean(sk_multilabel_accuracies)
    np.testing.assert_allclose(multilabel_accuracy, sk_multilabel_accuracy, rtol=1e-5)
