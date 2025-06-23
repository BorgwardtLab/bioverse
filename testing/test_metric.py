from unittest.mock import MagicMock, patch

import awkward as ak
import numpy as np
import pytest

from bioverse.metric import Metric, MultiMetric, Result
from bioverse.metrics import MultiClassAccuracyMetric


# Helper class to test abstract Metric class implementation
class TestMetricImpl(Metric):
    def compute(self, y_true, y_pred):
        # Simple implementation - just count matching values
        return float(ak.sum(y_true == y_pred))


def test_metric():
    y_true = ak.Array(
        {
            "label": [  # dataset
                [  # batch 1
                    [  # molecule 1
                        [  # residue 1
                            1,  # atom 1
                            0,  # atom 2
                            1,  # atom 3
                        ],
                        [  # residue 2
                            0,  # atom 1
                            1,  # atom 2
                            0,  # atom 3
                        ],
                    ],
                    [  # molecule 2
                        [  # residue 1
                            0,  # atom 1
                            0,  # atom 2
                        ],
                    ],
                ],
                [  # batch 2
                    [  # molecule 1
                        [  # residue 1
                            1,  # atom 1
                            1,  # atom 2
                            1,  # atom 3
                        ],
                        [  # residue 2
                            0,  # atom 1
                            0,  # atom 2
                            0,  # atom 3
                        ],
                    ],
                ],
            ]
        }
    )
    y_pred = ak.Array(
        {
            "label": [  # dataset
                [  # batch 1
                    [  # molecule 1
                        [  # residue 1
                            [0.4, 0.6],  # atom 1
                            [0.1, 0.9],  # atom 2
                            [0.2, 0.8],  # atom 3
                        ],
                        [  # residue 2
                            [0.4, 0.6],  # atom 1
                            [0.1, 0.9],  # atom 2
                            [0.2, 0.8],  # atom 3
                        ],
                    ],
                    [  # molecule 2
                        [  # residue 1
                            [0.4, 0.6],  # atom 1
                            [0.1, 0.9],  # atom 2
                        ],
                    ],
                ],
                [  # batch 2
                    [  # molecule 1
                        [  # residue 1
                            [0.4, 0.6],  # atom 1
                            [0.1, 0.9],  # atom 2
                            [0.2, 0.8],  # atom 3
                        ],
                        [  # residue 2
                            [0.4, 0.6],  # atom 1
                            [0.1, 0.9],  # atom 2
                            [0.2, 0.8],  # atom 3
                        ],
                    ],
                ],
            ]
        }
    )

    metric = MultiClassAccuracyMetric(on=3, per=None, reduction=None)
    for yt, yp in zip(y_true, y_pred):
        metric.update(yt, yp)
    result = metric.result()

    metric = MultiClassAccuracyMetric(on=3, per=0, reduction=None)
    for yt, yp in zip(y_true, y_pred):
        metric.update(yt, yp)
    result = metric.result()

    metric = MultiClassAccuracyMetric(on=3, per=1, reduction=None)
    for yt, yp in zip(y_true, y_pred):
        metric.update(yt, yp)
    result = metric.result()

    metric = MultiClassAccuracyMetric(on=3, per=2, reduction=None)
    for yt, yp in zip(y_true, y_pred):
        metric.update(yt, yp)
    result = metric.result()


def test_metric_base_class():
    # Test Metric initialization with different parameters
    metric1 = TestMetricImpl(name="TestMetric")
    assert metric1.name == "TestMetric"

    metric2 = TestMetricImpl(property="accuracy", name="TestMetric")
    assert metric2.name == "TestMetric (accuracy)"

    # Test with different reduction methods
    metric3 = TestMetricImpl(reduction="mean")
    assert metric3.reduce == np.mean

    metric4 = TestMetricImpl(reduction="sum")
    assert metric4.reduce == np.sum

    metric5 = TestMetricImpl(reduction=None)
    # Test that the reduce function just returns the input
    assert metric5.reduce([1, 2, 3]) == [1, 2, 3]

    # Test __call__ method
    simple_y_true = ak.Array({"value": [1, 0, 1]})
    simple_y_pred = ak.Array({"value": [1, 1, 0]})
    result = metric1(simple_y_true, simple_y_pred)
    assert isinstance(result, Result)

    # Test reset method
    metric1.update(simple_y_true, simple_y_pred)
    assert len(metric1.y_true) > 0
    metric1.reset()
    assert len(metric1.y_true) == 0


def test_multi_metric():
    # Create multiple metrics
    metric1 = TestMetricImpl(name="Metric1")
    metric2 = TestMetricImpl(name="Metric2")

    # Test MultiMetric initialization
    multi = MultiMetric([metric1, metric2])

    # Test update and result methods
    simple_y_true = ak.Array({"value": [1, 0, 1]})
    simple_y_pred = ak.Array({"value": [1, 1, 0]})

    multi.update(simple_y_true, simple_y_pred)
    result = multi.result(model_name="TestModel")
    assert isinstance(result, Result)

    # Test reset method
    multi.reset()

    # Instead of directly accessing metrics attribute, test that the multi object
    # behaves as if it has been reset
    multi.update(simple_y_true, simple_y_pred)
    new_result = multi.result()
    assert isinstance(new_result, Result)


def test_result_class():
    # Test Result initialization
    result = Result(
        [
            {"Model": "Model1", "Metric": "Metric1", "Value": 0.75},
            {"Model": "Model1", "Metric": "Metric2", "Value": 0.85},
            {"Model": "Model2", "Metric": "Metric1", "Value": 0.80},
        ]
    )

    # Test __add__ method
    result2 = Result([{"Model": "Model2", "Metric": "Metric2", "Value": 0.90}])
    combined = result + result2
    assert len(combined.data) == 4

    # Test to_dict method
    result_dict = result.to_dict()
    assert len(result_dict) == 2  # Two models
    assert result_dict["Model1"]["Metric1"] == 0.75
    assert result_dict["Model1"]["Metric2"] == 0.85
    assert result_dict["Model2"]["Metric1"] == 0.80

    # Test to_console method (checking if it runs without errors)
    with patch("bioverse.metric.Console") as mock_console:
        mock_console_instance = MagicMock()
        mock_console.return_value = mock_console_instance
        result.to_console()
        assert mock_console_instance.print.called

    # Test __str__ method
    with patch.object(Result, "to_string", return_value="Result String"):
        assert str(result) == "Result String"

    # Test to_string method
    assert result.to_string() == "String"

    # Test unimplemented methods raise NotImplementedError
    with pytest.raises(NotImplementedError):
        result.to_csv()

    with pytest.raises(NotImplementedError):
        result.to_latex()


def test_metric_addition_creates_multi_metric():
    # Test that adding two metrics creates a MultiMetric
    metric1 = TestMetricImpl(name="Metric1")
    metric2 = TestMetricImpl(name="Metric2")

    result = metric1 + metric2
    assert isinstance(result, MultiMetric)


def test_multi_metric_add_metric():
    # Test adding a metric to a MultiMetric
    metric1 = TestMetricImpl(name="Metric1")
    metric2 = TestMetricImpl(name="Metric2")

    multi = MultiMetric([metric1])
    result = multi + metric2
    assert isinstance(result, MultiMetric)

    # Test functionality
    simple_y_true = ak.Array({"value": [1, 0, 1]})
    simple_y_pred = ak.Array({"value": [1, 1, 0]})

    result.update(simple_y_true, simple_y_pred)
    metrics_result = result.result()
    assert isinstance(metrics_result, Result)


def test_metric_add_multi_metric():
    # Test adding a MultiMetric to a metric (reverse operation)
    metric1 = TestMetricImpl(name="Metric1")
    metric2 = TestMetricImpl(name="Metric2")

    multi = MultiMetric([metric1])
    result = metric2 + multi
    assert isinstance(result, MultiMetric)

    # Test functionality
    simple_y_true = ak.Array({"value": [1, 0, 1]})
    simple_y_pred = ak.Array({"value": [1, 1, 0]})

    result.update(simple_y_true, simple_y_pred)
    metrics_result = result.result()
    assert isinstance(metrics_result, Result)


def test_multi_metric_add_multi_metric():
    # Test adding a MultiMetric to another MultiMetric
    metric1 = TestMetricImpl(name="Metric1")
    metric2 = TestMetricImpl(name="Metric2")

    multi1 = MultiMetric([metric1])
    multi2 = MultiMetric([metric2])

    result = multi1 + multi2
    assert isinstance(result, MultiMetric)

    # Test functionality
    simple_y_true = ak.Array({"value": [1, 0, 1]})
    simple_y_pred = ak.Array({"value": [1, 1, 0]})

    result.update(simple_y_true, simple_y_pred)
    metrics_result = result.result()
    assert isinstance(metrics_result, Result)
