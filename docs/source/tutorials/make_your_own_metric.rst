Create a metric
===============

Overview
--------

Metrics accumulate predictions and ground truth across evaluation batches, then
reduce them to scalar scores displayed on the leaderboard. They plug into
benchmarks via the ``metric`` YAML key.

When to create one
------------------

Add a metric when no existing implementation in :doc:`../implementations/metrics`
matches your scoring function (rank correlation, structural similarity, custom
domain scores, etc.).

Walkthrough
-----------

Step 1 — Subclass :class:`~bioverse.metric.Metric`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import awkward as ak
   import numpy as np

   from bioverse.metric import Metric


   class MeanAbsoluteErrorMetric(Metric):
       """Example regression metric (already built-in; shown for structure)."""

       better = "lower"

       def __init__(self, name="MAE", **kwargs):
           super().__init__(name=name, **kwargs)

       def compute(self, y_true: ak.Array, y_pred: ak.Array) -> float:
           return float(np.mean(np.abs(ak.to_numpy(y_true - y_pred))))

Step 2 — Register via filename convention
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Save as ``bioverse/metrics/my_score.py`` → class ``MyScoreMetric`` is discovered
automatically.

Step 3 — Reference from benchmark YAML
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   metric: MyScoreMetric

With options:

.. code-block:: yaml

   metric:
     MyScoreMetric:
       name: "Custom label"
       property: target

Required interface
------------------

- ``compute(y_true, y_pred)`` — return a float contribution for the current batch
- ``better`` — ``"higher"`` or ``"lower"`` for leaderboard sorting

Optional:

- ``before_compute()`` — preprocess arrays before scoring
- ``__init__`` kwargs passed to ``super()``: ``name``, ``property``, ``on``,
  ``per``, ``reduction``

Structured targets
------------------

When ``y`` is a structured Awkward array, pass ``property="target"`` (or another
field name) to select the column to score. Use ``on`` and ``per`` to control
aggregation axes (see :class:`~bioverse.metric.Metric`).

Multiple metrics
----------------

.. code-block:: yaml

   metric:
     - BinaryAccuracyMetric
     - MyScoreMetric

The factory wraps lists as :class:`~bioverse.metric.MultiMetric`.

Testing
-------

.. code-block:: python

   import awkward as ak
   from bioverse.metrics.my_score import MyScoreMetric

   metric = MyScoreMetric()
   y = ak.Array({"target": [1.0, 2.0, 3.0]})
   pred = ak.Array({"target": [1.1, 1.9, 3.2]})
   metric.update(y, pred)
   result = metric.result()
   result.to_console()

Add cases to ``tests/test_metrics.py``.

Common pitfalls
---------------

- **Inverse transforms** — :meth:`~bioverse.benchmark.Benchmark.update` applies
  inverse transforms before metrics; ensure ``compute`` expects original units
- **Per-batch vs global reduction** — ``compute`` runs on accumulated data when
  ``per`` is ``None``; set ``per`` for per-example scores
- **Wrong ``better`` direction** — lower-is-better metrics (MAE, MSE) must set
  ``better = "lower"``

Submitting upstream
-------------------

Document expected ``y_true`` / ``y_pred`` shapes and the ``better`` direction in
the class docstring. See :doc:`../guides/contributor`.
