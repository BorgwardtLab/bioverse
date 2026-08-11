Metric
======

A **metric** scores model predictions against ground truth. Metrics accumulate
``y_true`` and ``y_pred`` arrays across batches via
:meth:`~bioverse.metric.Metric.update`, then reduce them in
:meth:`~bioverse.metric.Metric.compute` to a single value (accuracy, MAE, TM-score,
Spearman's ρ, etc.).

:class:`~bioverse.metric.MultiMetric` combines several metrics into one evaluation
pass. :class:`~bioverse.metric.Result` formats scores for console output, CSV, or
LaTeX tables and tracks whether higher or lower values are better.

Benchmarks call :meth:`~bioverse.benchmark.Benchmark.update` during evaluation,
which inverse-transforms predictions back to the original label space before
updating the metric. See :doc:`../implementations/metrics`.

.. automodule:: bioverse.metric
   :members: Metric, MultiMetric, Result
