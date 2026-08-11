Benchmark
=========

A **benchmark** is the central evaluation unit in Bioverse. It wires together four
components declared as class attributes:

- a :class:`~bioverse.dataset.Dataset`
- a :class:`~bioverse.sampler.Sampler`
- a :class:`~bioverse.task.Task`
- a :class:`~bioverse.metric.Metric`

:meth:`~bioverse.benchmark.Benchmark.loader` drives the full data path: the
sampler picks batch indices, the task extracts features and targets from a
:class:`~bioverse.virtual.VirtualBatch`, and an optional collater prepares
framework-ready batches for the trainer. After inference,
:meth:`~bioverse.benchmark.Benchmark.update` feeds predictions to the metric.

Benchmark configs (``B_*.yaml``) select a dataset and override sampler, task, or
metric settings. See :doc:`../implementations/benchmarks`.

.. automodule:: bioverse.benchmark
   :members: Benchmark
