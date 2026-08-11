Create a benchmark
==================

Overview
--------

A **benchmark** is a reproducible evaluation setting: one dataset, one sampler,
one task, and one or more metrics. Most benchmarks are a single ``B_*.yaml``
file referencing existing components.

When to create one
------------------

Create a benchmark when you want to:

- Fix a standard train/val/test evaluation protocol for a dataset
- Swap task or metric while keeping the same data pipeline
- Publish a leaderboard-friendly setting others can rerun with ``bioverse val``

Walkthrough
-----------

Step 1 — Pick building blocks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Browse :doc:`../implementations/datasets`, :doc:`../implementations/tasks`, and
:doc:`../implementations/metrics`. For a first benchmark, clone an existing
``B_*.yaml`` and change one component at a time.

Step 2 — Add ``B_MYBENCH.yaml``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # Property prediction on QM9 dipole moment (mu).
   dataset: D_QNTMA9
   sampler: MoleculeSampler
   task:
     PropertyPredictionTask:
       property: mu
       level: molecule
   metric: MeanAbsoluteErrorMetric

Step 3 — Smoke-test the loader
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from bioverse.factory import BenchmarkFactory

   benchmark = BenchmarkFactory("B_MYBENCH")
   loader = benchmark.loader(partition="train", batch_size=2, progress=True)
   (X, y), data = next(iter(loader))
   print(y)

Step 4 — Wire into an experiment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   benchmark: B_MYBENCH
   transforms: []
   model: .model.MyModel
   trainer:
     backend: TorchBackend
     collater: LongCollater
     epochs: 10
     batch_size: 32

Required interface
------------------

.. list-table::
   :header-rows: 1

   * - Key
     - Value
   * - ``dataset``
     - ``D_*`` name, path, or inline dataset config
   * - ``sampler``
     - Sampler class name or kwargs dict
   * - ``task``
     - Task class name or kwargs dict
   * - ``metric``
     - Metric class, kwargs dict, or list for :class:`~bioverse.metric.MultiMetric`

Optional ``split`` overrides the default split name used by the benchmark.

Documentation metadata
----------------------

Add ``description`` and ``citation`` keys to your YAML — they appear on
:doc:`../implementations/benchmarks` and are ignored at runtime:

.. code-block:: yaml

   description: Inverse folding benchmark with recovery and BLOSUM metrics.
   citation:
     - |
       Li et al. "ProteinInvBench: A Benchmark for Protein Inverse Folding."
       (see ProteinInvBench repository for the current citation).
   dataset: D_INVC42
   ...

Multiple metrics
----------------

.. code-block:: yaml

   metric:
     - RecoveryMetric
     - BlosumScoreMetric

Offline vs live transforms
---------------------------

- **Experiment ``transforms``** — applied via :meth:`~bioverse.benchmark.Benchmark.apply`
  and persisted to disk before training
- **Experiment ``live_transforms``** — applied via :meth:`~bioverse.benchmark.Benchmark.live`
  on each loader access (augmentation)

Testing
-------

.. code-block:: python

   benchmark = BenchmarkFactory("B_MYBENCH")
   loader = benchmark.loader(partition="val", batch_size=4, progress=False)
   for (X, y), _ in loader:
       benchmark.metric.update(y, y)  # perfect predictions
       break
   benchmark.metric.result().to_console()

Run ``bioverse val experiment.yaml`` on a tiny ``limit_train_batches`` config
before opening a PR.

Common pitfalls
---------------

- **Sampler / task mismatch** — mutation tasks need :class:`~bioverse.samplers.mutation.MutationSampler`
- **Metric property** — regression metrics need the correct ``property`` field on
  structured ``y`` arrays
- **Missing ``transforms: []`` in experiment YAML** — the CLI requires the key even
  when empty

Submitting upstream
-------------------

Name files ``B_<SHORT>.yaml``. Ensure referenced components exist, add pytest
coverage (see ``tests/test_progym.py``), and add ``description`` and ``citation``
keys to the YAML. See :doc:`../guides/contributor`.
