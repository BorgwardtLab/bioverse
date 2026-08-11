Code Structure
==============

Bioverse is organized as a pipeline from raw repositories to benchmark scores.
Each stage is defined by a small abstract base class; concrete implementations live
under :doc:`../implementations/adapters` and the other **Implementations** pages.

Pipeline
--------

.. code-block:: text

   Adapter.download()
        ↓
   Processor.process()   (optional, during download)
        ↓
   Batch + Split + Assets
        ↓
   Dataset.release() / apply(Transform)
        ↓
   Benchmark.loader()
        ├── Sampler.sample()  → indices into the dataset
        ├── Task.__call__()   → (features, targets)
        └── Metric.update()   → aggregated scores
        ↓
   Trainer + Backend + Collater

Core abstractions
-----------------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Component
     - Role
   * - :doc:`adapter`
     - Download raw data from an external source and produce initial batches, splits, and assets.
   * - :doc:`processor`
     - Parse structure files (PDB, mmCIF, …) into Awkward records during ingestion.
   * - :doc:`data`
     - In-memory and on-disk data containers: scenes, splits, and auxiliary lookup tables.
   * - :doc:`dataset`
     - Versioned, sharded datasets built from an adapter or derived from another dataset.
   * - :doc:`transform`
     - Offline or live modifications to batches, splits, and assets.
   * - :doc:`benchmark`
     - Bind a dataset to a sampler, task, and metric; expose data loaders for train/val/test.
   * - :doc:`sampler`
     - Decide which scenes, frames, or molecules form each training/evaluation batch.
   * - :doc:`task`
     - Extract model inputs and targets from a lazy view over on-disk shards.
   * - :doc:`metric`
     - Compare predictions to ground truth and produce leaderboard-ready results.
   * - :doc:`utilities`
     - Shared configuration, I/O, geometry, and array helpers used across the library.

Configuration files (``D_*.yaml``, ``B_*.yaml``) reference implementation classes
by name. :mod:`bioverse.factory` resolves those names at runtime. See the
:doc:`../guides/developer` guide for extension patterns.
