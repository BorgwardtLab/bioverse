Dataset
=======

A **dataset** is a versioned, on-disk collection of sharded batches produced by
an adapter (or derived from another dataset). Datasets handle downloading,
releasing new versions, applying offline :class:`~bioverse.transform.Transform`
pipelines, and exposing lazy access to shards, splits, and assets.

Each dataset is stored under ``config.dataset_path`` as
``<Name>/v<version>/<transform-hash>/``. Offline transforms are materialized to
disk; live transforms are applied when batches are loaded inside a benchmark.

Implement :meth:`~bioverse.dataset.Dataset.release` to define how raw data
becomes sharded batches. Dataset configs (``D_*.yaml``) specify either an adapter
plus transforms or a parent dataset plus additional transforms. See
:doc:`../implementations/datasets`.

.. automodule:: bioverse.dataset
   :members: Dataset, ComposedDataset
