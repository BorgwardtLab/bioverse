Utilities
=========

The :mod:`bioverse.utilities` package provides shared infrastructure used across
adapters, datasets, benchmarks, and transforms:

- **Configuration** (:mod:`bioverse.utilities.config`) — default paths
  (``dataset_path``, ``raw_path``, ``benchmarks_path``), random seed, worker
  count, and scratch storage. Override via environment variables such as
  ``BIOVERSE_ROOT``.
- **I/O** (:mod:`bioverse.utilities.io`) — shard serialization, parallel
  processing helpers, logging utilities, and the :class:`~bioverse.utilities.io.IteratorWithLength`
  wrapper used throughout the loader pipeline.
- **Arrays & geometry** (:mod:`bioverse.utilities.array`,
  :mod:`bioverse.utilities.geometry`, :mod:`bioverse.utilities.align`) —
  Awkward Array helpers, structural alignment, and coordinate math.
- **Constants & requirements** — shared enumerations and optional-dependency
  checks.

Most users interact with utilities indirectly through config paths and the CLI.
Extension authors may import these modules when building custom adapters or
transforms.

.. automodule:: bioverse.utilities.config
   :members:

.. automodule:: bioverse.utilities.io
   :members: save, load, IteratorWithLength, parallelize, save_shards, rebatch
