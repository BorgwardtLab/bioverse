Create a transform
==================

Overview
--------

Transforms modify **batches, splits, and assets**. They run either **offline**
(materialized to disk when a dataset is released) or **live** (applied on each
loader access for augmentation or expensive features).

When to create one
------------------

Add a transform when preprocessing or augmentation logic is missing from
:doc:`../implementations/transforms` (tokenization, graph construction,
normalization, filtering, etc.).

Walkthrough
-----------

Step 1 — Subclass :class:`~bioverse.transform.Transform`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from bioverse.transform import Transform


   class ScalePositions(Transform):
       """Multiply atom coordinates by a constant factor."""

       def __init__(self, scale: float = 1.0):
           super().__init__()
           self.scale = scale

       def transform_batch(self, batch):
           if "atom_pos" in batch:
               batch.data["atom_pos"] = batch.data["atom_pos"] * self.scale
           return batch

       def inverse_transform(self, y):
           # optional: undo scaling on predictions for metrics
           return y

Step 2 — Apply offline on a dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from bioverse.factory import DatasetFactory

   dataset = DatasetFactory("D_AFCATH")
   dataset.apply(ScalePositions(scale=0.1))

Or in dataset YAML:

.. code-block:: yaml

   adapter:
     AlphaFoldInvBenchAdapter:
       af_name: swissprot_pdb
       af_version: v4
   transforms:
     - ScalePositions:
         scale: 0.1

Step 3 — Apply live on a benchmark
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   benchmark: B_AFCATH
   live_transforms:
     - NormalizeVector:
         field: atom_force

Required interface
------------------

Override one or more of:

- ``transform_batch(batch)`` — per-shard changes (most common)
- ``transform_assets(assets)`` / ``transform_split(split)`` — metadata updates
- ``fit(batches, split, assets)`` — compute statistics before transforming
  (see :class:`~bioverse.transforms.filter_sequence_length.FilterSequenceLength`)

Set ``filter = "scenes"`` to drop scenes where a ``scene_filter`` column is false.

Offline vs live
---------------

.. list-table::
   :header-rows: 1

   * - Mode
     - Config key
     - When it runs
   * - Offline
     - ``transforms`` (dataset or experiment)
     - During dataset release; written to content-addressed directory
   * - Live
     - ``live_transforms`` (experiment)
     - On each loader access inside :class:`~bioverse.virtual.VirtualBatch`

Composition
-----------

Transforms chain via :class:`~bioverse.transform.Compose`. Order matters — later
transforms see output from earlier ones. Constructor kwargs participate in
:meth:`~bioverse.transform.Transform.hash` for cache directories.

Testing
-------

.. code-block:: python

   from bioverse.data import Batch
   import awkward as ak

   batch = Batch({"atom_pos": ak.Array([[[0.0, 0.0, 0.0]]])})
   out = ScalePositions(2.0).transform_batch(batch)
   assert out.atom_pos[0][0][0] == 0.0  # check scaling

Add regression tests in ``tests/test_transforms.py`` with small synthetic batches.

Common pitfalls
---------------

- **Forgetting ``inverse_transform``** — metrics may need predictions in original
  units; implement inverse when transforms change label scales
- **Non-deterministic live transforms** — acceptable for training; disable for
  reproducible evaluation runs
- **Scene filtering** — set ``scene_filter`` on batches and ``filter = "scenes"``
  to drop invalid structures consistently from splits

Submitting upstream
-------------------

Note compute/memory cost in the docstring. Prefer live transforms for heavy
augmentation. See :doc:`../guides/contributor`.
