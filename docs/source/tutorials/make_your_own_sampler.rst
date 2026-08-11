Create a sampler
================

Overview
--------

Samplers define **which rows of the dataset table-of-contents** participate in
training and evaluation. :meth:`~bioverse.sampler.Sampler.index` returns Awkward
indices; :meth:`~bioverse.sampler.Sampler.sample` handles shuffling, distributed
training, and batch grouping.

When to create one
------------------

Add a sampler when the unit of prediction is not covered by existing
implementations (:doc:`../implementations/samplers`):

- per-molecule, per-residue, per-frame, per-mutation, pairwise interfaces, etc.

Walkthrough
-----------

Step 1 — Implement ``index``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import awkward as ak

   from bioverse.sampler import Sampler


   class MyUnitSampler(Sampler):
       """Sample one row per molecule in the active split."""

       def index(self, toc, mask):
           molecules = toc[mask]["chain"]
           index = ak.zip([ak.local_index(molecules, i) for i in range(molecules.ndim)])
           # ... remap to scene / frame / molecule fields ...
           return ak.Array({"scene": scenes, "frame": frames, "molecule": molecules})

See :class:`~bioverse.samplers.molecule.MoleculeSampler` for a complete example.

Step 2 — Choose compatible ``batch_on``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Experiment config controls batch grouping:

.. code-block:: yaml

   trainer:
     batch_on: molecules   # scenes | frames | molecules | residues | mutations

Your ``index`` output must include the fields required for the chosen ``batch_on``
value.

Step 3 — Validate with a benchmark
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   dataset: D_AFCATH
   sampler: MyUnitSampler
   task: PropertyPredictionTask
   metric: BinaryAccuracyMetric

Required interface
------------------

Implement :meth:`~bioverse.sampler.Sampler.index(toc, mask) -> ak.Array` with at
least a ``scene`` field. Additional fields (``frame``, ``molecule``, ``residue``,
…) depend on the task and ``batch_on`` setting.

You usually **do not** override :meth:`~bioverse.sampler.Sampler.sample` — it
handles DDP padding, shuffling, and batch sizing.

Testing
-------

.. code-block:: python

   from bioverse.factory import BenchmarkFactory

   benchmark = BenchmarkFactory("B_AFCATH")
   toc = benchmark.dataset.toc
   mask = benchmark.dataset.split["train", "train"]
   index = benchmark.sampler.index(toc, mask)
   assert "scene" in index.fields
   batches = benchmark.sampler.sample(
       benchmark.dataset, partition="train", split="default", batch_size=4
   )
   assert len(batches) > 0

Common pitfalls
---------------

- **Split mask shape** — ``mask`` selects scenes in the active partition; remap
  global scene IDs through ``toc`` after filtering
- **DDP batch count mismatch** — ``sample`` trims batches so every rank sees the
  same count; extremely small partitions may yield empty loaders
- **Mutation vs molecule batching** — mutation benchmarks typically use
  ``batch_on: mutations`` with :class:`~bioverse.samplers.mutation.MutationSampler`

Submitting upstream
-------------------

Document supported ``batch_on`` values and expected index fields. See
:doc:`../guides/contributor`.
