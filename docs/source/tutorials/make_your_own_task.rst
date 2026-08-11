Create a task
=============

Overview
--------

Tasks translate sampler indices into **model inputs** ``X`` and **targets** ``y``.
They are the bridge between on-disk shards and your model's ``train_step``.

When to create one
------------------

Add a task when your prediction problem is not covered by
:doc:`../implementations/tasks` (property prediction, inverse folding, virtual
screening, structure prediction, etc.).

Walkthrough
-----------

Step 1 — Subclass :class:`~bioverse.task.Task`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import awkward as ak

   from bioverse.task import Task


   class MyTask(Task):
       """Describe what X contains and what y supervises."""

       def __init__(self, target_field="label"):
           self.target_field = target_field

       def __call__(self, vbatch, assets, index):
           X = vbatch[index["scene"], index["frame"], index["molecule"]]
           X.resolution = "atom"
           targets = X.molecules.__getattr__(f"molecule_{self.target_field}")
           y = ak.Array({"target": ak.flatten(targets, axis=None)})
           return X, y

Step 2 — Pair with sampler and metric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   dataset: D_AFCATH
   sampler: MoleculeSampler
   task:
     MyTask:
       target_field: label
   metric: BinaryAccuracyMetric

Step 3 — Inspect loader output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   benchmark = BenchmarkFactory("B_MYBENCH")
   loader = benchmark.loader(partition="train", batch_size=2, progress=False)
   (X, y), data = next(iter(loader))
   print(X.resolution, y.fields)

Required interface
------------------

Implement :meth:`~bioverse.task.Task.__call__(vbatch, assets, index)` returning
``(X, y)``:

- ``X`` — :class:`~bioverse.data.Batch` view with ``resolution`` set to
  ``"atom"`` or ``"residue"``
- ``y`` — Awkward array; metrics typically expect a ``target`` field

Reference tasks
---------------

- :class:`~bioverse.tasks.property_prediction.PropertyPredictionTask` — scalar/vector
  properties at configurable levels
- :class:`~bioverse.tasks.inverse_folding.InverseFoldingTask` — sequence from structure
- :class:`~bioverse.tasks.virtual_screen.VirtualScreenTask` — ligand-target scores

Configuration
-------------

Tasks are stateless; pass hyperparameters via YAML kwargs. The collater and
model receive collated ``data`` objects produced from ``X`` and ``y``.

Testing
-------

.. code-block:: python

   benchmark = BenchmarkFactory("B_MYBENCH")
   loader = benchmark.loader(partition="train", batch_size=1, progress=False)
   (X, y), collated = next(iter(loader))
   assert "target" in y.fields
   loss, out, tgt = benchmark.model.eval_step((X, y), collated)  # if model wired

Common pitfalls
---------------

- **Wrong resolution** — atom-level models need ``X.resolution = "atom"``; residue-level
  models use ``"residue"``
- **Variable-length targets** — pack sizes in ``y`` (see ``PropertyPredictionTask``)
  so collaters can pad correctly
- **Assets not loaded** — token featurization may require keys in
  :class:`~bioverse.data.Assets`; ensure the adapter populates them

Submitting upstream
-------------------

Document target semantics, compatible samplers/metrics, and expected collater in
the class docstring. See :doc:`../guides/contributor`.
