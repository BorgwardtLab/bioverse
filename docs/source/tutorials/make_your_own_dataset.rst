Create a dataset
================

Overview
--------

A **dataset** is a versioned on-disk artifact that many benchmarks can share.
Dataset configs (``D_*.yaml``) live under ``bioverse/datasets/`` and declare
either an :class:`~bioverse.adapter.Adapter` that downloads raw data or a
``parent`` dataset to derive from.

When to create one
------------------

Create a new dataset when:

- You have a data source that should be reused across multiple benchmarks
- No existing ``D_*`` config covers your adapter + transform pipeline
- You want a stable, versioned shard store separate from experiment configs

Walkthrough
-----------

Step 1 — Implement or choose an adapter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the data source is new, implement :meth:`~bioverse.adapter.Adapter.download`
(see :doc:`make_your_own_processor` and :doc:`../implementations/adapters`). Otherwise
reuse an existing adapter such as ``ProteinShakeAdapter``.

Step 2 — Add ``D_MYDATA.yaml``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Minimal config:

.. code-block:: yaml

   # My lab's structure collection released as versioned shards.
   adapter: MyDataAdapter

With adapter kwargs:

.. code-block:: yaml

   adapter:
     AlphaFoldAdapter:
       name: swissprot_pdb
       version: v4

Derived dataset (apply transforms on release):

.. code-block:: yaml

   parent: D_AFFULL
   transforms:
     - FilterSequenceLength:
         max_length: 512
     - TokenizeResidues

Step 3 — Release and inspect
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from bioverse.factory import DatasetFactory

   dataset = DatasetFactory("D_MYDATA")
   print(len(dataset), dataset.split.names)
   batch = next(iter(dataset.shards))
   print(batch.data.keys())

The first call downloads, transforms, and writes shards under
``~/.bioverse/datasets/D_MYDATA/``.

Required interface
------------------

Dataset configs must specify **exactly one** of:

- ``adapter`` — class name or ``{ClassName: {kwargs}}`` under ``bioverse.adapters``
- ``parent`` — another ``D_*`` name or list of names (composed via
  :class:`~bioverse.dataset.ComposedDataset`)

Optional ``transforms`` run during :meth:`~bioverse.dataset.Dataset.release`.

Documentation metadata
----------------------

Add ``description`` and ``citation`` keys to your YAML — they appear on
:doc:`../implementations/datasets` and are ignored at runtime:

.. code-block:: yaml

   description: AlphaFold Swiss-Prot structures with default parsing pipeline.
   citation:
     - |
       Jumper et al. "Highly accurate protein structure prediction with AlphaFold."
       Nature 596, 583–589 (2021). https://doi.org/10.1038/s41586-021-03819-2
   adapter: AlphaFoldAdapter

Creating an adapter
-------------------

.. code-block:: python

   from bioverse.adapter import Adapter
   from bioverse.data import Assets, Split
   from bioverse.processors.pdb import PdbProcessor
   from bioverse.utilities.io import rebatch


   class MyDataAdapter(Adapter):
       """Download structures from our internal repository."""

       @classmethod
       def download(cls, split="train", **kwargs):
           records = PdbProcessor.process("/data/structures")
           batches = rebatch(records)
           split = Split({"scene_split": [split] * len(batches)})
           assets = Assets({"residue_tokens": list("ACDEFGHIKLMNPQRSTVWY")})
           return batches, split, assets

Place the module at ``bioverse/adapters/my_data.py``. Lazy import resolves
``MyDataAdapter`` automatically.

Testing
-------

.. code-block:: python

   def test_my_dataset_release():
       from bioverse.factory import DatasetFactory
       from bioverse.utilities import config

       config.workers = 1
       dataset = DatasetFactory("D_MYDATA")
       assert len(dataset) > 0
       assert dataset.split.names

Add fixture-sized downloads in tests; mock network calls when possible.

Common pitfalls
---------------

- **Missing transforms on release** — expensive transforms belong in dataset YAML,
  not repeated in every benchmark
- **Unstable splits** — define explicit ``Split`` partitions in the adapter rather
  than shuffling at download time without recording assignments
- **Large assets in every batch** — put vocabularies and embedding tables in
  :class:`~bioverse.data.Assets`, not duplicated per shard

Submitting upstream
-------------------

See :doc:`../guides/contributor`. Ship ``D_*.yaml`` under ``bioverse/datasets/``
and the adapter under ``bioverse/adapters/``. Add a pytest that calls
``DatasetFactory`` with ``config.workers = 1``.
