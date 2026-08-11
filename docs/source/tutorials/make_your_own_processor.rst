Create a processor
==================

Overview
--------

Processors parse **raw files on disk** into Awkward ``Record`` objects. Adapters
call :meth:`~bioverse.processor.Processor.process` to walk directories and
parallelize parsing before batching into :class:`~bioverse.data.Batch` shards.

When to create one
------------------

Add a processor when you need to support a new file format (custom text/binary
structure formats, lab-specific exports) before writing an adapter.

Walkthrough
-----------

Step 1 — Declare extensions and implement ``process_file``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pathlib import Path

   import awkward as ak

   from bioverse.processor import Processor


   class MyFormatProcessor(Processor):
       """Parse .myf files into Bioverse records."""

       valid_extensions = [".myf"]

       @classmethod
       def process_file(cls, path: str | Path) -> ak.Record | None:
           path = Path(path)
           if path.stat().st_size == 0:
               return None
           # populate scene / frame / molecule / residue / atom fields
           return ak.Record({"scene_id": path.stem, "atom_pos": ..., ...})

Step 2 — Process a directory tree
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from bioverse.processors.my_format import MyFormatProcessor

   records = MyFormatProcessor.process("/data/my_structures")
   for record in records:
       print(record.scene_id)

Step 3 — Use from an adapter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class MyDataAdapter(Adapter):
       def download(self, path="/data/raw"):
           records = MyFormatProcessor.process(path)
           batches = rebatch(records)
           ...

Required interface
------------------

- ``valid_extensions`` — suffixes handled by :meth:`~bioverse.processor.Processor.process`
- ``process_file(path)`` — return one ``ak.Record`` per file, or ``None`` to skip

Optional:

- ``exclude_key(path)`` — deduplicate or skip files during directory walks

Reference implementations
-------------------------

- :class:`~bioverse.processors.pdb.PdbProcessor` — PDB / PDB.gz
- :class:`~bioverse.processors.cif.CifProcessor` — mmCIF
- :class:`~bioverse.adapters.ares.AresPdbProcessor` — dataset-specific PDB layout

Testing
-------

Add small fixture files under ``tests/`` and assert field names/shapes in
``tests/test_processors.py``:

.. code-block:: python

   def test_my_format_processor(tmp_path):
       sample = tmp_path / "x.myf"
       sample.write_text("...")
       record = MyFormatProcessor.process_file(sample)
       assert record is not None
       assert "atom_pos" in record

Common pitfalls
---------------

- **Inconsistent hierarchy** — records must use Bioverse prefix conventions
  (``scene_``, ``molecule_``, ``residue_``, ``atom_`` fields)
- **Heavy optional dependencies** — gate imports and document extras in
  ``pyproject.toml`` when parsers need large libraries
- **Returning empty records** — return ``None`` instead of empty structures so
  ``process`` filters them out

Submitting upstream
-------------------

Keep parsers focused on I/O; defer featurization to transforms. See
:doc:`../guides/contributor`.
