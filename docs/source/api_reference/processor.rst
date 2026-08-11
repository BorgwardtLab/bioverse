Processor
=========

A **processor** converts raw structure files on disk into Awkward Array records
that Bioverse can shard and index. Processors walk a directory tree, filter by
file extension (``.pdb``, ``.cif``, …), and parse each file in parallel via
:meth:`~bioverse.processor.Processor.process_file`.

Adapters call processors during :meth:`~bioverse.adapter.Adapter.download` to
turn downloaded archives into uniform records before a dataset is released.
Different processors handle different file formats; see
:doc:`../implementations/processors`.

.. automodule:: bioverse.processor
   :members: Processor
