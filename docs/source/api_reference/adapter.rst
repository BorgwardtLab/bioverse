Adapter
=======

An **adapter** is the entry point for bringing external data into Bioverse. It
knows how to fetch or generate raw files from a repository (AlphaFold, ProteinShake,
PDB, etc.), optionally runs a :class:`~bioverse.processor.Processor` to parse
structure files, and returns the three objects every downstream component expects:
a stream of :class:`~bioverse.data.Batch` shards, a :class:`~bioverse.data.Split`
with train/validation/test partitions, and :class:`~bioverse.data.Assets` for
auxiliary lookup tables (vocabularies, embeddings, metadata maps).

Adapters are typically invoked once when building or refreshing a
:class:`~bioverse.dataset.Dataset`. See :doc:`../implementations/adapters` for
available sources.

.. automodule:: bioverse.adapter
   :members: Adapter
