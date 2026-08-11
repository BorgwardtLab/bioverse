Data
====

Bioverse stores biomolecular data as **Awkward Array** records organized in a
fixed hierarchy: scenes → frames → molecules → residues → atoms. The three core
containers below travel through adapters, transforms, tasks, and metrics.

- :class:`~bioverse.data.Batch` — one shard of structured data with nested fields
  (coordinates, sequences, labels, graphs, …) and a table-of-contents (``toc``)
  that describes how many elements exist at each level.
- :class:`~bioverse.data.Split` — partition assignments (train, validation, test,
  or custom splits) keyed by scene, frame, molecule, or other levels.
- :class:`~bioverse.data.Assets` — a dictionary of shared resources referenced
  across batches (token vocabularies, precomputed embeddings, ID maps, etc.).

Tasks read data through :class:`~bioverse.virtual.VirtualBatch`, a lazy,
cache-backed view over on-disk shards that applies live transforms at load time.

.. automodule:: bioverse.data
   :members: Batch, Split, Assets
