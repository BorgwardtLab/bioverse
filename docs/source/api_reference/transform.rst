Transform
=========

A **transform** modifies batches, splits, and assets. Transforms follow a
scikit-learn-style API: optional :meth:`~bioverse.transform.Transform.fit` on the
full dataset, then :meth:`~bioverse.transform.Transform.transform` applied to each
shard. Transforms can be composed with :class:`~bioverse.transform.Compose`.

There are two execution modes:

- **Offline** transforms are applied via :meth:`~bioverse.dataset.Dataset.apply`
  and written to a new on-disk hash directory. Use these for expensive,
  dataset-wide operations (tokenization, graph construction, normalization).
- **Live** transforms are attached via :meth:`~bioverse.dataset.Dataset.live`
  and run at load time inside a benchmark loader. Use these for augmentations or
  featurization that should vary between epochs.

See :doc:`../implementations/transforms` for the full catalog.

.. automodule:: bioverse.transform
   :members: Transform, Compose
